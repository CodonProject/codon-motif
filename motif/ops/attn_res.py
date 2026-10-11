from typing import Optional, Sequence

import torch

from .autocast import autocast_safe

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:
    HAS_TRITON = False


MAX_SRC = 4


if HAS_TRITON:

    @triton.jit
    def _ar_logit(base, sb, pid_b, offs_t, offs_d, stride_t, mask2, t_mask, w, qv, D, eps):
        '''
        Attention-residual logit `q . (w * x)` for one (batch, time, channel) block.

        The tile is normalised by its own RMS, scaled by the weight and dotted with
        the pseudo-query. Masked times are set to `-inf` so the online softmax skips
        them.

        Args:
            base: One source tensor, `[B, T, D]`.
            sb: Batch stride of the source.
            pid_b: Batch index handled by this program.
            offs_t: Time offsets of the tile, `[BLOCK_T]`.
            offs_d: Channel offsets of the tile, `[BLOCK_D]`.
            stride_t: Time stride of the source.
            mask2: 2-D `[BLOCK_T, BLOCK_D]` load mask.
            t_mask: 1-D `[BLOCK_T]` mask that keeps the valid times.
            w: RMSNorm weight of length `D`.
            qv: `[D]` pseudo-query vector.
            D: Number of channels.
            eps: RMSNorm epsilon.

        Returns:
            tuple: `(s, v)` where `s` is the `[BLOCK_T]` logit and `v` the fp32 tile.
        '''
        ptrs = base + pid_b * sb + offs_t[:, None] * stride_t + offs_d[None, :]
        v = tl.load(ptrs, mask=mask2, other=0.0).to(tl.float32)
        ms = tl.sum(v * v, axis=1) / D
        rstd = 1.0 / tl.sqrt(ms + eps)
        k = v * rstd[:, None] * w[None, :]
        s = tl.sum(k * qv[None, :], axis=1)
        s = tl.where(t_mask, s, float('-inf'))
        return s, v

    @triton.jit
    def _ar_logit_dalpha(base, sb, pid_b, offs_t, offs_d, stride_t, mask2, t_mask, w, qv, dh, D, eps):
        '''
        Gradient of the attention-residual logit with respect to the softmax weight.

        Recomputes the same logit as `_ar_logit` and additionally contracts the
        upstream gradient `dh` with the source tile.

        Args:
            base: One source tensor, `[B, T, D]`.
            sb: Batch stride of the source.
            pid_b: Batch index handled by this program.
            offs_t: Time offsets of the tile, `[BLOCK_T]`.
            offs_d: Channel offsets of the tile, `[BLOCK_D]`.
            stride_t: Time stride of the source.
            mask2: 2-D `[BLOCK_T, BLOCK_D]` load mask.
            t_mask: 1-D `[BLOCK_T]` mask that keeps the valid times.
            w: RMSNorm weight of length `D`.
            qv: `[D]` pseudo-query vector.
            dh: `[BLOCK_T, BLOCK_D]` upstream gradient of the output.
            D: Number of channels.
            eps: RMSNorm epsilon.

        Returns:
            tuple: `(s, dalpha)` where `s` is the `[BLOCK_T]` logit and `dalpha` the
            gradient with respect to the softmax weight.
        '''
        ptrs = base + pid_b * sb + offs_t[:, None] * stride_t + offs_d[None, :]
        v = tl.load(ptrs, mask=mask2, other=0.0).to(tl.float32)
        ms = tl.sum(v * v, axis=1) / D
        rstd = 1.0 / tl.sqrt(ms + eps)
        k = v * rstd[:, None] * w[None, :]
        s = tl.sum(k * qv[None, :], axis=1)
        s = tl.where(t_mask, s, float('-inf'))
        dalpha = tl.sum(dh * v, axis=1)
        return s, dalpha

    @triton.jit
    def _attn_res_fwd_kernel(
        V0, V1, V2, V3, SB0, SB1, SB2, SB3,
        W, Q, Out, NSRC,
        stride_t, stride_ob, stride_ot,
        SEQ_LEN, D, eps,
        BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr,
    ):
        '''
        Fused forward kernel of the cross-block attention residual.

        Each program handles one time tile of one batch element and runs the online
        softmax over the live sources: the `NSRC`-dependent `if` chain reads a source
        only when it exists, and `m_i` / `l_i` / `acc` are rescaled after every
        source, so the sources are read once and never materialised together.

        Args:
            V0: First source, `[B, T, D]`.
            V1: Second source, unused when `NSRC <= 1`.
            V2: Third source, unused when `NSRC <= 2`.
            V3: Fourth source, unused when `NSRC <= 3`.
            SB0: Batch stride of the first source.
            SB1: Batch stride of the second source.
            SB2: Batch stride of the third source.
            SB3: Batch stride of the fourth source.
            W: RMSNorm weight of length `D`.
            Q: `[D]` pseudo-query vector.
            Out: Output `[B, T, D]`, in the dtype of the sources.
            NSRC: Number of live sources, at most `MAX_SRC`.
            stride_t: Time stride shared by every source.
            stride_ob: Batch stride of the output.
            stride_ot: Time stride of the output.
            SEQ_LEN: Sequence length `T`.
            D: Number of channels.
            eps: RMSNorm epsilon.
            BLOCK_D (tl.constexpr): Power-of-two block size over the channels.
            BLOCK_T (tl.constexpr): Time block size handled by one program.
        '''
        pid_t = tl.program_id(0)
        pid_b = tl.program_id(1).to(tl.int64)

        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = tl.arange(0, BLOCK_D)
        t_mask = offs_t < SEQ_LEN
        d_mask = offs_d < D
        mask2 = t_mask[:, None] & d_mask[None, :]

        w = tl.load(W + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        qv = tl.load(Q + offs_d, mask=d_mask, other=0.0).to(tl.float32)

        m_i = tl.full([BLOCK_T], float('-inf'), tl.float32)
        l_i = tl.zeros([BLOCK_T], tl.float32)
        acc = tl.zeros([BLOCK_T, BLOCK_D], tl.float32)

        if NSRC > 0:
            s, v = _ar_logit(V0, SB0, pid_b, offs_t, offs_d, stride_t,
                             mask2, t_mask, w, qv, D, eps)
            m_new = tl.maximum(m_i, s)
            p = tl.exp(tl.where(t_mask, s - m_new, 0.0))
            sc = tl.exp(tl.where(t_mask, m_i - m_new, 0.0))
            l_i = l_i * sc + p
            acc = acc * sc[:, None] + p[:, None] * v
            m_i = m_new
        if NSRC > 1:
            s, v = _ar_logit(V1, SB1, pid_b, offs_t, offs_d, stride_t,
                             mask2, t_mask, w, qv, D, eps)
            m_new = tl.maximum(m_i, s)
            p = tl.exp(tl.where(t_mask, s - m_new, 0.0))
            sc = tl.exp(tl.where(t_mask, m_i - m_new, 0.0))
            l_i = l_i * sc + p
            acc = acc * sc[:, None] + p[:, None] * v
            m_i = m_new
        if NSRC > 2:
            s, v = _ar_logit(V2, SB2, pid_b, offs_t, offs_d, stride_t,
                             mask2, t_mask, w, qv, D, eps)
            m_new = tl.maximum(m_i, s)
            p = tl.exp(tl.where(t_mask, s - m_new, 0.0))
            sc = tl.exp(tl.where(t_mask, m_i - m_new, 0.0))
            l_i = l_i * sc + p
            acc = acc * sc[:, None] + p[:, None] * v
            m_i = m_new
        if NSRC > 3:
            s, v = _ar_logit(V3, SB3, pid_b, offs_t, offs_d, stride_t,
                             mask2, t_mask, w, qv, D, eps)
            m_new = tl.maximum(m_i, s)
            p = tl.exp(tl.where(t_mask, s - m_new, 0.0))
            sc = tl.exp(tl.where(t_mask, m_i - m_new, 0.0))
            l_i = l_i * sc + p
            acc = acc * sc[:, None] + p[:, None] * v
            m_i = m_new

        h = acc / tl.maximum(l_i, 1e-20)[:, None]
        optrs = Out + pid_b * stride_ob + offs_t[:, None] * stride_ot + offs_d[None, :]
        tl.store(optrs, h, mask=mask2)

    @triton.jit
    def _attn_res_bwd_kernel(
        V0, V1, V2, V3, SB0, SB1, SB2, SB3,
        W, Q, dO, DV0, DV1, DV2, DV3, DW, DQ, NSRC,
        stride_t, stride_ob, stride_ot,
        SEQ_LEN, D, eps,
        BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr,
    ):
        '''
        Fused backward kernel of the cross-block attention residual.

        The logits are recomputed from the saved sources instead of being stored.
        The contribution of each source to `q` and `w` is accumulated in registers
        and flushed once per program with `tl.atomic_add`, because every time tile
        writes the same `[D]` vectors.

        Args:
            V0: First source, `[B, T, D]`.
            V1: Second source, unused when `NSRC <= 1`.
            V2: Third source, unused when `NSRC <= 2`.
            V3: Fourth source, unused when `NSRC <= 3`.
            SB0: Batch stride of the first source.
            SB1: Batch stride of the second source.
            SB2: Batch stride of the third source.
            SB3: Batch stride of the fourth source.
            W: RMSNorm weight of length `D`.
            Q: `[D]` pseudo-query vector.
            dO: Upstream gradient of the output, `[B, T, D]`.
            DV0: Value gradient of the first source, `[B, T, D]`.
            DV1: Value gradient of the second source, unused when `NSRC <= 1`.
            DV2: Value gradient of the third source, unused when `NSRC <= 2`.
            DV3: Value gradient of the fourth source, unused when `NSRC <= 3`.
            DW: `[D]` atomic accumulator of the weight gradient.
            DQ: `[D]` atomic accumulator of the query gradient.
            NSRC: Number of live sources, at most `MAX_SRC`.
            stride_t: Time stride shared by every source.
            stride_ob: Batch stride of the output gradient.
            stride_ot: Time stride of the output gradient.
            SEQ_LEN: Sequence length `T`.
            D: Number of channels.
            eps: RMSNorm epsilon.
            BLOCK_D (tl.constexpr): Power-of-two block size over the channels.
            BLOCK_T (tl.constexpr): Time block size handled by one program.
        '''
        pid_t = tl.program_id(0)
        pid_b = tl.program_id(1).to(tl.int64)

        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = tl.arange(0, BLOCK_D)
        t_mask = offs_t < SEQ_LEN
        d_mask = offs_d < D
        mask2 = t_mask[:, None] & d_mask[None, :]

        w = tl.load(W + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        qv = tl.load(Q + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        doptrs = dO + pid_b * stride_ob + offs_t[:, None] * stride_ot + offs_d[None, :]
        dh = tl.load(doptrs, mask=mask2, other=0.0).to(tl.float32)

        neg_inf = float('-inf')
        s0 = tl.full([BLOCK_T], neg_inf, tl.float32)
        s1 = tl.full([BLOCK_T], neg_inf, tl.float32)
        s2 = tl.full([BLOCK_T], neg_inf, tl.float32)
        s3 = tl.full([BLOCK_T], neg_inf, tl.float32)
        da0 = tl.zeros([BLOCK_T], tl.float32)
        da1 = tl.zeros([BLOCK_T], tl.float32)
        da2 = tl.zeros([BLOCK_T], tl.float32)
        da3 = tl.zeros([BLOCK_T], tl.float32)

        if NSRC > 0:
            s0, da0 = _ar_logit_dalpha(
                V0, SB0, pid_b, offs_t, offs_d, stride_t, mask2, t_mask,
                w, qv, dh, D, eps)
        if NSRC > 1:
            s1, da1 = _ar_logit_dalpha(
                V1, SB1, pid_b, offs_t, offs_d, stride_t, mask2, t_mask,
                w, qv, dh, D, eps)
        if NSRC > 2:
            s2, da2 = _ar_logit_dalpha(
                V2, SB2, pid_b, offs_t, offs_d, stride_t, mask2, t_mask,
                w, qv, dh, D, eps)
        if NSRC > 3:
            s3, da3 = _ar_logit_dalpha(
                V3, SB3, pid_b, offs_t, offs_d, stride_t, mask2, t_mask,
                w, qv, dh, D, eps)

        m = tl.maximum(tl.maximum(s0, s1), tl.maximum(s2, s3))
        m = tl.where(t_mask, m, 0.0)
        e0 = tl.exp(s0 - m)
        e1 = tl.exp(s1 - m)
        e2 = tl.exp(s2 - m)
        e3 = tl.exp(s3 - m)
        l = e0 + e1 + e2 + e3
        l = tl.maximum(l, 1e-20)
        a0, a1, a2, a3 = e0 / l, e1 / l, e2 / l, e3 / l

        c = a0 * da0 + a1 * da1 + a2 * da2 + a3 * da3

        ds0 = a0 * (da0 - c)
        ds1 = a1 * (da1 - c)
        ds2 = a2 * (da2 - c)
        ds3 = a3 * (da3 - c)

        dq_acc = tl.zeros([BLOCK_D], tl.float32)
        dw_acc = tl.zeros([BLOCK_D], tl.float32)

        if NSRC > 0:
            dv, dq_acc, dw_acc = _ar_dv(
                V0, SB0, DV0, pid_b, offs_t, offs_d, stride_t, mask2, t_mask,
                w, qv, dh, a0, ds0, D, eps, dq_acc, dw_acc)
        if NSRC > 1:
            dv, dq_acc, dw_acc = _ar_dv(
                V1, SB1, DV1, pid_b, offs_t, offs_d, stride_t, mask2, t_mask,
                w, qv, dh, a1, ds1, D, eps, dq_acc, dw_acc)
        if NSRC > 2:
            dv, dq_acc, dw_acc = _ar_dv(
                V2, SB2, DV2, pid_b, offs_t, offs_d, stride_t, mask2, t_mask,
                w, qv, dh, a2, ds2, D, eps, dq_acc, dw_acc)
        if NSRC > 3:
            dv, dq_acc, dw_acc = _ar_dv(
                V3, SB3, DV3, pid_b, offs_t, offs_d, stride_t, mask2, t_mask,
                w, qv, dh, a3, ds3, D, eps, dq_acc, dw_acc)

        tl.atomic_add(DQ + offs_d, dq_acc, mask=d_mask)
        tl.atomic_add(DW + offs_d, dw_acc, mask=d_mask)

    @triton.jit
    def _ar_dv(base, sb, dv_base, pid_b, offs_t, offs_d, stride_t, mask2, t_mask,
               w, qv, dh, alpha, ds, D, eps, dq_acc, dw_acc):
        '''
        Accumulate the gradient contribution of one source into the value tensor.

        Besides writing the source's value gradient, this folds the source's logit
        gradient into the running query and weight accumulators, so the caller can
        flush them with a single atomic add per program.

        Args:
            base: The source tensor, `[B, T, D]`.
            sb: Batch stride of the source.
            dv_base: Output holding the value gradient, `[B, T, D]`.
            pid_b: Batch index handled by this program.
            offs_t: Time offsets of the tile, `[BLOCK_T]`.
            offs_d: Channel offsets of the tile, `[BLOCK_D]`.
            stride_t: Time stride of the source.
            mask2: 2-D `[BLOCK_T, BLOCK_D]` load mask.
            t_mask: 1-D `[BLOCK_T]` mask that keeps the valid times.
            w: RMSNorm weight of length `D`.
            qv: `[D]` pseudo-query vector.
            dh: `[BLOCK_T, BLOCK_D]` upstream gradient of the output.
            alpha: `[BLOCK_T]` softmax weight of this source.
            ds: `[BLOCK_T]` gradient with respect to this source's logit.
            D: Number of channels.
            eps: RMSNorm epsilon.
            dq_acc: `[D]` running accumulator of the query gradient.
            dw_acc: `[D]` running accumulator of the weight gradient.

        Returns:
            tuple: `(dv, dq_acc, dw_acc)` with the value gradient and the updated
            query and weight accumulators.
        '''
        ptrs = base + pid_b * sb + offs_t[:, None] * stride_t + offs_d[None, :]
        v = tl.load(ptrs, mask=mask2, other=0.0).to(tl.float32)
        ms = tl.sum(v * v, axis=1) / D
        rstd = 1.0 / tl.sqrt(ms + eps)
        k = v * rstd[:, None] * w[None, :]

        dv = alpha[:, None] * dh + ds[:, None] * qv[None, :] * w[None, :] * rstd[:, None]
        drstd = tl.sum(ds[:, None] * qv[None, :] * w[None, :] * v, axis=1)
        dms = -0.5 * drstd * rstd * rstd * rstd
        dv += (dms * 2.0 / D)[:, None] * v

        dq_acc += tl.sum(ds[:, None] * k, axis=0)
        dw_acc += tl.sum((ds * rstd)[:, None] * v * qv[None, :], axis=0)

        dvptrs = (dv_base + pid_b * sb + offs_t[:, None] * stride_t
                  + offs_d[None, :])
        tl.store(dvptrs, dv, mask=mask2)
        return dv, dq_acc, dw_acc

    @autocast_safe
    class _AttnResFunction(torch.autograd.Function):
        '''
        Autograd wrapper of the fused Triton attn-res kernel, with a custom backward.

        The forward launches a grid over `(ceil(T / block_t), B)`. The backward
        recomputes the logits from the saved sources instead of keeping them, so the
        sources are the only large tensors saved for the backward.
        '''

        @staticmethod
        def forward(ctx, w, q, eps, block_t, *sources):
            '''
            Run the fused forward kernel and save what the backward needs.

            Fewer than `MAX_SRC` sources are padded up to `MAX_SRC` with a repeat of
            the first one so that the unrolled kernel always receives four pointers;
            the padding is never read because `NSRC` gates those branches.

            Args:
                ctx: Autograd context; stores the sources, the weight, the query and
                    the launch configuration.
                w: RMSNorm weight of length `D`.
                q: `[D]` pseudo-query vector.
                eps: RMSNorm epsilon.
                block_t: Time block size of the kernel.
                *sources: Between 1 and `MAX_SRC` aligned `[B, T, D]` sources.

            Returns:
                torch.Tensor: `h`, shaped `[B, T, D]`.
            '''
            n_src = len(sources)
            b, seq_len, d = sources[0].shape

            padded = list(sources) + [sources[0]] * (MAX_SRC - n_src)
            sbs = [t.stride(0) for t in padded]
            stride_t = sources[0].stride(1)

            out = torch.empty((b, seq_len, d), dtype=sources[0].dtype,
                              device=sources[0].device)
            grid = (triton.cdiv(seq_len, block_t), b)
            _attn_res_fwd_kernel[grid](
                *[t for t in padded], *sbs,
                w, q, out, n_src,
                stride_t, out.stride(0), out.stride(1),
                seq_len, d, eps,
                BLOCK_D=triton.next_power_of_2(d), BLOCK_T=block_t,
                num_warps=4,
            )

            ctx.save_for_backward(w, q, *sources)
            ctx.eps = eps
            ctx.block_t = block_t
            ctx.n_src = n_src
            ctx.strides = [t.stride(0) for t in sources]
            ctx.stride_t = stride_t
            return out

        @staticmethod
        def backward(ctx, grad_out):
            '''
            Backpropagate into the pseudo-query, the norm weight and every source.

            Args:
                ctx: Autograd context holding the sources and the launch
                    configuration.
                grad_out: Upstream gradient of `h`, `[B, T, D]`.

            Returns:
                tuple: `(dw, dq, None, None, *dvs)` matching the forward inputs; `dvs`
                holds one gradient per source, in the order they were passed.
            '''
            w, q = ctx.saved_tensors[0], ctx.saved_tensors[1]
            sources = list(ctx.saved_tensors[2:])
            n_src = ctx.n_src
            b, seq_len, d = sources[0].shape
            grad_out = grad_out.contiguous()

            padded = sources + [sources[0]] * (MAX_SRC - n_src)
            sbs = [t.stride(0) for t in padded]
            dvs = [
                torch.empty_strided(t.shape, t.stride(), dtype=t.dtype,
                                    device=t.device)
                for t in padded
            ]
            dw = torch.zeros(d, dtype=torch.float32, device=sources[0].device)
            dq = torch.zeros(d, dtype=torch.float32, device=sources[0].device)

            grid = (triton.cdiv(seq_len, ctx.block_t), b)
            _attn_res_bwd_kernel[grid](
                *padded, *sbs,
                w, q, grad_out, *dvs, dw, dq, n_src,
                ctx.stride_t, grad_out.stride(0), grad_out.stride(1),
                seq_len, d, ctx.eps,
                BLOCK_D=triton.next_power_of_2(d), BLOCK_T=ctx.block_t,
                num_warps=4,
            )
            return (dw.to(w.dtype), dq.to(q.dtype), None, None, *dvs[:n_src])


def can_use_triton(sources: Sequence[torch.Tensor],
                   weight: Optional[torch.Tensor],
                   query: torch.Tensor) -> bool:
    '''
    Whether the fused Triton attn-res kernel can handle this call.

    The fused path needs Triton, a weight tensor, between 1 and `MAX_SRC` sources
    that share one dtype out of `torch.float32` / `torch.float16` /
    `torch.bfloat16`, a CUDA weight of length `D`, a contiguous 1-D CUDA query of
    length `D`, and 3-D CUDA sources whose last dimension is contiguous, whose batch
    size matches the first source and whose time stride agrees across sources.

    Args:
        sources (Sequence[torch.Tensor]): Aligned `[B, T, D]` source representations.
        weight (Optional[torch.Tensor]): RMSNorm weight of length `D`; None disables
            the fused path.
        query (torch.Tensor): `[D]` pseudo-query vector.

    Returns:
        bool: True when the fused kernel may be used.
    '''
    if not HAS_TRITON or weight is None:
        return False
    if not (1 <= len(sources) <= MAX_SRC):
        return False
    d = sources[0].shape[-1]
    dtype = sources[0].dtype
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if any(t.dtype != dtype for t in sources):
        return False
    if query.dim() != 1 or query.shape[0] != d:
        return False
    if not query.is_cuda or not query.is_contiguous():
        return False
    if weight.numel() != d or not weight.is_cuda:
        return False
    stride_t = None
    for t in sources:
        if not t.is_cuda or t.dim() != 3:
            return False
        if t.stride(-1) != 1:
            return False
        if t.shape[-1] != d or t.shape[0] != sources[0].shape[0]:
            return False
        if stride_t is None:
            stride_t = t.stride(1)
        elif t.stride(1) != stride_t:
            return False
    return True


def attn_res_block(
    sources: Sequence[torch.Tensor],
    weight: torch.Tensor,
    query: torch.Tensor,
    eps: float,
    block_t: int = 8,
) -> torch.Tensor:
    '''
    Fused cross-block attention residual.

    Args:
        sources (Sequence[torch.Tensor]): Aligned `[B, T, D]` sources.
        weight (torch.Tensor): RMSNorm weight of length `D`.
        query (torch.Tensor): `[D]` pseudo-query vector.
        eps (float): RMSNorm epsilon.
        block_t (int): Time block size of the kernel. Defaults to 8.

    Returns:
        torch.Tensor: `h`, shaped `[B, T, D]`.
    '''
    return _AttnResFunction.apply(
        weight, query, float(eps), int(block_t), *sources
    )

import warnings

from motif.model.cache import AttnResCache

_warned_kernel = False
_warned_too_many = False


def _read_sources(
    sources: Sequence[torch.Tensor],
    query: torch.Tensor,
    norm,
) -> torch.Tensor:
    '''
    Read one list of sources with the fused kernel when possible, else eagerly.

    Args:
        sources (Sequence[torch.Tensor]): Aligned `[B, T, D]` sources.
        query (torch.Tensor): `[D]` pseudo-query vector.
        norm: The RMSNorm applied to the sources before they are scored.

    Returns:
        torch.Tensor: `h`, shaped `[B, T, D]`.
    '''
    global _warned_kernel, _warned_too_many

    if HAS_TRITON:
        weight = getattr(norm, 'weight', None)
        if can_use_triton(sources, weight, query):
            try:
                return attn_res_block(sources, weight, query, norm.eps)
            except Exception as exc:
                if not _warned_kernel:
                    _warned_kernel = True
                    warnings.warn(
                        f'the fused attn-res Triton kernel failed, falling back to the eager path: {exc!r}',
                        RuntimeWarning,
                    )
        elif len(sources) > MAX_SRC and not _warned_too_many:
            _warned_too_many = True
            warnings.warn(
                f'attn-res reads {len(sources)} sources, more than the {MAX_SRC} the '
                'fused Triton kernel is unrolled over: falling back to the eager path.',
                RuntimeWarning,
            )

    V = torch.stack(sources, dim=0)
    K = norm(V)
    logits = torch.einsum('d, n b t d -> n b t', query, K)
    alpha = logits.softmax(dim=0)
    h = torch.einsum('n b t, n b t d -> b t d', alpha, V)
    return h


def apply_block_attn_res(
    cache: AttnResCache,
    query: torch.Tensor,
    norm,
) -> torch.Tensor:
    '''
    Apply the cross-block attention residual over the cached representations.

    Every entry of the cache history is scored on its own, so the `i`-th decoder of a
    block reads the block input plus the `i` increments left behind by the decoders
    before it.

    Args:
        cache (AttnResCache): Provides the attention-residual history in
            `blocks` (internally trimmed to the tail positions of the current chunk).
        query (torch.Tensor): `[D]` pseudo-query vector.
        norm: The RMSNorm applied to the sources before they are scored.

    Returns:
        torch.Tensor: `h`, shaped `[B, T, D]`.
    '''
    sources = cache.aligned_values()

    if len(sources) == 1:
        return sources[0]

    return _read_sources(sources, query, norm)
