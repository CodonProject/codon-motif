from typing import NamedTuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

from .autocast import autocast_safe

__all__ = [
    'recurrent_gated_delta_rule',
    'chunk_gated_delta_rule',
    'chunk_gated_delta_rule_triton',
]


_SCAN_BACKEND = None
_SCAN_BACKEND_BWD = None


def recurrent_gated_delta_rule(q, k, v, beta, g, S=None):
    '''
    Recurrent (token-by-token) gated delta rule.

    The recurrence is evaluated with an explicit loop over the `L` time steps, so
    this function is the readable reference for the chunked variants rather than
    the fast path.

    Args:
        q (torch.Tensor): Queries, `[B, H, L, Dk]`.
        k (torch.Tensor): Keys, `[B, H, L, Dk]`.
        v (torch.Tensor): Values, `[B, H, L, Dv]`.
        beta (torch.Tensor): Write strength per step, `[B, H, L]`.
        g (torch.Tensor): Log decay per step, `[B, H, L]`.
        S (Optional[torch.Tensor]): Initial recurrent state, `[B, H, Dk, Dv]`;
            None starts from zeros. Defaults to None.

    Returns:
        tuple: `(output [B, H, L, Dv], final_state [B, H, Dk, Dv])`.
    '''
    B, H, L, Dk = q.shape
    Dv = v.shape[-1]
    if S is None:
        S = q.new_zeros(B, H, Dk, Dv)
    outs = []
    for t in range(L):
        S = S * g[:, :, t].exp()[..., None, None]
        kt, vt = k[:, :, t], v[:, :, t]
        bt = beta[:, :, t, None]
        v_old = (kt.unsqueeze(-2) @ S).squeeze(-2)
        delta = bt * (vt - v_old)
        S = S + kt.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((q[:, :, t].unsqueeze(-2) @ S).squeeze(-2))
    return torch.stack(outs, dim=2), S


class _Chunks(NamedTuple):
    '''
    The padded, chunked view of a gated delta-net input.

    Attributes:
        L (int): Original sequence length.
        Dk (int): Key dimension.
        Dv (int): Value dimension.
        C (int): Chunk length.
        N (int): Number of chunks after padding.
        q (torch.Tensor): Chunked queries, `[B, H, N, C, Dk]`.
        k (torch.Tensor): Chunked keys, `[B, H, N, C, Dk]`.
        g (torch.Tensor): Chunked log decay, `[B, H, N, C]`.
        decay_mask (torch.Tensor): `[C, C]` pairwise decay `exp(gamma_i - gamma_j)`.
        W (torch.Tensor): Chunked `W` factor, `[B, H, N, C, Dk]`.
        U (torch.Tensor): Chunked `U` factor, `[B, H, N, C, Dv]`.
    '''
    L: int
    Dk: int
    Dv: int
    C: int
    N: int
    q: torch.Tensor
    k: torch.Tensor
    g: torch.Tensor
    decay_mask: torch.Tensor
    W: torch.Tensor
    U: torch.Tensor


def _vecdot(a, b):
    '''
    Batched vector dot product along the last dimension.

    Args:
        a (torch.Tensor): Left operand, `[..., D]`.
        b (torch.Tensor): Right operand, `[..., D]`.

    Returns:
        torch.Tensor: The dot products, with the last dimension removed.
    '''
    return torch.linalg.vecdot(a, b)


@autocast_safe
class _PrepareChunksFunction(torch.autograd.Function):
    '''
    Autograd wrapper that builds the chunked factors `W` / `U` with a custom backward.

    `forward` pads the sequence to a whole number of chunks of length `C` and
    solves the chunk-local lower-triangular system that produces `W` and `U`;
    `backward` differentiates that solve and the `[C, C]` decay mask in closed
    form from the factors saved by the forward.
    '''

    @staticmethod
    def forward(ctx, q, k, v, beta, g, C):
        '''
        Pad the sequence into chunks and build `W`, `U` and the decay mask.

        Args:
            ctx: Autograd context; stores the chunked keys, values and `beta`,
                the factors `Z`, `G1` and `M_inv`, the decay mask, the padding
                length and the original sequence length.
            q (torch.Tensor): Queries, `[B, H, L, Dk]`.
            k (torch.Tensor): Keys, `[B, H, L, Dk]`.
            v (torch.Tensor): Values, `[B, H, L, Dv]`.
            beta (torch.Tensor): Write strength per step, `[B, H, L]`.
            g (torch.Tensor): Log decay per step, `[B, H, L]`.
            C (int): Chunk length.

        Returns:
            tuple: The chunked `q`, `k` and `g` (each padded up to `N` chunks),
            the `[C, C]` decay mask and the `W` / `U` factors.
        '''
        B, H, L, Dk = q.shape
        Dv = v.shape[-1]
        pad = (C - L % C) % C
        if pad:
            q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
            beta, g = (F.pad(t, (0, pad)) for t in (beta, g))
        N = (L + pad) // C

        beta_c = beta.reshape(B, H, N, C)

        k = k.reshape(B, H, N, C, Dk).contiguous()
        vc = v.reshape(B, H, N, C, Dv).contiguous()
        v_beta = vc * beta_c[..., None]
        k_beta = k * beta_c[..., None]
        g = g.reshape(B, H, N, C).contiguous().cumsum(-1)

        dev = q.device
        tril = torch.tril(torch.ones(C, C, dtype=torch.bool, device=dev))
        strict_tril = torch.tril(torch.ones(C, C, dtype=torch.bool, device=dev), diagonal=-1)

        decay_mask = (g[..., :, None] - g[..., None, :]).masked_fill(~tril, float("-inf")).exp()

        G1 = k_beta @ k.transpose(-1, -2)
        A = (-G1 * decay_mask).masked_fill(~strict_tril, 0.0)
        M = torch.eye(C, device=dev, dtype=A.dtype) - A

        M_inv = torch.linalg.solve_triangular(
            M, torch.eye(C, device=dev, dtype=A.dtype).expand_as(A),
            upper=False, unitriangular=True
        )
        E = g.exp()
        Z = k_beta * E[..., None]
        W = M_inv @ Z
        U = M_inv @ v_beta

        q = q.reshape(B, H, N, C, Dk).contiguous()
        ctx.save_for_backward(k, vc, v_beta, k_beta, g, beta_c, decay_mask, Z, G1, M_inv)
        ctx.strict = strict_tril
        ctx.pad = pad
        ctx.L = L
        return q, k, g, decay_mask, W, U

    @staticmethod
    def backward(ctx, dqc, dkc, dgc, ddecay, dW, dU):
        '''
        Backpropagate through the chunk preparation.

        The whole preparation is differentiated from the saved factors, so the
        triangular solve is never re-run.

        Args:
            ctx: Autograd context holding `k`, `v`, `v_beta`, `k_beta`, `g`,
                `beta_c`, `decay_mask`, `Z`, `G1` and `M_inv`.
            dqc: Upstream gradient of the chunked queries.
            dkc: Upstream gradient of the chunked keys.
            dgc: Upstream gradient of the chunked log decay, or None.
            ddecay: Upstream gradient of the decay mask, or None.
            dW: Upstream gradient of `W`, or None.
            dU: Upstream gradient of `U`, or None.

        Returns:
            tuple: `(dq, dk, dv, dbeta, dg, None)`, each sliced back to the
            original sequence length `L`.
        '''
        kc, vc, vbc, kbc, gc, beta_c, decay, Z, G1, Mi = ctx.saved_tensors
        strict, pad, L = ctx.strict, ctx.pad, ctx.L
        B, H, N, C, Dk = kc.shape
        Dv = vbc.shape[-1]
        M = B * H * N

        kc_ = kc.reshape(M, C, Dk)
        kbc_ = kbc.reshape(M, C, Dk)
        vb_ = vbc.reshape(M, C, Dv)
        Z_ = Z.reshape(M, C, Dk)
        g_ = gc.reshape(M, C)
        beta_ = beta_c.reshape(M, C)
        decay_ = decay.reshape(M, C, C)
        G1_ = G1.reshape(M, C, C)
        Mi_ = Mi.reshape(M, C, C)
        MiT = Mi_.transpose(-1, -2)

        zeros = lambda ref: torch.zeros_like(ref)
        dW_ = zeros(Z_) if dW is None else dW.reshape(M, C, Dk)
        dU_ = zeros(vb_) if dU is None else dU.reshape(M, C, Dv)
        dgc_ = zeros(g_) if dgc is None else dgc.reshape(M, C)
        ddecay_ = zeros(decay_) if ddecay is None else ddecay.reshape(M, C, C)

        E = g_.exp()

        dMi = torch.bmm(dW_, Z_.transpose(-1, -2))
        dMi.baddbmm_(dU_, vb_.transpose(-1, -2))
        dZ = torch.bmm(MiT, dW_)
        dvb = torch.bmm(MiT, dU_)

        dA = torch.bmm(torch.bmm(MiT, dMi), MiT)
        dA.masked_fill_(~strict, 0.0)

        ddecay_ = torch.addcmul(ddecay_, dA, G1_, value=-1.0)
        dG1 = dA.mul_(decay_).neg_()

        dkbc = torch.bmm(dG1, kc_)
        dkc_ = zeros(kc_) if dkc is None else dkc.reshape(M, C, Dk)
        dkc_ = torch.baddbmm(dkc_, dG1.transpose(-1, -2), kbc_)

        dkbc.addcmul_(dZ, E[..., None])
        dgc_ = dgc_.add(_vecdot(dZ, Z_))

        dS = ddecay_.mul_(decay_)
        dgc_ = dgc_.add_(dS.sum(-1)).sub_(dS.sum(-2))

        dg = dgc_.flip(-1).cumsum(-1).flip(-1)

        dbeta = _vecdot(dkbc, kc_).add_(_vecdot(dvb, vc.reshape(M, C, Dv)))

        dk = dkbc.mul_(beta_[..., None]).add_(dkc_)
        dv = dvb.mul_(beta_[..., None])

        n_l = N * C
        dq = dqc.reshape(B, H, n_l, Dk)[:, :, :L]
        dk = dk.reshape(B, H, n_l, Dk)[:, :, :L]
        dv = dv.reshape(B, H, n_l, Dv)[:, :, :L]
        dbeta = dbeta.reshape(B, H, n_l)[:, :, :L]
        dg = dg.reshape(B, H, n_l)[:, :, :L]
        return dq, dk, dv, dbeta, dg, None


def _prepare_chunks(q, k, v, beta, g, chunk_size):
    '''
    Pad the sequence and build the chunked factors `W` / `U` and the decay mask.

    Args:
        q (torch.Tensor): Queries, `[B, H, L, Dk]`.
        k (torch.Tensor): Keys, `[B, H, L, Dk]`.
        v (torch.Tensor): Values, `[B, H, L, Dv]`.
        beta (torch.Tensor): Write strength, `[B, H, L]`.
        g (torch.Tensor): Log decay, `[B, H, L]`.
        chunk_size (int): Chunk length.

    Returns:
        _Chunks: The chunked view of the input.
    '''
    B, H, L, Dk = q.shape
    Dv = v.shape[-1]
    C = chunk_size
    pad = (C - L % C) % C
    N = (L + pad) // C
    q_c, k_c, g_c, decay_mask, W, U = _PrepareChunksFunction.apply(q, k, v, beta, g, C)
    return _Chunks(L=L, Dk=Dk, Dv=Dv, C=C, N=N, q=q_c, k=k_c, g=g_c, decay_mask=decay_mask, W=W, U=U)


def chunk_gated_delta_rule(q, k, v, beta, g, S=None, chunk_size=64):
    '''
    Chunked gated delta rule in pure PyTorch.

    Args:
        q (torch.Tensor): Queries, `[B, H, L, Dk]`.
        k (torch.Tensor): Keys, `[B, H, L, Dk]`.
        v (torch.Tensor): Values, `[B, H, L, Dv]`.
        beta (torch.Tensor): Write strength per step, `[B, H, L]`.
        g (torch.Tensor): Log decay per step, `[B, H, L]`.
        S (Optional[torch.Tensor]): Initial recurrent state, `[B, H, Dk, Dv]`;
            None starts from zeros. Defaults to None.
        chunk_size (int): Chunk length. Defaults to 64.

    Returns:
        tuple: `(output [B, H, L, Dv], final_state [B, H, Dk, Dv])`.
    '''
    B, H = q.shape[0], q.shape[1]
    ch = _prepare_chunks(q, k, v, beta, g, chunk_size)
    q, k, g_chunk = ch.q, ch.k, ch.g
    decay_mask, W, U = ch.decay_mask, ch.W, ch.U

    if S is None:
        S = q.new_zeros(B, H, ch.Dk, ch.Dv)
    o = torch.empty(B, H, ch.N, ch.C, ch.Dv, device=q.device, dtype=q.dtype)
    for i in range(ch.N):
        qi, ki, gi = q[:, :, i], k[:, :, i], g_chunk[:, :, i]
        attn = (qi @ ki.transpose(-1, -2)) * decay_mask[:, :, i]
        u = U[:, :, i] - W[:, :, i] @ S
        o_inter = (qi * gi.exp()[..., None]) @ S
        o[:, :, i] = o_inter + attn @ u
        S = S * gi[:, :, -1].exp()[..., None, None] + \
            (ki * (gi[:, :, -1:] - gi).exp()[..., None]).transpose(-1, -2) @ u
    o = o.reshape(B, H, ch.N * ch.C, ch.Dv)[:, :, :ch.L]
    return o, S


if HAS_TRITON:

    @triton.jit
    def _gdn_fused_scan_kernel(
        K, W, U, G, S0, P, X, Sout,
        N,
        DK: tl.constexpr,
        DV: tl.constexpr,
        C: tl.constexpr,
        BLOCK_DK: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_DV: tl.constexpr,
        IS_FP32: tl.constexpr,
    ):
        '''
        Fused Triton kernel performing the chunk scan.

        Grid is `(B * H, cdiv(Dv, BLOCK_DV))`. Walking the chunks forward, the
        program stores the state *before* each chunk into `P`, the chunk-local
        residual `x = U - W @ S` into `X`, and the final state into `Sout`.

        Args:
            K: Chunked keys, `[B * H, N, C, Dk]`.
            W: Chunked `W` factor, `[B * H, N, C, Dk]`.
            U: Chunked `U` factor, `[B * H, N, C, Dv]`.
            G: Chunked log decay, `[B * H, N, C]`.
            S0: Initial recurrent state, `[B * H, Dk, Dv]`.
            P: Output prefix states, `[B * H, N, Dk, Dv]`.
            X: Output residuals, `[B * H, N, C, Dv]`.
            Sout: Output final state, `[B * H, Dk, Dv]`.
            N: Number of chunks.
            DK (tl.constexpr): Key dimension.
            DV (tl.constexpr): Value dimension.
            C (tl.constexpr): Chunk length.
            BLOCK_DK (tl.constexpr): Power-of-two block over the key dimension.
            BLOCK_C (tl.constexpr): Power-of-two block over the chunk length.
            BLOCK_DV (tl.constexpr): Power-of-two block over the value dimension.
            IS_FP32 (tl.constexpr): Whether the inputs are fp32; it selects
                `input_precision="tf32x3"` for the `tl.dot` calls.
        '''
        pid_bh = tl.program_id(0).to(tl.int64)
        pid_v = tl.program_id(1)

        offs_k = tl.arange(0, BLOCK_DK)
        offs_c = tl.arange(0, BLOCK_C)
        offs_v = pid_v * BLOCK_DV + tl.arange(0, BLOCK_DV)
        k_mask = offs_k < DK
        c_mask = offs_c < C
        v_mask = offs_v < DV

        offs_out = offs_k[:, None] * DV + offs_v[None, :]
        out_mask = k_mask[:, None] & v_mask[None, :]
        x_mask = c_mask[:, None] & v_mask[None, :]

        k_base = K + pid_bh * N * C * DK
        w_base = W + pid_bh * N * C * DK
        u_base = U + pid_bh * N * C * DV
        g_base = G + pid_bh * N * C
        p_base = P + pid_bh * N * DK * DV
        x_base = X + pid_bh * N * C * DV

        S = tl.load(S0 + pid_bh * DK * DV + offs_out, mask=out_mask, other=0.0)
        for i in range(N):
            tl.store(p_base + i * DK * DV + offs_out, S, mask=out_mask)

            glast = tl.load(g_base + i * C + C - 1)
            gi = tl.load(g_base + i * C + offs_c, mask=c_mask, other=0.0)
            decay = tl.exp(glast - gi)

            kT = tl.load(
                k_base + i * C * DK + offs_c[None, :] * DK + offs_k[:, None],
                mask=k_mask[:, None] & c_mask[None, :], other=0.0
            )
            mmatT = kT * decay[None, :]

            w = tl.load(
                w_base + i * C * DK + offs_c[:, None] * DK + offs_k[None, :],
                mask=c_mask[:, None] & k_mask[None, :], other=0.0
            )
            u = tl.load(
                u_base + i * C * DV + offs_c[:, None] * DV + offs_v[None, :],
                mask=c_mask[:, None] & v_mask[None, :], other=0.0
            )

            if IS_FP32:
                x = u - tl.dot(w, S, input_precision="tf32x3")
                tl.store(
                    x_base + i * C * DV + offs_c[:, None] * DV + offs_v[None, :],
                    x, mask=x_mask
                )
                S = tl.exp(glast) * S + tl.dot(mmatT, x, input_precision="tf32x3")
            else:
                x = u - tl.dot(w, S)
                tl.store(
                    x_base + i * C * DV + offs_c[:, None] * DV + offs_v[None, :],
                    x, mask=x_mask
                )
                S = tl.exp(glast) * S + tl.dot(mmatT, x)
        tl.store(Sout + pid_bh * DK * DV + offs_out, S, mask=out_mask)


if HAS_TRITON:

    @triton.jit
    def _gdn_bwd_scan_kernel(
        K, W, G, dX, dP, dS_final, dSn, dS0,
        N,
        DK: tl.constexpr,
        DV: tl.constexpr,
        C: tl.constexpr,
        BLOCK_DK: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_DV: tl.constexpr,
        HAS_DP: tl.constexpr,
        IS_FP32: tl.constexpr,
    ):
        '''
        Fused Triton kernel performing the backward chunk scan.

        Grid is `(B * H, cdiv(Dv, BLOCK_DV))`. Walking the chunks backwards, the
        program accumulates the gradient of the recurrent state and writes the
        per-chunk state gradient into `dSn`, the gradient of the initial state
        into `dS0`.

        Args:
            K: Chunked keys, `[B * H, N, C, Dk]`.
            W: Chunked `W` factor, `[B * H, N, C, Dk]`.
            G: Chunked log decay, `[B * H, N, C]`.
            dX: Upstream gradient of the residuals, `[B * H, N, C, Dv]`.
            dP: Upstream gradient of the prefix states, `[B * H, N, Dk, Dv]`;
                only read when `HAS_DP` is set.
            dS_final: Upstream gradient of the final state, `[B * H, Dk, Dv]`.
            dSn: Output per-chunk state gradients, `[B * H, N, Dk, Dv]`.
            dS0: Output gradient of the initial state, `[B * H, Dk, Dv]`.
            N: Number of chunks.
            DK (tl.constexpr): Key dimension.
            DV (tl.constexpr): Value dimension.
            C (tl.constexpr): Chunk length.
            BLOCK_DK (tl.constexpr): Power-of-two block over the key dimension.
            BLOCK_C (tl.constexpr): Power-of-two block over the chunk length.
            BLOCK_DV (tl.constexpr): Power-of-two block over the value dimension.
            HAS_DP (tl.constexpr): Whether `dP` is added to the accumulated state
                gradient.
            IS_FP32 (tl.constexpr): Whether the inputs are fp32; it selects
                `input_precision="tf32x3"` for the `tl.dot` calls.
        '''
        pid_bh = tl.program_id(0).to(tl.int64)
        pid_v = tl.program_id(1)
        offs_k = tl.arange(0, BLOCK_DK)
        offs_c = tl.arange(0, BLOCK_C)
        offs_v = pid_v * BLOCK_DV + tl.arange(0, BLOCK_DV)
        k_mask = offs_k < DK
        c_mask = offs_c < C
        v_mask = offs_v < DV
        offs_out = offs_k[:, None] * DV + offs_v[None, :]
        out_mask = k_mask[:, None] & v_mask[None, :]
        t_mask = c_mask[:, None] & k_mask[None, :]
        x_mask = c_mask[:, None] & v_mask[None, :]

        k_base = K + pid_bh * N * C * DK
        w_base = W + pid_bh * N * C * DK
        g_base = G + pid_bh * N * C
        dx_base = dX + pid_bh * N * C * DV
        dp_base = dP + pid_bh * N * DK * DV
        dsn_base = dSn + pid_bh * N * DK * DV

        gS = tl.load(dS_final + pid_bh * DK * DV + offs_out, mask=out_mask, other=0.0)
        for i in range(N - 1, -1, -1):
            tl.store(dsn_base + i * DK * DV + offs_out, gS, mask=out_mask)

            glast = tl.load(g_base + i * C + C - 1)
            gi = tl.load(g_base + i * C + offs_c, mask=c_mask, other=0.0)
            kk = tl.load(
                k_base + i * C * DK + offs_c[:, None] * DK + offs_k[None, :],
                mask=t_mask, other=0.0
            )
            mmat = kk * tl.exp(glast - gi)[:, None]
            dxin = tl.load(
                dx_base + i * C * DV + offs_c[:, None] * DV + offs_v[None, :],
                mask=x_mask, other=0.0
            )
            wT = tl.load(
                w_base + i * C * DK + offs_c[None, :] * DK + offs_k[:, None],
                mask=k_mask[:, None] & c_mask[None, :], other=0.0
            )
            if IS_FP32:
                dx = tl.dot(mmat, gS, input_precision="tf32x3") + dxin
                gS = tl.exp(glast) * gS - tl.dot(wT, dx, input_precision="tf32x3")
            else:
                dx = tl.dot(mmat, gS) + dxin
                gS = tl.exp(glast) * gS - tl.dot(wT, dx)
            if HAS_DP:
                gS = gS + tl.load(
                    dp_base + i * DK * DV + offs_out,
                    mask=out_mask, other=0.0
                )
        tl.store(dS0 + pid_bh * DK * DV + offs_out, gS, mask=out_mask)


def _bwd_scan_torch(k, W, g, dX, dP, dS_final):
    '''
    Pure-PyTorch backward scan of the chunk recurrence.

    Args:
        k (torch.Tensor): Chunked keys, `[B, H, N, C, Dk]`.
        W (torch.Tensor): Chunked `W` factor, `[B, H, N, C, Dk]`.
        g (torch.Tensor): Chunked log decay, `[B, H, N, C]`.
        dX (Optional[torch.Tensor]): Upstream gradient of the residuals, or None.
        dP (Optional[torch.Tensor]): Upstream gradient of the prefix states, or
            None.
        dS_final (torch.Tensor): Upstream gradient of the final state,
            `[B, H, Dk, Dv]`.

    Returns:
        tuple: `(dSn [B, H, N, Dk, Dv], dS0 [B, H, Dk, Dv])`.
    '''
    B, H, N, C, Dk = k.shape
    Dv = dS_final.shape[-1]
    dSn = torch.empty(B, H, N, Dk, Dv, device=k.device, dtype=k.dtype)
    gS = dS_final
    for i in range(N - 1, -1, -1):
        dSn[:, :, i] = gS
        gi = g[:, :, i]
        gl = gi[..., -1:]
        mmat = k[:, :, i] * (gl - gi).exp()[..., None]
        dx = mmat @ gS
        if dX is not None:
            dx = dx + dX[:, :, i]
        gS = gl.exp()[..., None] * gS - W[:, :, i].transpose(-1, -2) @ dx
        if dP is not None:
            gS = gS + dP[:, :, i]
    return dSn, gS


def _bwd_scan(k, W, g, dX, dP, dS_final):
    '''
    Backward chunk scan, using the fused Triton kernel when it is available.

    The fused kernel is only attempted for fp32 CUDA tensors; any failure falls
    back to `_bwd_scan_torch`, and `_SCAN_BACKEND_BWD` records which path ran.

    Args:
        k (torch.Tensor): Chunked keys, `[B, H, N, C, Dk]`.
        W (torch.Tensor): Chunked `W` factor, `[B, H, N, C, Dk]`.
        g (torch.Tensor): Chunked log decay, `[B, H, N, C]`.
        dX (Optional[torch.Tensor]): Upstream gradient of the residuals, or None.
        dP (Optional[torch.Tensor]): Upstream gradient of the prefix states, or
            None.
        dS_final (torch.Tensor): Upstream gradient of the final state,
            `[B, H, Dk, Dv]`.

    Returns:
        tuple: `(dSn [B, H, N, Dk, Dv], dS0 [B, H, Dk, Dv])`.
    '''
    global _SCAN_BACKEND_BWD
    if HAS_TRITON and k.dtype == torch.float32 and k.is_cuda:
        try:
            B, H, N, C, Dk = k.shape
            Dv = dS_final.shape[-1]
            device, dtype = k.device, k.dtype
            kf = k.reshape(B * H, N, C, Dk).contiguous()
            Wf = W.reshape(B * H, N, C, Dk).contiguous()
            gf = g.reshape(B * H, N, C).contiguous()
            dxf = (dX if dX is not None else k).reshape(B * H, N, C, Dv)
            dxf = dxf.contiguous() if dX is not None else dxf
            dpf = (dP if dP is not None else k).reshape(B * H, N, Dk, Dv)
            dpf = dpf.contiguous() if dP is not None else dpf
            dsf = dS_final.reshape(B * H, Dk, Dv).contiguous()
            dSn = torch.empty(B * H, N, Dk, Dv, dtype=dtype, device=device)
            dS0 = torch.empty(B * H, Dk, Dv, dtype=dtype, device=device)
            block_dk = triton.next_power_of_2(Dk)
            block_c = triton.next_power_of_2(C)
            block_dv = min(32, triton.next_power_of_2(Dv))
            per_stage = (block_dk * block_c * 2 + block_c * block_dv) * 4
            num_stages = 1 if per_stage > 32 * 1024 else 2
            _gdn_bwd_scan_kernel[(B * H, triton.cdiv(Dv, block_dv))](
                kf, Wf, gf, dxf, dpf, dsf, dSn, dS0, N,
                DK=Dk, DV=Dv, C=C,
                BLOCK_DK=block_dk, BLOCK_C=block_c, BLOCK_DV=block_dv,
                HAS_DP=dP is not None, IS_FP32=dtype == torch.float32,
                num_warps=4, num_stages=num_stages,
            )
            _SCAN_BACKEND_BWD = 'triton'
            return dSn.reshape(B, H, N, Dk, Dv), dS0.reshape(B, H, Dk, Dv)
        except Exception:
            pass
    _SCAN_BACKEND_BWD = 'torch'
    return _bwd_scan_torch(k, W, g, dX, dP, dS_final)


def _fused_scan_torch(k, W, U, g, S0):
    '''
    Pure-PyTorch chunk scan, used as the reference and as the fallback.

    Args:
        k (torch.Tensor): Chunked keys, `[B, H, N, C, Dk]`.
        W (torch.Tensor): Chunked `W` factor, `[B, H, N, C, Dk]`.
        U (torch.Tensor): Chunked `U` factor, `[B, H, N, C, Dv]`.
        g (torch.Tensor): Chunked log decay, `[B, H, N, C]`.
        S0 (torch.Tensor): Initial recurrent state, `[B, H, Dk, Dv]`.

    Returns:
        tuple: `(P [B, H, N, Dk, Dv], S [B, H, Dk, Dv], X [B, H, N, C, Dv])`.
    '''
    B, H, N, C, Dk = k.shape
    Dv = U.shape[-1]
    P = torch.empty(B, H, N, Dk, Dv, device=k.device, dtype=k.dtype)
    X = torch.empty(B, H, N, C, Dv, device=k.device, dtype=k.dtype)
    S = S0
    for i in range(N):
        P[:, :, i] = S
        gi = g[:, :, i]
        gl = gi[..., -1:]
        mmat = k[:, :, i] * (gl - gi).exp()[..., None]
        x = U[:, :, i] - W[:, :, i] @ S
        X[:, :, i] = x
        S = gl.exp()[..., None] * S + mmat.transpose(-1, -2) @ x
    return P, S, X


def _fused_scan_triton(k, W, U, g, S0):
    '''
    Chunk scan running on the fused Triton kernel.

    `num_stages` is dropped from 2 to 1 when the estimated per-stage buffer
    `(BLOCK_DK * BLOCK_C * 2 + BLOCK_C * BLOCK_DV) * 4` bytes exceeds 32 KiB.

    Args:
        k (torch.Tensor): Chunked keys, `[B, H, N, C, Dk]`.
        W (torch.Tensor): Chunked `W` factor, `[B, H, N, C, Dk]`.
        U (torch.Tensor): Chunked `U` factor, `[B, H, N, C, Dv]`.
        g (torch.Tensor): Chunked log decay, `[B, H, N, C]`.
        S0 (torch.Tensor): Initial recurrent state, `[B, H, Dk, Dv]`.

    Returns:
        tuple: `(P [B, H, N, Dk, Dv], Sout [B, H, Dk, Dv], X [B, H, N, C, Dv])`.
    '''
    B, H, N, C, Dk = k.shape
    Dv = U.shape[-1]
    device = k.device
    dtype = k.dtype

    kf = k.reshape(B * H, N, C, Dk).contiguous()
    Wf = W.reshape(B * H, N, C, Dk).contiguous()
    Uf = U.reshape(B * H, N, C, Dv).contiguous()
    gf = g.reshape(B * H, N, C).contiguous()
    S0f = S0.reshape(B * H, Dk, Dv).contiguous()
    P = torch.empty(B * H, N, Dk, Dv, dtype=dtype, device=device)
    X = torch.empty(B * H, N, C, Dv, dtype=dtype, device=device)
    Sout = torch.empty(B * H, Dk, Dv, dtype=dtype, device=device)

    block_dk = triton.next_power_of_2(Dk)
    block_c = triton.next_power_of_2(C)

    block_dv = min(32, triton.next_power_of_2(Dv))

    per_stage = (block_dk * block_c * 2 + block_c * block_dv) * 4
    num_stages = 1 if per_stage > 32 * 1024 else 2
    grid = (B * H, triton.cdiv(Dv, block_dv))
    _gdn_fused_scan_kernel[grid](
        kf, Wf, Uf, gf, S0f, P, X, Sout, N,
        DK=Dk, DV=Dv, C=C,
        BLOCK_DK=block_dk, BLOCK_C=block_c, BLOCK_DV=block_dv,
        IS_FP32=dtype == torch.float32,
        num_warps=4, num_stages=num_stages,
    )
    return (
        P.reshape(B, H, N, Dk, Dv), Sout.reshape(B, H, Dk, Dv),
        X.reshape(B, H, N, C, Dv)
    )


@autocast_safe
class _FusedScanFunction(torch.autograd.Function):
    '''
    Autograd wrapper of the fused Triton chunk scan, with a custom backward.

    `forward` returns the prefix states, the final state and the chunk-local
    residuals; `backward` reruns the scan backwards to build the state gradients.
    '''

    @staticmethod
    def forward(ctx, k, W, U, g, S0):
        '''
        Run the fused chunk scan and save what the backward needs.

        Args:
            ctx: Autograd context; stores the chunked keys, `W`, `U`, `g`, the
                prefix states `P` and the residuals `X`.
            k (torch.Tensor): Chunked keys, `[B, H, N, C, Dk]`.
            W (torch.Tensor): Chunked `W` factor, `[B, H, N, C, Dk]`.
            U (torch.Tensor): Chunked `U` factor, `[B, H, N, C, Dv]`.
            g (torch.Tensor): Chunked log decay, `[B, H, N, C]`.
            S0 (torch.Tensor): Initial recurrent state, `[B, H, Dk, Dv]`.

        Returns:
            tuple: `(P [B, H, N, Dk, Dv], S [B, H, Dk, Dv], X [B, H, N, C, Dv])`.
        '''
        P, S, X = _fused_scan_triton(k, W, U, g, S0)
        ctx.save_for_backward(k.contiguous(), W, U, g, P, X)
        return P, S, X

    @staticmethod
    def backward(ctx, dP, dS_final, dX):
        '''
        Backpropagate through the fused chunk scan.

        Args:
            ctx: Autograd context holding `k`, `W`, `U`, `g`, `P` and `X`.
            dP (Optional[torch.Tensor]): Upstream gradient of the prefix states,
                or None.
            dS_final (Optional[torch.Tensor]): Upstream gradient of the final
                state; None is replaced by zeros.
            dX (Optional[torch.Tensor]): Upstream gradient of the residuals, or
                None.

        Returns:
            tuple: `(dk, dW, dU, dg, dS0)`.
        '''
        k, W, U, g, P, X = ctx.saved_tensors
        B, H, N, C, Dk = k.shape
        Dv = U.shape[-1]
        if dS_final is None:
            dS_final = torch.zeros(B, H, Dk, Dv, dtype=k.dtype, device=k.device)

        dSn, dS0 = _bwd_scan(k, W, g, dX, dP, dS_final)

        gl = g[..., -1:]
        decay = (gl - g).exp()
        dMt = dSn @ X.transpose(-1, -2)

        dx = k @ dSn
        dx = dx.mul_(decay[..., None]) if dX is None else torch.addcmul(dX, dx, decay[..., None])
        dU = dx
        dW = (dx @ P.transpose(-1, -2)).neg_()
        dk = dMt.transpose(-1, -2) * decay[..., None]
        prod = (dk * k).transpose(-1, -2)
        dg = -prod.sum(dim=-2)

        corr = torch.linalg.vecdot(dSn.reshape(-1, Dk * Dv), P.reshape(-1, Dk * Dv)).reshape(B, H, N)
        dg[..., -1] += prod.sum(dim=(-1, -2)) + gl.exp().squeeze(-1) * corr
        return dk, dW, dU, dg, dS0


def _fused_scan(k, W, U, g, S0):
    '''
    Run the chunk scan, preferring the fused Triton kernel.

    Falls back to the pure-PyTorch scan when Triton is unavailable, the dtype is not
    fp32 or the inputs are not on CUDA; `_SCAN_BACKEND` records which path ran.

    Args:
        k (torch.Tensor): Chunked keys, `[B, H, N, C, Dk]`.
        W (torch.Tensor): Chunked `W` factor, `[B, H, N, C, Dk]`.
        U (torch.Tensor): Chunked `U` factor, `[B, H, N, C, Dv]`.
        g (torch.Tensor): Chunked log decay, `[B, H, N, C]`.
        S0 (torch.Tensor): Initial recurrent state.

    Returns:
        tuple: `(P, S_final, u)` as consumed by the chunked combination.
    '''
    global _SCAN_BACKEND
    if HAS_TRITON and k.dtype == torch.float32 and k.is_cuda:
        try:
            out = _FusedScanFunction.apply(k, W, U, g, S0)
            _SCAN_BACKEND = 'triton'
            return out
        except Exception: pass
    _SCAN_BACKEND = 'torch'
    return _fused_scan_torch(k, W, U, g, S0)


def chunk_gated_delta_rule_triton(q, k, v, beta, g, S=None, chunk_size=64):
    '''
    Chunked gated delta rule with the chunk scan in fused Triton kernels.

    Args:
        q (torch.Tensor): Queries, `[B, H, L, Dk]`.
        k (torch.Tensor): Keys, `[B, H, L, Dk]`.
        v (torch.Tensor): Values, `[B, H, L, Dv]`.
        beta (torch.Tensor): Write strength per step, `[B, H, L]`.
        g (torch.Tensor): Log decay per step, `[B, H, L]`.
        S (Optional[torch.Tensor]): Initial recurrent state, `[B, H, Dk, Dv]`;
            None starts from zeros. Defaults to None.
        chunk_size (int): Chunk length; it has to be at least 16 because `tl.dot`
            requires a K dimension of that size. Defaults to 64.

    Returns:
        tuple: `(output [B, H, L, Dv], final_state [B, H, Dk, Dv])`.

    Raises:
        NotImplementedError: If Triton is unavailable, or `chunk_size` is below 16.
    '''
    if not HAS_TRITON:
        raise NotImplementedError('Triton is not available for GatedDeltaNet chunk scan.')

    B, H = q.shape[0], q.shape[1]
    ch = _prepare_chunks(q, k, v, beta, g, chunk_size)
    if ch.C < 16:
        raise NotImplementedError('chunk_size must be >= 16 (the K-dimension lower bound of tl.dot)')

    if S is None:
        S = ch.q.new_zeros(B, H, ch.Dk, ch.Dv)

    P, S_final, u = _fused_scan(ch.k, ch.W, ch.U, ch.g, S)

    attn = (ch.q @ ch.k.transpose(-1, -2)) * ch.decay_mask
    o_inter = (ch.q * ch.g.exp()[..., None]) @ P
    o = o_inter + attn @ u

    return o.reshape(B, H, ch.N * ch.C, ch.Dv)[:, :, :ch.L], S_final
