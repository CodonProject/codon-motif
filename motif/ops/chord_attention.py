from typing import Optional

import torch

from .autocast import autocast_safe

try:
    import triton
    import triton.language as tl
    from triton.runtime.errors import OutOfResources

    HAS_TRITON = True
except Exception:
    HAS_TRITON = False
    OutOfResources = RuntimeError


if HAS_TRITON:

    _LOG2E = tl.constexpr(1.4426950408889634)
    _LN2 = tl.constexpr(0.6931471805599453)

    _FWD_CONFIGS_WIDE = [
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ]
    _FWD_CONFIGS_NARROW = [
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_warps=4, num_stages=1),
    ]
    _FWD_CONFIGS_WIDE_TAIL = [
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
    ]
    _BWD_DQ_CONFIGS_WIDE = [
        triton.Config({'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 32}, num_warps=4, num_stages=2),
    ]
    _BWD_DQ_CONFIGS_NARROW = [
        triton.Config({'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 32}, num_warps=4, num_stages=1),
    ]
    _BWD_DKDV_CONFIGS_WIDE = [
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 32}, num_warps=4, num_stages=2),
    ]
    _BWD_DKDV_CONFIGS_NARROW = [
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 16, 'BLOCK_M': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 32}, num_warps=4, num_stages=1),
    ]

    _TUNE_KEY = ['Q_LEN', 'KV_LEN', 'HEAD_DIM', 'GROUP_SIZE']

    @triton.jit
    def _chord_fwd_kernel(
        Q, K, V, ALPHA, BETA, MU, SSMAX, LOG_N, Out, LSE,
        stride_qb, stride_qh, stride_qm,
        stride_kb, stride_kh, stride_kn,
        stride_vb, stride_vh, stride_vn,
        stride_ob, stride_oh, stride_om,
        stride_lb, stride_lm,
        Q_LEN, KV_LEN, Q_START,
        sm_scale, eps,
        GROUP_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        PAD_D: tl.constexpr,
        FP32_INPUT: tl.constexpr,
        STORE_LSE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        '''
        Fused forward kernel: GGD-biased softmax attention over one (q, kv) tile.

        The logits add a gated geometric decay (GGD) bias
        `b2 = -(alpha * |d - mu| ** beta)` to the scaled `q . k`, all in base-2 log
        space because the online softmax uses `exp2`. `mul` folds the per-head
        `ssmax` logit scale and the `log_n` scaling into that bias, and causality is
        enforced by comparing the absolute query position `q_abs` with the key
        offset. When `FP32_INPUT` is set, `tl.dot` uses the three-pass `tf32x3`
        decomposition so fp32 tensors still take the Triton path.

        Args:
            Q: Queries, `[B, H, L, D]`.
            K: Keys, `[B, H_kv, S, D]`.
            V: Values, `[B, H_kv, S, D]`.
            ALPHA: `[H]` GGD decay scale.
            BETA: `[H]` GGD decay exponent.
            MU: `[H]` GGD preferred distance.
            SSMAX: `[H]` logit scale applied before the softmax.
            LOG_N: `[B, q_len]` fp32 scaling input.
            Out: Output `[B, H, L, D]`, in the query dtype.
            LSE: `[B, H, L]` fp32 log-sum-exp output, written only when `STORE_LSE`.
            stride_qb: Batch stride of `Q`.
            stride_qh: Head stride of `Q`.
            stride_qm: Query stride of `Q`.
            stride_kb: Batch stride of `K`.
            stride_kh: Head stride of `K`.
            stride_kn: Key stride of `K`.
            stride_vb: Batch stride of `V`.
            stride_vh: Head stride of `V`.
            stride_vn: Key stride of `V`.
            stride_ob: Batch stride of `Out`.
            stride_oh: Head stride of `Out`.
            stride_om: Query stride of `Out`.
            stride_lb: Batch stride of `LSE`.
            stride_lm: Query stride of `LSE`.
            Q_LEN: Number of query rows `L`.
            KV_LEN: Number of key / value rows `S`.
            Q_START: Absolute position of the first query, used for the GGD
                distances and the causal mask.
            sm_scale: Softmax scale applied to the raw `q . k`.
            eps: Epsilon of the GGD bias.
            GROUP_SIZE (tl.constexpr): Number of query heads sharing one kv head.
            HEAD_DIM (tl.constexpr): Real head dimension `D`.
            BLOCK_D (tl.constexpr): Padded head dimension, a power of two.
            PAD_D (tl.constexpr): Whether `BLOCK_D > HEAD_DIM`.
            FP32_INPUT (tl.constexpr): Whether `tf32x3` dots are needed for fp32
                inputs.
            STORE_LSE (tl.constexpr): Whether the log-sum-exp is written.
            BLOCK_M (tl.constexpr): Query rows handled by one program.
            BLOCK_N (tl.constexpr): Key rows handled per inner iteration.
        '''
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        q_mask = offs_m < Q_LEN

        qk_scale = sm_scale * _LOG2E
        kv_head = pid_h // GROUP_SIZE

        alpha_h = tl.load(ALPHA + pid_h)
        beta_h = tl.load(BETA + pid_h)
        mu_h = tl.load(MU + pid_h)
        log_n = tl.load(LOG_N + pid_b * stride_lb + offs_m * stride_lm,
                        mask=q_mask, other=0.0)
        mul = tl.load(SSMAX + pid_h) * log_n
        q_abs = offs_m + Q_START

        q_ptrs = (Q + pid_b * stride_qb + pid_h * stride_qh
                  + offs_m[:, None] * stride_qm + offs_d[None, :])
        if PAD_D:
            q_tile = tl.load(q_ptrs, mask=q_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
                             other=0.0)
        else:
            q_tile = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

        k_base = K + pid_b * stride_kb + kv_head * stride_kh + offs_d[:, None]
        v_base = V + pid_b * stride_vb + kv_head * stride_vh + offs_d[None, :]

        m_i = tl.full([BLOCK_M], float('-inf'), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

        hi = tl.minimum(KV_LEN, (pid_m + 1) * BLOCK_M + Q_START)

        for start_n in range(0, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < KV_LEN

            if PAD_D:
                k_tile = tl.load(k_base + offs_n[None, :] * stride_kn,
                                 mask=n_mask[None, :] & (offs_d[:, None] < HEAD_DIM),
                                 other=0.0)
            else:
                k_tile = tl.load(k_base + offs_n[None, :] * stride_kn,
                                 mask=n_mask[None, :], other=0.0)

            if FP32_INPUT:
                qk = tl.dot(q_tile, k_tile, input_precision='tf32x3')
            else:
                qk = tl.dot(q_tile, k_tile)
            qk = qk * qk_scale

            d = offs_n[None, :] - q_abs[:, None]
            logf = tl.log(tl.abs(d - mu_h) + eps)
            b2 = -(alpha_h * tl.exp(beta_h * logf)) * _LOG2E
            qk = (qk + b2) * mul[:, None]

            visible = q_abs[:, None] >= offs_n[None, :]
            qk = tl.where(visible & n_mask[None, :] & q_mask[:, None],
                          qk, float('-inf'))

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha_r = tl.exp2(m_i - m_ij)
            l_i = l_i * alpha_r + tl.sum(p, 1)
            acc = acc * alpha_r[:, None]

            if PAD_D:
                v_tile = tl.load(v_base + offs_n[:, None] * stride_vn,
                                 mask=n_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
                                 other=0.0)
            else:
                v_tile = tl.load(v_base + offs_n[:, None] * stride_vn,
                                 mask=n_mask[:, None], other=0.0)

            p = p.to(v_tile.dtype)
            if FP32_INPUT:
                acc = tl.dot(p, v_tile, acc, input_precision='tf32x3')
            else:
                acc = tl.dot(p, v_tile, acc)

            m_i = m_ij

        lse = m_i + tl.log2(l_i)
        if STORE_LSE:
            tl.store(LSE + (pid_b * tl.num_programs(1) + pid_h) * Q_LEN + offs_m,
                     lse, mask=q_mask)

        acc = acc / tl.maximum(l_i, 1e-20)[:, None]

        o_ptrs = (Out + pid_b * stride_ob + pid_h * stride_oh
                  + offs_m[:, None] * stride_om + offs_d[None, :])
        if PAD_D:
            tl.store(o_ptrs, acc.to(Out.dtype.element_ty),
                     mask=q_mask[:, None] & (offs_d[None, :] < HEAD_DIM))
        else:
            tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask[:, None])

    @triton.jit
    def _chord_fwd_split_kernel(
        Q, K, V, ALPHA, BETA, MU, SSMAX, LOG_N, OutP, Mpart, Lpart,
        stride_qb, stride_qh, stride_qm,
        stride_kb, stride_kh, stride_kn,
        stride_vb, stride_vh, stride_vn,
        stride_os, stride_ob, stride_oh, stride_om,
        stride_ps, stride_pb, stride_ph,
        stride_lb2, stride_lm2,
        Q_LEN, KV_LEN, Q_START,
        sm_scale, eps,
        GROUP_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        PAD_D: tl.constexpr,
        FP32_INPUT: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        '''
        Forward kernel writing the partial (unnormalised) results of a split range.

        Each program owns one `(query tile, split)` pair and only iterates the key
        range `[pid_s * chunk, min(hi_row, lo + chunk))`, so the key ranges of the
        splits are disjoint. It stores the raw `acc` together with the split's local
        `m_i` / `l_i`, and `_chord_merge_kernel` recombines them with the
        online-softmax correction `exp2(m - m_max)`.

        Args:
            Q: Queries, `[B, H, L, D]`.
            K: Keys, `[B, H_kv, S, D]`.
            V: Values, `[B, H_kv, S, D]`.
            ALPHA: `[H]` GGD decay scale.
            BETA: `[H]` GGD decay exponent.
            MU: `[H]` GGD preferred distance.
            SSMAX: `[H]` logit scale applied before the softmax.
            LOG_N: `[B, q_len]` fp32 scaling input.
            OutP: Partial outputs `[NUM_SPLITS, B, H, L, D]` in fp32.
            Mpart: Per-split `[NUM_SPLITS, B, H, L]` fp32 running maximum.
            Lpart: Per-split `[NUM_SPLITS, B, H, L]` fp32 running sum.
            stride_qb: Batch stride of `Q`.
            stride_qh: Head stride of `Q`.
            stride_qm: Query stride of `Q`.
            stride_kb: Batch stride of `K`.
            stride_kh: Head stride of `K`.
            stride_kn: Key stride of `K`.
            stride_vb: Batch stride of `V`.
            stride_vh: Head stride of `V`.
            stride_vn: Key stride of `V`.
            stride_os: Split stride of `OutP`.
            stride_ob: Batch stride of `OutP`.
            stride_oh: Head stride of `OutP`.
            stride_om: Query stride of `OutP`.
            stride_ps: Split stride of `Mpart` / `Lpart`.
            stride_pb: Batch stride of `Mpart` / `Lpart`.
            stride_ph: Head stride of `Mpart` / `Lpart`.
            stride_lb2: Batch stride of `LOG_N`.
            stride_lm2: Query stride of `LOG_N`.
            Q_LEN: Number of query rows `L`.
            KV_LEN: Number of key / value rows `S`.
            Q_START: Absolute position of the first query.
            sm_scale: Softmax scale applied to the raw `q . k`.
            eps: Epsilon of the GGD bias.
            GROUP_SIZE (tl.constexpr): Number of query heads sharing one kv head.
            HEAD_DIM (tl.constexpr): Real head dimension `D`.
            BLOCK_D (tl.constexpr): Padded head dimension, a power of two.
            PAD_D (tl.constexpr): Whether `BLOCK_D > HEAD_DIM`.
            FP32_INPUT (tl.constexpr): Whether `tf32x3` dots are needed.
            NUM_SPLITS (tl.constexpr): Number of key-range splits.
            BLOCK_M (tl.constexpr): Query rows handled by one program.
            BLOCK_N (tl.constexpr): Key rows handled per inner iteration.
        '''
        num_m = tl.num_programs(0) // NUM_SPLITS
        pid = tl.program_id(0)
        pid_m = pid % num_m
        pid_s = pid // num_m
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        q_mask = offs_m < Q_LEN

        qk_scale = sm_scale * _LOG2E
        kv_head = pid_h // GROUP_SIZE

        alpha_h = tl.load(ALPHA + pid_h)
        beta_h = tl.load(BETA + pid_h)
        mu_h = tl.load(MU + pid_h)
        log_n = tl.load(LOG_N + pid_b * stride_lb2 + offs_m * stride_lm2,
                        mask=q_mask, other=0.0)
        mul = tl.load(SSMAX + pid_h) * log_n
        q_abs = offs_m + Q_START

        q_ptrs = (Q + pid_b * stride_qb + pid_h * stride_qh
                  + offs_m[:, None] * stride_qm + offs_d[None, :])
        if PAD_D:
            q_tile = tl.load(q_ptrs, mask=q_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
                             other=0.0)
        else:
            q_tile = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

        k_base = K + pid_b * stride_kb + kv_head * stride_kh + offs_d[:, None]
        v_base = V + pid_b * stride_vb + kv_head * stride_vh + offs_d[None, :]

        m_i = tl.full([BLOCK_M], float('-inf'), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

        chunk = tl.cdiv(KV_LEN, NUM_SPLITS)
        lo = pid_s * chunk
        hi_row = tl.minimum(KV_LEN, (pid_m + 1) * BLOCK_M + Q_START)
        hi = tl.minimum(hi_row, lo + chunk)

        for start_n in range(lo, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < KV_LEN

            if PAD_D:
                k_tile = tl.load(k_base + offs_n[None, :] * stride_kn,
                                 mask=n_mask[None, :] & (offs_d[:, None] < HEAD_DIM),
                                 other=0.0)
            else:
                k_tile = tl.load(k_base + offs_n[None, :] * stride_kn,
                                 mask=n_mask[None, :], other=0.0)

            if FP32_INPUT:
                qk = tl.dot(q_tile, k_tile, input_precision='tf32x3')
            else:
                qk = tl.dot(q_tile, k_tile)
            qk = qk * qk_scale

            d = offs_n[None, :] - q_abs[:, None]
            logf = tl.log(tl.abs(d - mu_h) + eps)
            b2 = -(alpha_h * tl.exp(beta_h * logf)) * _LOG2E
            qk = (qk + b2) * mul[:, None]

            visible = q_abs[:, None] >= offs_n[None, :]
            qk = tl.where(visible & n_mask[None, :] & q_mask[:, None],
                          qk, float('-inf'))

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha_r = tl.exp2(m_i - m_ij)
            l_i = l_i * alpha_r + tl.sum(p, 1)
            acc = acc * alpha_r[:, None]

            if PAD_D:
                v_tile = tl.load(v_base + offs_n[:, None] * stride_vn,
                                 mask=n_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
                                 other=0.0)
            else:
                v_tile = tl.load(v_base + offs_n[:, None] * stride_vn,
                                 mask=n_mask[:, None], other=0.0)

            p = p.to(v_tile.dtype)
            if FP32_INPUT:
                acc = tl.dot(p, v_tile, acc, input_precision='tf32x3')
            else:
                acc = tl.dot(p, v_tile, acc)

            m_i = m_ij

        out_ptrs = (OutP + pid_s * stride_os + pid_b * stride_ob + pid_h * stride_oh
                    + offs_m[:, None] * stride_om + offs_d[None, :])
        if PAD_D:
            o_mask = q_mask[:, None] & (offs_d[None, :] < HEAD_DIM)
        else:
            o_mask = q_mask[:, None]
        tl.store(out_ptrs, acc, mask=o_mask)
        part = pid_s * stride_ps + pid_b * stride_pb + pid_h * stride_ph + offs_m
        tl.store(Mpart + part, m_i, mask=q_mask)
        tl.store(Lpart + part, l_i, mask=q_mask)

    @triton.jit
    def _chord_merge_kernel(
        OutP, Mpart, Lpart, Out, LSE,
        stride_os, stride_ob, stride_oh, stride_om,
        stride_ps, stride_pb, stride_ph,
        stride_ob2, stride_oh2, stride_om2,
        Q_LEN,
        NUM_SPLITS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        PAD_D: tl.constexpr,
        STORE_LSE: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        '''
        Merge the partial results produced by the split forward kernels.

        The per-split maxima are reduced first, then every partial output is rescaled
        by `exp2(m - m_max)` and accumulated, which is the usual online-softmax merge
        performed in a second pass over the splits.

        Args:
            OutP: Partial outputs `[NUM_SPLITS, B, H, L, D]` in fp32.
            Mpart: Per-split `[NUM_SPLITS, B, H, L]` fp32 running maximum.
            Lpart: Per-split `[NUM_SPLITS, B, H, L]` fp32 running sum.
            Out: Output `[B, H, L, D]`, in the query dtype.
            LSE: `[B, H, L]` fp32 log-sum-exp output, written only when `STORE_LSE`.
            stride_os: Split stride of `OutP`.
            stride_ob: Batch stride of `OutP`.
            stride_oh: Head stride of `OutP`.
            stride_om: Query stride of `OutP`.
            stride_ps: Split stride of `Mpart` / `Lpart`.
            stride_pb: Batch stride of `Mpart` / `Lpart`.
            stride_ph: Head stride of `Mpart` / `Lpart`.
            stride_ob2: Batch stride of `Out`.
            stride_oh2: Head stride of `Out`.
            stride_om2: Query stride of `Out`.
            Q_LEN: Number of query rows `L`.
            NUM_SPLITS (tl.constexpr): Number of key-range splits.
            HEAD_DIM (tl.constexpr): Real head dimension `D`.
            BLOCK_D (tl.constexpr): Padded head dimension, a power of two.
            PAD_D (tl.constexpr): Whether `BLOCK_D > HEAD_DIM`.
            STORE_LSE (tl.constexpr): Whether the log-sum-exp is written.
            BLOCK_M (tl.constexpr): Query rows handled by one program.
        '''
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        q_mask = offs_m < Q_LEN

        row = pid_b * stride_pb + pid_h * stride_ph + offs_m
        m_max = tl.full([BLOCK_M], float('-inf'), tl.float32)
        for s in tl.static_range(NUM_SPLITS):
            m = tl.load(Mpart + s * stride_ps + row, mask=q_mask, other=float('-inf'))
            m_max = tl.maximum(m_max, m)

        acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
        denom = tl.zeros([BLOCK_M], tl.float32)
        for s in tl.static_range(NUM_SPLITS):
            m = tl.load(Mpart + s * stride_ps + row, mask=q_mask, other=float('-inf'))
            l = tl.load(Lpart + s * stride_ps + row, mask=q_mask, other=0.0)
            w = tl.where(q_mask, tl.exp2(m - m_max), 0.0)
            a_ptrs = (OutP + s * stride_os + pid_b * stride_ob + pid_h * stride_oh
                      + offs_m[:, None] * stride_om + offs_d[None, :])
            if PAD_D:
                a = tl.load(a_ptrs, mask=q_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
                            other=0.0)
            else:
                a = tl.load(a_ptrs, mask=q_mask[:, None], other=0.0)
            acc += a * w[:, None]
            denom += w * l

        acc = acc / tl.maximum(denom, 1e-20)[:, None]

        o_ptrs = (Out + pid_b * stride_ob2 + pid_h * stride_oh2
                  + offs_m[:, None] * stride_om2 + offs_d[None, :])
        if PAD_D:
            tl.store(o_ptrs, acc.to(Out.dtype.element_ty),
                     mask=q_mask[:, None] & (offs_d[None, :] < HEAD_DIM))
        else:
            tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask[:, None])
        if STORE_LSE:
            tl.store(LSE + (pid_b * tl.num_programs(1) + pid_h) * Q_LEN + offs_m,
                     m_max + tl.log2(denom), mask=q_mask)

    _chord_fwd_wide = triton.autotune(configs=_FWD_CONFIGS_WIDE, key=_TUNE_KEY, warmup=10, rep=30)(
        _chord_fwd_kernel)
    _chord_fwd_narrow = triton.autotune(configs=_FWD_CONFIGS_NARROW, key=_TUNE_KEY, warmup=10, rep=30)(
        _chord_fwd_kernel)
    _chord_fwd_split_wide = triton.autotune(configs=_FWD_CONFIGS_WIDE_TAIL,
                                            key=_TUNE_KEY, warmup=10, rep=30)(_chord_fwd_split_kernel)
    _chord_fwd_split_narrow = triton.autotune(configs=_FWD_CONFIGS_NARROW,
                                              key=_TUNE_KEY, warmup=10, rep=30)(_chord_fwd_split_kernel)

    @triton.jit
    def _chord_bwd_dq_kernel(
        Q, K, V, dO, LSE, Delta, ALPHA, BETA, MU, SSMAX, LOG_N, dQ,
        DALPHA, DBETA, DMU, DSSMAX,
        stride_qb, stride_qh, stride_qm,
        stride_kb, stride_kh, stride_kn,
        stride_vb, stride_vh, stride_vn,
        stride_dob, stride_doh, stride_dom,
        stride_dqb, stride_dqh, stride_dqm,
        stride_lb, stride_lm,
        Q_LEN, KV_LEN, Q_START,
        sm_scale, eps,
        GROUP_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        PAD_D: tl.constexpr,
        FP32_INPUT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        '''
        Backward kernel producing the query gradient.

        Besides `dQ`, each program accumulates the gradients of the per-head GGD
        parameters in registers (`dth_alpha`, `dth_beta`, `dth_mu`) and of the logit
        scale (`dssmax`), then writes them into one slot per
        `(query tile, head, batch)` of the four `partial` planes; the caller sums
        those slots to finish the reduction. The per-head parameters are shared by
        every query row of a head, so they cannot be reduced inside this kernel.

        Args:
            Q: Queries, `[B, H, L, D]`.
            K: Keys, `[B, H_kv, S, D]`.
            V: Values, `[B, H_kv, S, D]`.
            dO: Upstream gradient of the output, `[B, H, L, D]`.
            LSE: `[B, H, L]` fp32 log-sum-exp saved by the forward pass.
            Delta: `[B, H, L]` fp32 row-wise `sum_d out * dout`.
            ALPHA: `[H]` GGD decay scale.
            BETA: `[H]` GGD decay exponent.
            MU: `[H]` GGD preferred distance.
            SSMAX: `[H]` logit scale applied before the softmax.
            LOG_N: `[B, q_len]` fp32 scaling input.
            dQ: Query-gradient output `[B, H, L, D]`.
            DALPHA: `[grid_m, H, B]` fp32 partial gradient of `alpha`.
            DBETA: `[grid_m, H, B]` fp32 partial gradient of `beta`.
            DMU: `[grid_m, H, B]` fp32 partial gradient of `mu`.
            DSSMAX: `[grid_m, H, B]` fp32 partial gradient of `ssmax`.
            stride_qb: Batch stride of `Q`.
            stride_qh: Head stride of `Q`.
            stride_qm: Query stride of `Q`.
            stride_kb: Batch stride of `K`.
            stride_kh: Head stride of `K`.
            stride_kn: Key stride of `K`.
            stride_vb: Batch stride of `V`.
            stride_vh: Head stride of `V`.
            stride_vn: Key stride of `V`.
            stride_dob: Batch stride of `dO`.
            stride_doh: Head stride of `dO`.
            stride_dom: Query stride of `dO`.
            stride_dqb: Batch stride of `dQ`.
            stride_dqh: Head stride of `dQ`.
            stride_dqm: Query stride of `dQ`.
            stride_lb: Batch stride of `LOG_N`.
            stride_lm: Query stride of `LOG_N`.
            Q_LEN: Number of query rows `L`.
            KV_LEN: Number of key / value rows `S`.
            Q_START: Absolute position of the first query.
            sm_scale: Softmax scale applied to the raw `q . k`.
            eps: Epsilon of the GGD bias.
            GROUP_SIZE (tl.constexpr): Number of query heads sharing one kv head.
            HEAD_DIM (tl.constexpr): Real head dimension `D`.
            BLOCK_D (tl.constexpr): Padded head dimension, a power of two.
            PAD_D (tl.constexpr): Whether `BLOCK_D > HEAD_DIM`.
            FP32_INPUT (tl.constexpr): Whether `tf32x3` dots are needed.
            BLOCK_M (tl.constexpr): Query rows handled by one program.
            BLOCK_N (tl.constexpr): Key rows handled per inner iteration.
        '''
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)
        num_heads = tl.num_programs(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        q_mask = offs_m < Q_LEN

        qk_scale = sm_scale * _LOG2E
        kv_head = pid_h // GROUP_SIZE

        alpha_h = tl.load(ALPHA + pid_h)
        beta_h = tl.load(BETA + pid_h)
        mu_h = tl.load(MU + pid_h)
        s_h = tl.load(SSMAX + pid_h)
        log_n = tl.load(LOG_N + pid_b * stride_lb + offs_m * stride_lm,
                        mask=q_mask, other=0.0)
        mul = s_h * log_n
        q_abs = offs_m + Q_START

        q_ptrs = (Q + pid_b * stride_qb + pid_h * stride_qh
                  + offs_m[:, None] * stride_qm + offs_d[None, :])
        do_ptrs = (dO + pid_b * stride_dob + pid_h * stride_doh
                   + offs_m[:, None] * stride_dom + offs_d[None, :])
        if PAD_D:
            d_mask_2d = offs_d[None, :] < HEAD_DIM
            q_tile = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask_2d, other=0.0)
            do_tile = tl.load(do_ptrs, mask=q_mask[:, None] & d_mask_2d, other=0.0)
        else:
            q_tile = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
            do_tile = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0)

        row = (pid_b * num_heads + pid_h) * Q_LEN + offs_m
        lse = tl.load(LSE + row, mask=q_mask, other=0.0)
        delta = tl.load(Delta + row, mask=q_mask, other=0.0)

        k_base = K + pid_b * stride_kb + kv_head * stride_kh + offs_d[:, None]
        v_base = V + pid_b * stride_vb + kv_head * stride_vh + offs_d[None, :]

        dq = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
        dth_alpha = 0.0
        dth_beta = 0.0
        dth_mu = 0.0
        dssmax = 0.0

        hi = tl.minimum(KV_LEN, (pid_m + 1) * BLOCK_M + Q_START)

        for start_n in range(0, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < KV_LEN

            if PAD_D:
                k_tile = tl.load(k_base + offs_n[None, :] * stride_kn,
                                 mask=n_mask[None, :] & (offs_d[:, None] < HEAD_DIM),
                                 other=0.0)
                v_tile = tl.load(v_base + offs_n[:, None] * stride_vn,
                                 mask=n_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
                                 other=0.0)
            else:
                k_tile = tl.load(k_base + offs_n[None, :] * stride_kn,
                                 mask=n_mask[None, :], other=0.0)
                v_tile = tl.load(v_base + offs_n[:, None] * stride_vn,
                                 mask=n_mask[:, None], other=0.0)

            if FP32_INPUT:
                qk = tl.dot(q_tile, k_tile, input_precision='tf32x3')
            else:
                qk = tl.dot(q_tile, k_tile)
            qk = qk * qk_scale

            d = offs_n[None, :] - q_abs[:, None]
            u = tl.abs(d - mu_h) + eps
            F = tl.exp(beta_h * tl.log(u))
            b2 = -(alpha_h * F) * _LOG2E

            logits = (qk + b2) * mul[:, None]
            p = tl.exp2(logits - lse[:, None])
            visible = q_abs[:, None] >= offs_n[None, :]
            keep = visible & n_mask[None, :] & q_mask[:, None]
            p = tl.where(keep, p, 0.0)

            if FP32_INPUT:
                dp = tl.dot(do_tile, tl.trans(v_tile), input_precision='tf32x3')
            else:
                dp = tl.dot(do_tile, tl.trans(v_tile))

            ds = p * (dp - delta[:, None]) * _LN2

            dmul = ds * mul[:, None]
            dth_alpha += tl.sum(dmul * (-F * _LOG2E))
            dth_beta += tl.sum(dmul * (b2 * tl.log(u)))
            sgn = tl.where(d - mu_h > 0, 1.0, tl.where(d - mu_h < 0, -1.0, 0.0))
            dth_mu += tl.sum(dmul * (alpha_h * beta_h * F * sgn * _LOG2E / u))
            dssmax += tl.sum(ds * (qk + b2) * log_n[:, None])

            dl_cast = (dmul * qk_scale).to(q_tile.dtype)
            if FP32_INPUT:
                dq = tl.dot(dl_cast, tl.trans(k_tile), dq, input_precision='tf32x3')
            else:
                dq = tl.dot(dl_cast, tl.trans(k_tile), dq)

        slot = (pid_m * tl.num_programs(1) + pid_h) * tl.num_programs(2) + pid_b
        tl.store(DALPHA + slot, dth_alpha)
        tl.store(DBETA + slot, dth_beta)
        tl.store(DMU + slot, dth_mu)
        tl.store(DSSMAX + slot, dssmax)

        dq_ptrs = (dQ + pid_b * stride_dqb + pid_h * stride_dqh
                   + offs_m[:, None] * stride_dqm + offs_d[None, :])
        if PAD_D:
            tl.store(dq_ptrs, dq.to(dQ.dtype.element_ty),
                     mask=q_mask[:, None] & (offs_d[None, :] < HEAD_DIM))
        else:
            tl.store(dq_ptrs, dq.to(dQ.dtype.element_ty), mask=q_mask[:, None])

    @triton.jit
    def _chord_delta_kernel(
        Out, dO, Delta,
        stride_ob, stride_oh, stride_om,
        stride_dob, stride_doh, stride_dom,
        Q_LEN,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        PAD_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        '''
        Kernel computing the row-wise delta used by the backward pass.

        `delta = sum_d out * dout` is the correction term of the softmax backward; it
        is materialised once instead of being recomputed in both backward kernels.

        Args:
            Out: Output of the forward pass, `[B, H, L, D]`.
            dO: Upstream gradient of the output, `[B, H, L, D]`.
            Delta: `[B, H, L]` fp32 output holding the row-wise delta.
            stride_ob: Batch stride of `Out`.
            stride_oh: Head stride of `Out`.
            stride_om: Query stride of `Out`.
            stride_dob: Batch stride of `dO`.
            stride_doh: Head stride of `dO`.
            stride_dom: Query stride of `dO`.
            Q_LEN: Number of query rows `L`.
            HEAD_DIM (tl.constexpr): Real head dimension `D`.
            BLOCK_D (tl.constexpr): Padded head dimension, a power of two.
            PAD_D (tl.constexpr): Whether `BLOCK_D > HEAD_DIM`.
            BLOCK_M (tl.constexpr): Query rows handled by one program.
        '''
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)
        num_heads = tl.num_programs(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        m_mask = offs_m < Q_LEN
        if PAD_D:
            mask = m_mask[:, None] & (offs_d[None, :] < HEAD_DIM)
        else:
            mask = m_mask[:, None]

        o_ptrs = (Out + pid_b * stride_ob + pid_h * stride_oh
                  + offs_m[:, None] * stride_om + offs_d[None, :])
        do_ptrs = (dO + pid_b * stride_dob + pid_h * stride_doh
                   + offs_m[:, None] * stride_dom + offs_d[None, :])
        o = tl.load(o_ptrs, mask=mask, other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=mask, other=0.0).to(tl.float32)

        delta = tl.sum(o * do, axis=1)
        tl.store(Delta + (pid_b * num_heads + pid_h) * Q_LEN + offs_m,
                 delta, mask=m_mask)

    @triton.jit
    def _chord_bwd_dkdv_kernel(
        Q, K, V, dO, LSE, Delta, ALPHA, BETA, MU, SSMAX, LOG_N, dK, dV,
        stride_qb, stride_qh, stride_qm,
        stride_kb, stride_kh, stride_kn,
        stride_vb, stride_vh, stride_vn,
        stride_dob, stride_doh, stride_dom,
        stride_dkb, stride_dkh, stride_dkn,
        stride_dvb, stride_dvh, stride_dvn,
        stride_lb, stride_lm,
        Q_LEN, KV_LEN, Q_START,
        sm_scale, eps,
        GROUP_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        PAD_D: tl.constexpr,
        FP32_INPUT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        '''
        Backward kernel producing the key and value gradients.

        One program owns a key tile of one kv head and walks the `GROUP_SIZE` query
        heads that share it (`for g in tl.static_range(GROUP_SIZE)`), accumulating
        `dk` / `dv` across them. `lo` starts at the first query tile that can see this
        key tile, so the zero blocks below the causal diagonal are never visited.

        Args:
            Q: Queries, `[B, H, L, D]`.
            K: Keys, `[B, H_kv, S, D]`.
            V: Values, `[B, H_kv, S, D]`.
            dO: Upstream gradient of the output, `[B, H, L, D]`.
            LSE: `[B, H, L]` fp32 log-sum-exp saved by the forward pass.
            Delta: `[B, H, L]` fp32 row-wise `sum_d out * dout`.
            ALPHA: `[H]` GGD decay scale.
            BETA: `[H]` GGD decay exponent.
            MU: `[H]` GGD preferred distance.
            SSMAX: `[H]` logit scale applied before the softmax.
            LOG_N: `[B, q_len]` fp32 scaling input.
            dK: Key-gradient output `[B, H_kv, S, D]`.
            dV: Value-gradient output `[B, H_kv, S, D]`.
            stride_qb: Batch stride of `Q`.
            stride_qh: Head stride of `Q`.
            stride_qm: Query stride of `Q`.
            stride_kb: Batch stride of `K`.
            stride_kh: Head stride of `K`.
            stride_kn: Key stride of `K`.
            stride_vb: Batch stride of `V`.
            stride_vh: Head stride of `V`.
            stride_vn: Key stride of `V`.
            stride_dob: Batch stride of `dO`.
            stride_doh: Head stride of `dO`.
            stride_dom: Query stride of `dO`.
            stride_dkb: Batch stride of `dK`.
            stride_dkh: Head stride of `dK`.
            stride_dkn: Key stride of `dK`.
            stride_dvb: Batch stride of `dV`.
            stride_dvh: Head stride of `dV`.
            stride_dvn: Key stride of `dV`.
            stride_lb: Batch stride of `LOG_N`.
            stride_lm: Query stride of `LOG_N`.
            Q_LEN: Number of query rows `L`.
            KV_LEN: Number of key / value rows `S`.
            Q_START: Absolute position of the first query.
            sm_scale: Softmax scale applied to the raw `q . k`.
            eps: Epsilon of the GGD bias.
            GROUP_SIZE (tl.constexpr): Number of query heads sharing one kv head.
            HEAD_DIM (tl.constexpr): Real head dimension `D`.
            BLOCK_D (tl.constexpr): Padded head dimension, a power of two.
            PAD_D (tl.constexpr): Whether `BLOCK_D > HEAD_DIM`.
            FP32_INPUT (tl.constexpr): Whether `tf32x3` dots are needed.
            BLOCK_M (tl.constexpr): Query rows handled per inner iteration.
            BLOCK_N (tl.constexpr): Key rows handled by one program.
        '''
        pid_n = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)
        num_heads = tl.num_programs(1) * GROUP_SIZE

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        n_mask = offs_n < KV_LEN

        qk_scale = sm_scale * _LOG2E

        k_base = K + pid_b * stride_kb + pid_h * stride_kh + offs_d[:, None]
        v_base = V + pid_b * stride_vb + pid_h * stride_vh + offs_d[None, :]
        if PAD_D:
            k_tile = tl.load(k_base + offs_n[None, :] * stride_kn,
                             mask=n_mask[None, :] & (offs_d[:, None] < HEAD_DIM),
                             other=0.0)
            v_tile = tl.load(v_base + offs_n[:, None] * stride_vn,
                             mask=n_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
                             other=0.0)
        else:
            k_tile = tl.load(k_base + offs_n[None, :] * stride_kn,
                             mask=n_mask[None, :], other=0.0)
            v_tile = tl.load(v_base + offs_n[:, None] * stride_vn,
                             mask=n_mask[:, None], other=0.0)

        dk = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)

        lo = tl.maximum(0, pid_n * BLOCK_N - Q_START)
        lo = (lo // BLOCK_M) * BLOCK_M

        for g in tl.static_range(GROUP_SIZE):
            q_head = pid_h * GROUP_SIZE + g
            alpha_g = tl.load(ALPHA + q_head)
            beta_g = tl.load(BETA + q_head)
            mu_g = tl.load(MU + q_head)
            s_g = tl.load(SSMAX + q_head)
            q_base = Q + pid_b * stride_qb + q_head * stride_qh + offs_d[None, :]
            do_base = dO + pid_b * stride_dob + q_head * stride_doh + offs_d[None, :]
            row0 = pid_b * num_heads + q_head
            for start_m in range(lo, Q_LEN, BLOCK_M):
                start_m = tl.multiple_of(start_m, BLOCK_M)
                offs_m = start_m + tl.arange(0, BLOCK_M)
                m_mask = offs_m < Q_LEN

                if PAD_D:
                    d_mask_2d = offs_d[None, :] < HEAD_DIM
                    q_tile = tl.load(q_base + offs_m[:, None] * stride_qm,
                                     mask=m_mask[:, None] & d_mask_2d, other=0.0)
                    do_tile = tl.load(do_base + offs_m[:, None] * stride_dom,
                                      mask=m_mask[:, None] & d_mask_2d, other=0.0)
                else:
                    q_tile = tl.load(q_base + offs_m[:, None] * stride_qm,
                                     mask=m_mask[:, None], other=0.0)
                    do_tile = tl.load(do_base + offs_m[:, None] * stride_dom,
                                      mask=m_mask[:, None], other=0.0)

                row = row0 * Q_LEN + offs_m
                lse = tl.load(LSE + row, mask=m_mask, other=0.0)
                delta = tl.load(Delta + row, mask=m_mask, other=0.0)

                log_n = tl.load(LOG_N + pid_b * stride_lb + offs_m * stride_lm,
                                mask=m_mask, other=0.0)
                mul = s_g * log_n
                q_abs = offs_m + Q_START

                if FP32_INPUT:
                    qk = tl.dot(q_tile, k_tile, input_precision='tf32x3')
                    dp = tl.dot(do_tile, tl.trans(v_tile), input_precision='tf32x3')
                else:
                    qk = tl.dot(q_tile, k_tile)
                    dp = tl.dot(do_tile, tl.trans(v_tile))
                qk = qk * qk_scale

                d = offs_n[None, :] - q_abs[:, None]
                logf = tl.log(tl.abs(d - mu_g) + eps)
                b2 = -(alpha_g * tl.exp(beta_g * logf)) * _LOG2E

                logits = (qk + b2) * mul[:, None]
                p = tl.exp2(logits - lse[:, None])
                visible = q_abs[:, None] >= offs_n[None, :]
                p = tl.where(visible & n_mask[None, :] & m_mask[:, None], p, 0.0)

                ds = p * (dp - delta[:, None]) * _LN2
                dmul = ds * mul[:, None]
                p_cast = p.to(do_tile.dtype)
                dl_cast = (dmul * qk_scale).to(q_tile.dtype)

                if FP32_INPUT:
                    dv = tl.dot(tl.trans(p_cast), do_tile, dv, input_precision='tf32x3')
                    dk = tl.dot(tl.trans(dl_cast), q_tile, dk, input_precision='tf32x3')
                else:
                    dv = tl.dot(tl.trans(p_cast), do_tile, dv)
                    dk = tl.dot(tl.trans(dl_cast), q_tile, dk)

        if PAD_D:
            out_mask = n_mask[:, None] & (offs_d[None, :] < HEAD_DIM)
        else:
            out_mask = n_mask[:, None]
        dk_ptrs = (dK + pid_b * stride_dkb + pid_h * stride_dkh
                   + offs_n[:, None] * stride_dkn + offs_d[None, :])
        dv_ptrs = (dV + pid_b * stride_dvb + pid_h * stride_dvh
                   + offs_n[:, None] * stride_dvn + offs_d[None, :])
        tl.store(dk_ptrs, dk.to(dK.dtype.element_ty), mask=out_mask)
        tl.store(dv_ptrs, dv.to(dV.dtype.element_ty), mask=out_mask)

    _chord_bwd_dq_wide = triton.autotune(
        configs=_BWD_DQ_CONFIGS_WIDE, key=_TUNE_KEY, warmup=10, rep=30)(_chord_bwd_dq_kernel)
    _chord_bwd_dq_narrow = triton.autotune(
        configs=_BWD_DQ_CONFIGS_NARROW, key=_TUNE_KEY, warmup=10, rep=30)(_chord_bwd_dq_kernel)
    _chord_bwd_dkdv_wide = triton.autotune(
        configs=_BWD_DKDV_CONFIGS_WIDE, key=_TUNE_KEY, warmup=10, rep=30)(_chord_bwd_dkdv_kernel)
    _chord_bwd_dkdv_narrow = triton.autotune(
        configs=_BWD_DKDV_CONFIGS_NARROW, key=_TUNE_KEY, warmup=10, rep=30)(_chord_bwd_dkdv_kernel)


_DTYPES = (torch.float16, torch.bfloat16, torch.float32) if HAS_TRITON else ()
_ELEM_SIZE = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}
_WIDE_HEAD_DIM = 64


def _check_tensors(q, k, v, log_n):
    '''
    Validate the shapes, dtypes and layouts of the chord attention inputs.

    Args:
        q: Queries, `[B, H, L, D]`.
        k: Keys, `[B, H_kv, S, D]`.
        v: Values, `[B, H_kv, S, D]`.
        log_n: `[B, q_len]` fp32 scaling input.

    Returns:
        tuple: `(B, H, L, D, Hkv, S)`.

    Raises:
        ValueError: On any unsupported device, shape, stride, dtype or head layout.
    '''
    for name, t in (('q', q), ('k', k), ('v', v)):
        if not t.is_cuda:
            raise ValueError(f'{name} must live on CUDA')
        if t.dim() != 4:
            raise ValueError(f'{name} must be 4-D [B, H, L, D], got {tuple(t.shape)}')
        if t.stride(-1) != 1:
            raise ValueError(f'{name} must be contiguous along the last dimension (stride(-1) == 1)')
        if t.dtype not in _DTYPES:
            raise ValueError(f'{name} has unsupported dtype {t.dtype} (only fp16 / bf16 / fp32)')

    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError('q / k / v must all have the same dtype')
    if k.shape != v.shape:
        raise ValueError(f'k / v must have the same shape, got {tuple(k.shape)} vs {tuple(v.shape)}')

    B, H, L, D = q.shape
    Bk, Hkv, S, Dk = k.shape
    if B != Bk or D != Dk:
        raise ValueError(f'q {tuple(q.shape)} and k {tuple(k.shape)} disagree on B / D')
    if H % Hkv != 0:
        raise ValueError(f'num_heads {H} must be divisible by num_kv_heads {Hkv}')
    if D < 16:
        raise ValueError(f'head_dim {D} is too small (tl.dot requires >= 16)')
    if log_n.shape != (B, L):
        raise ValueError(f'log_n must be [B, q_len] = {(B, L)}, got {tuple(log_n.shape)}')
    if log_n.dtype != torch.float32:
        raise ValueError(f'log_n must be fp32, got {log_n.dtype}')
    if log_n.stride(-1) != 1:
        raise ValueError('log_n must be contiguous along the last dimension')
    return B, H, L, D, Hkv, S


def _pick_tuner(wide_tuner, narrow_tuner, dtype, head_dim):
    '''
    Pick the autotuner configuration set for a dtype / head dimension.

    The "wide" set (where `BLOCK_N` may reach 256) is only used for half precision
    with `head_dim <= 64`. An fp32 tile is twice as wide as a half-precision one, and
    the same holds for `head_dim >= 96`, so both take the conservative set to avoid
    `OutOfResources` (shared memory is only ~99KB).

    Args:
        wide_tuner: Autotuner holding the configurations for the wide shapes.
        narrow_tuner: Autotuner holding the conservative configurations.
        dtype (torch.dtype): Dtype of the query / key / value tensors.
        head_dim (int): Real head dimension `D`.

    Returns:
        Autotuner: The autotuner to launch through.
    '''
    if dtype in (torch.float16, torch.bfloat16) and head_dim <= _WIDE_HEAD_DIM:
        return wide_tuner
    return narrow_tuner


_TUNE_CACHE = {}


def _q_bucket(n: int) -> int:
    '''
    Round a query length to the bucket used as part of the autotune key.

    Args:
        n (int): Query length `L`.

    Returns:
        int: The bucket, always a power of two.
    '''
    if n <= 1:
        return 1
    if n <= 8:
        return 8
    if n <= 64:
        return 64
    if n <= 512:
        return 512
    if n <= 4096:
        return 4096
    return 1 << (n - 1).bit_length()


def _kv_bucket(n: int) -> int:
    '''
    Round a key / value length to the bucket used as part of the autotune key.

    Args:
        n (int): Key / value length `S`.

    Returns:
        int: The bucket, always a power of two.
    '''
    if n <= 512:
        return 512
    if n <= 4096:
        return 4096
    if n <= 32768:
        return 32768
    return 1 << (n - 1).bit_length()


def _launch(tuner, kernel, bucket, grid, args, kwargs):
    '''
    Launch `kernel` through its autotuner, caching the chosen config per bucket.

    The first call for a bucket goes through the autotuner, which benchmarks the
    configurations; later calls skip that bookkeeping and launch directly with the
    cached configuration.

    Args:
        tuner: Autotuner wrapping `kernel`.
        kernel: The `@triton.jit` kernel to launch.
        bucket (tuple): Autotune-cache key identifying the shape and dtype.
        grid (tuple): Launch grid.
        args (tuple): Positional kernel arguments.
        kwargs (dict): Constexpr kernel arguments.
    '''
    cfg = _TUNE_CACHE.get(bucket)
    if cfg is None:
        tuner[grid](*args, **kwargs)
        _TUNE_CACHE[bucket] = tuner.best_config
        return
    kernel[grid](*args, **kwargs, **cfg.all_kwargs())


_SM_COUNT = {}


def _sm_count(device) -> int:
    '''
    Number of streaming multiprocessors of `device`, used to size the split.

    The value is cached per device string because querying the device properties is
    comparatively expensive.

    Args:
        device: Device index, `torch.device` or string accepted by the CUDA runtime.

    Returns:
        int: The device's `multi_processor_count`.
    '''
    key = str(device)
    if key not in _SM_COUNT:
        _SM_COUNT[key] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SM_COUNT[key]


def _split_count(B: int, H: int, L: int, S: int, sm_count: int) -> int:
    '''
    How many programs the forward pass is split over for this shape.

    Splitting only pays off on a long sequence with too few resident programs to fill
    the SMs: `S < 1024` and an already-occupied device stay on the single-program
    kernel. The split count is capped at `16` and at one 512-key chunk per split.

    Args:
        B (int): Batch size.
        H (int): Number of query heads.
        L (int): Query length.
        S (int): Key / value length.
        sm_count (int): Number of streaming multiprocessors of the device.

    Returns:
        int: Number of splits, at least `1`.
    '''
    programs = triton.cdiv(max(L, 1), 16) * H * B
    if S < 1024 or programs >= 4 * sm_count:
        return 1
    want = triton.cdiv(4 * sm_count, max(H * B, 1))
    return int(max(1, min(want, 16, triton.cdiv(S, 512))))


def _chord_forward_split_impl(q, k, v, alpha, beta, mu, ssmax, log_n, q_start,
                              softmax_scale, eps, num_splits, need_lse):
    '''
    Forward pass split over several programs, merged afterwards.

    Allocates the fp32 partial planes, launches the split kernel and then runs
    `_chord_merge_kernel` to rescale and reduce the splits into the query-dtype output.

    Args:
        q (torch.Tensor): Queries, `[B, H, L, D]`.
        k (torch.Tensor): Keys, `[B, H_kv, S, D]`.
        v (torch.Tensor): Values, `[B, H_kv, S, D]`.
        alpha (torch.Tensor): `[H]` GGD decay scale.
        beta (torch.Tensor): `[H]` GGD decay exponent.
        mu (torch.Tensor): `[H]` GGD preferred distance.
        ssmax (torch.Tensor): `[H]` logit scale applied before the softmax.
        log_n (torch.Tensor): `[B, q_len]` fp32 scaling input.
        q_start (int): Absolute position of the first query.
        softmax_scale (float): Scale applied to the raw scores.
        eps (float): Epsilon of the GGD bias.
        num_splits (int): Number of key-range splits.
        need_lse (bool): Whether the log-sum-exp is wanted.

    Returns:
        tuple: `(out, lse)` where `lse` is `[B, H, L]` fp32, or None when `need_lse`
        is False.
    '''
    B, H, L, D, Hkv, S = _check_tensors(q, k, v, log_n)

    outp = torch.empty((num_splits, B, H, L, D), dtype=torch.float32, device=q.device)
    mpart = torch.empty((num_splits, B, H, L), dtype=torch.float32, device=q.device)
    lpart = torch.empty((num_splits, B, H, L), dtype=torch.float32, device=q.device)
    out = torch.empty((B, H, L, D), dtype=q.dtype, device=q.device)
    lse = (torch.empty((B, H, L), dtype=torch.float32, device=q.device)
           if need_lse else torch.empty(0, dtype=torch.float32, device=q.device))

    block_d = triton.next_power_of_2(D)
    tuner = _pick_tuner(_chord_fwd_split_wide, _chord_fwd_split_narrow, q.dtype, D)
    grid = lambda meta: (triton.cdiv(L, meta['BLOCK_M']) * num_splits, H, B)
    _launch(
        tuner, _chord_fwd_split_kernel,
        ('fwd_split', _q_bucket(L), _kv_bucket(S), D, H // Hkv, q.dtype, num_splits),
        grid,
        (q, k, v, alpha, beta, mu, ssmax, log_n, outp, mpart, lpart,
         q.stride(0), q.stride(1), q.stride(2),
         k.stride(0), k.stride(1), k.stride(2),
         v.stride(0), v.stride(1), v.stride(2),
         outp.stride(0), outp.stride(1), outp.stride(2), outp.stride(3),
         mpart.stride(0), mpart.stride(1), mpart.stride(2),
         log_n.stride(0), log_n.stride(1),
         L, S, int(q_start), float(softmax_scale), float(eps)),
        dict(
            GROUP_SIZE=H // Hkv,
            HEAD_DIM=D,
            BLOCK_D=block_d,
            PAD_D=block_d != D,
            FP32_INPUT=q.dtype == torch.float32,
            NUM_SPLITS=num_splits,
        ),
    )

    merge_m = 64
    _chord_merge_kernel[(triton.cdiv(max(L, 1), merge_m), H, B)](
        outp, mpart, lpart, out, lse,
        outp.stride(0), outp.stride(1), outp.stride(2), outp.stride(3),
        mpart.stride(0), mpart.stride(1), mpart.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        L,
        NUM_SPLITS=num_splits,
        HEAD_DIM=D,
        BLOCK_D=block_d,
        PAD_D=block_d != D,
        STORE_LSE=bool(need_lse),
        BLOCK_M=merge_m,
        num_warps=4,
    )
    return out, (lse if need_lse else None)


def _chord_forward_impl(q, k, v, alpha, beta, mu, ssmax, log_n, q_start,
                        softmax_scale, eps, need_lse):
    '''
    Single-program forward pass of the chord attention kernel.

    Picks the split count from the shape and delegates to
    `_chord_forward_split_impl` when more than one split is worth it, otherwise
    launches `_chord_fwd_kernel` on the `(ceil(L / BLOCK_M), H, B)` grid.

    Args:
        q (torch.Tensor): Queries, `[B, H, L, D]`.
        k (torch.Tensor): Keys, `[B, H_kv, S, D]`.
        v (torch.Tensor): Values, `[B, H_kv, S, D]`.
        alpha (torch.Tensor): `[H]` GGD decay scale.
        beta (torch.Tensor): `[H]` GGD decay exponent.
        mu (torch.Tensor): `[H]` GGD preferred distance.
        ssmax (torch.Tensor): `[H]` logit scale applied before the softmax.
        log_n (torch.Tensor): `[B, q_len]` fp32 scaling input.
        q_start (int): Absolute position of the first query.
        softmax_scale (float): Scale applied to the raw scores.
        eps (float): Epsilon of the GGD bias.
        need_lse (bool): Whether the log-sum-exp is wanted.

    Returns:
        tuple: `(out, lse)` where `lse` is `[B, H, L]` fp32, or None when `need_lse`
        is False.
    '''
    B, H, L, D, Hkv, S = _check_tensors(q, k, v, log_n)

    out = torch.empty((B, H, L, D), dtype=q.dtype, device=q.device)
    lse = (torch.empty((B, H, L), dtype=torch.float32, device=q.device)
           if need_lse else torch.empty(0, dtype=torch.float32, device=q.device))

    if S > 0 and L > 0:
        num_splits = _split_count(B, H, L, S, _sm_count(q.device))
        if num_splits > 1:
            return _chord_forward_split_impl(
                q, k, v, alpha, beta, mu, ssmax, log_n, q_start,
                softmax_scale, eps, num_splits, need_lse)

        block_d = triton.next_power_of_2(D)
        grid = lambda meta: (triton.cdiv(L, meta['BLOCK_M']), H, B)
        tuner = _pick_tuner(_chord_fwd_wide, _chord_fwd_narrow, q.dtype, D)
        args = (q, k, v, alpha, beta, mu, ssmax, log_n, out, lse,
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                log_n.stride(0), log_n.stride(1),
                L, S, int(q_start), float(softmax_scale), float(eps))
        kwargs = dict(
            GROUP_SIZE=H // Hkv,
            HEAD_DIM=D,
            BLOCK_D=block_d,
            PAD_D=block_d != D,
            FP32_INPUT=q.dtype == torch.float32,
            STORE_LSE=bool(need_lse),
        )
        try:
            _launch(tuner, _chord_fwd_kernel,
                    ('fwd', _q_bucket(L), _kv_bucket(S), D, H // Hkv, q.dtype),
                    grid, args, kwargs)
        except OutOfResources as exc:
            raise RuntimeError(
                f'the Triton chord kernel does not fit on this device (head_dim={D}, '
                f'dtype={q.dtype}): {exc}'
            ) from exc
    return out, (lse if need_lse else None)


def _chord_delta_impl(out, dout):
    '''
    Row-wise delta `sum_d out * dout` of the attention output, used by the backward.

    Args:
        out (torch.Tensor): Forward output, `[B, H, L, D]`.
        dout (torch.Tensor): Upstream gradient, `[B, H, L, D]`.

    Returns:
        torch.Tensor: The `[B, H, L]` fp32 delta, left empty when `L == 0`.
    '''
    B, H, L, D = out.shape
    delta = torch.empty((B, H, L), dtype=torch.float32, device=out.device)
    if L == 0:
        return delta
    block_m = 128
    block_d = triton.next_power_of_2(D)
    _chord_delta_kernel[(triton.cdiv(L, block_m), H, B)](
        out, dout, delta,
        out.stride(0), out.stride(1), out.stride(2),
        dout.stride(0), dout.stride(1), dout.stride(2),
        L,
        HEAD_DIM=D,
        BLOCK_D=block_d,
        PAD_D=block_d != D,
        BLOCK_M=block_m,
        num_warps=4,
    )
    return delta


def _dq_block_m(q_len: int, block_d: int, elem_size: int) -> int:
    '''
    Choose the query block size of the dq kernel from the shape and element size.

    Args:
        q_len (int): Query length `L`.
        block_d (int): Padded head dimension of the kernel.
        elem_size (int): Size in bytes of one element of the query dtype.

    Returns:
        int: The query block size, currently always `16`.
    '''
    return 16


def _chord_backward_impl(q, k, v, out, dout, lse, alpha, beta, mu, ssmax, log_n,
                         q_start, softmax_scale, eps):
    '''
    Backward pass: gradients of q, k, v and of the per-head GGD parameters.

    The row-wise delta is computed first, then the `dq` kernel and the `dk` / `dv`
    kernel are launched on disjoint grids. The four per-head gradients are only
    partial inside the kernels, so their planes are summed here.

    Args:
        q (torch.Tensor): Saved queries, `[B, H, L, D]`.
        k (torch.Tensor): Saved keys, `[B, H_kv, S, D]`.
        v (torch.Tensor): Saved values, `[B, H_kv, S, D]`.
        out (torch.Tensor): Forward output, `[B, H, L, D]`.
        dout (torch.Tensor): Upstream gradient, `[B, H, L, D]`.
        lse (torch.Tensor): `[B, H, L]` fp32 log-sum-exp from the forward pass.
        alpha (torch.Tensor): `[H]` GGD decay scale.
        beta (torch.Tensor): `[H]` GGD decay exponent.
        mu (torch.Tensor): `[H]` GGD preferred distance.
        ssmax (torch.Tensor): `[H]` logit scale applied before the softmax.
        log_n (torch.Tensor): `[B, q_len]` fp32 scaling input.
        q_start (int): Absolute position of the first query.
        softmax_scale (float): Scale applied to the raw scores.
        eps (float): Epsilon of the GGD bias.

    Returns:
        tuple: `(dq, dk, dv, d_alpha, d_beta, d_mu, d_ssmax)`; the last four are `[H]`
        fp32.
    '''
    B, H, L, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]

    delta = _chord_delta_impl(out, dout)

    dq = torch.empty((B, H, L, D), dtype=q.dtype, device=q.device)
    dk = torch.empty((B, Hkv, S, D), dtype=q.dtype, device=q.device)
    dv = torch.empty((B, Hkv, S, D), dtype=q.dtype, device=q.device)
    block_m = _dq_block_m(L, triton.next_power_of_2(D), _ELEM_SIZE[q.dtype])
    grid_m = triton.cdiv(max(L, 1), block_m)
    partial = torch.empty((4, grid_m, H, B), dtype=torch.float32, device=q.device)

    if S > 0 and L > 0:
        block_d = triton.next_power_of_2(D)
        grid_dq = (grid_m, H, B)
        grid_dkdv = lambda meta: (triton.cdiv(S, meta['BLOCK_N']), Hkv, B)
        tuner_dq = _pick_tuner(_chord_bwd_dq_wide, _chord_bwd_dq_narrow, q.dtype, D)
        tuner_dkdv = _pick_tuner(_chord_bwd_dkdv_wide, _chord_bwd_dkdv_narrow,
                                 q.dtype, D)
        consts = dict(
            GROUP_SIZE=H // Hkv,
            HEAD_DIM=D,
            BLOCK_D=block_d,
            PAD_D=block_d != D,
            FP32_INPUT=q.dtype == torch.float32,
        )
        try:
            _launch(
                tuner_dq, _chord_bwd_dq_kernel,
                ('dq', _q_bucket(L), _kv_bucket(S), D, H // Hkv, q.dtype),
                grid_dq,
                (q, k, v, dout, lse, delta, alpha, beta, mu, ssmax, log_n, dq,
                 partial[0], partial[1], partial[2], partial[3],
                 q.stride(0), q.stride(1), q.stride(2),
                 k.stride(0), k.stride(1), k.stride(2),
                 v.stride(0), v.stride(1), v.stride(2),
                 dout.stride(0), dout.stride(1), dout.stride(2),
                 dq.stride(0), dq.stride(1), dq.stride(2),
                 log_n.stride(0), log_n.stride(1),
                 L, S, int(q_start), float(softmax_scale), float(eps)),
                dict(BLOCK_M=block_m, **consts),
            )
            _launch(
                tuner_dkdv, _chord_bwd_dkdv_kernel,
                ('dkdv', _q_bucket(L), _kv_bucket(S), D, H // Hkv, q.dtype),
                grid_dkdv,
                (q, k, v, dout, lse, delta, alpha, beta, mu, ssmax, log_n, dk, dv,
                 q.stride(0), q.stride(1), q.stride(2),
                 k.stride(0), k.stride(1), k.stride(2),
                 v.stride(0), v.stride(1), v.stride(2),
                 dout.stride(0), dout.stride(1), dout.stride(2),
                 dk.stride(0), dk.stride(1), dk.stride(2),
                 dv.stride(0), dv.stride(1), dv.stride(2),
                 log_n.stride(0), log_n.stride(1),
                 L, S, int(q_start), float(softmax_scale), float(eps)),
                dict(consts),
            )
        except OutOfResources as exc:
            raise RuntimeError(
                f'the Triton chord backward kernel does not fit on this device (head_dim={D}, '
                f'dtype={q.dtype}): {exc}'
            ) from exc

        d_alpha, d_beta, d_mu, d_ssmax = partial.sum(dim=(1, 3)).unbind(0)
    else:
        zero = torch.zeros((H,), dtype=torch.float32, device=q.device)
        d_alpha = d_beta = d_mu = d_ssmax = zero

    return dq, dk, dv, d_alpha, d_beta, d_mu, d_ssmax


def _check_params(alpha, beta, mu, ssmax, num_heads):
    '''
    Validate the dtype, device, shape and contiguity of the per-head GGD parameters.

    Args:
        alpha: `[H]` GGD decay scale.
        beta: `[H]` GGD decay exponent.
        mu: `[H]` GGD preferred distance.
        ssmax: `[H]` logit scale applied before the softmax.
        num_heads (int): Expected number of heads `H`.

    Raises:
        ValueError: If a parameter is not a contiguous fp32 CUDA tensor of shape
            `[num_heads]`.
    '''
    for name, t in (('alpha', alpha), ('beta', beta), ('mu', mu),
                    ('ssmax', ssmax)):
        if not t.is_cuda:
            raise ValueError(f'{name} must live on CUDA')
        if t.dtype != torch.float32:
            raise ValueError(f'{name} must be fp32, got {t.dtype}')
        if t.shape != (num_heads,):
            raise ValueError(f'{name} must be [{num_heads}], got {tuple(t.shape)}')
        if not t.is_contiguous():
            raise ValueError(f'{name} must be contiguous')


@autocast_safe
class _TritonChordFunction(torch.autograd.Function):
    '''
    Autograd wrapper of the fused Triton chord attention, with a custom backward.

    The forward keeps `q`, `k`, `v`, `out`, `lse` and the per-head parameters for the
    backward, which re-derives the probabilities from the saved log-sum-exp instead of
    storing the attention matrix.
    '''

    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, mu, ssmax, log_n, q_start,
                softmax_scale, eps):
        '''
        Run the fused forward pass and save the tensors the backward needs.

        Args:
            ctx: Autograd context; stores the inputs, the output, the log-sum-exp and
                the launch configuration.
            q: Queries, `[B, H, L, D]`.
            k: Keys, `[B, H_kv, S, D]`.
            v: Values, `[B, H_kv, S, D]`.
            alpha: `[H]` GGD decay scale.
            beta: `[H]` GGD decay exponent.
            mu: `[H]` GGD preferred distance.
            ssmax: `[H]` logit scale applied before the softmax.
            log_n: `[B, q_len]` fp32 scaling input.
            q_start: Absolute position of the first query.
            softmax_scale: Scale applied to the raw scores.
            eps: Epsilon of the GGD bias.

        Returns:
            torch.Tensor: The attention output, `[B, H, L, D]`.
        '''
        out, lse = _chord_forward_impl(
            q, k, v, alpha, beta, mu, ssmax, log_n, q_start,
            softmax_scale, eps, need_lse=True,
        )
        ctx.save_for_backward(q, k, v, out, lse, alpha, beta, mu, ssmax, log_n)
        ctx.q_start = q_start
        ctx.softmax_scale = softmax_scale
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, dout):
        '''
        Backpropagate into q / k / v and the per-head GGD parameters.

        Args:
            ctx: Autograd context holding the saved forward tensors.
            dout: Upstream gradient of the output, `[B, H, L, D]`.

        Returns:
            tuple: One gradient per forward input, in the same order; `None` for the
            inputs that do not require a gradient, and for `log_n`, `q_start`,
            `softmax_scale` and `eps`.
        '''
        q, k, v, out, lse, alpha, beta, mu, ssmax, log_n = ctx.saved_tensors
        if not dout.is_contiguous():
            dout = dout.contiguous()

        dq, dk, dv, d_alpha, d_beta, d_mu, d_ssmax = _chord_backward_impl(
            q, k, v, out, dout, lse, alpha, beta, mu, ssmax, log_n,
            ctx.q_start, ctx.softmax_scale, ctx.eps,
        )
        needs = ctx.needs_input_grad
        return (
            dq if needs[0] else None,
            dk if needs[1] else None,
            dv if needs[2] else None,
            d_alpha if needs[3] else None,
            d_beta if needs[4] else None,
            d_mu if needs[5] else None,
            d_ssmax if needs[6] else None,
            None,
            None,
            None,
            None,
        )


def apply_chord_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    mu: torch.Tensor,
    ssmax: torch.Tensor,
    log_n: torch.Tensor,
    q_start: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    '''
    Chord attention forward pass on the Triton kernels.

    When gradients are enabled for any input the autograd Function is used; otherwise
    the raw forward implementation runs and no `lse` is kept.

    Args:
        q (torch.Tensor): Queries, `[B, H, L, D]`.
        k (torch.Tensor): Keys, `[B, H_kv, S, D]`.
        v (torch.Tensor): Values, `[B, H_kv, S, D]`.
        alpha (torch.Tensor): `[H]` GGD decay scale.
        beta (torch.Tensor): `[H]` GGD decay exponent.
        mu (torch.Tensor): `[H]` GGD preferred distance.
        ssmax (torch.Tensor): `[H]` logit scale applied before the softmax.
        log_n (torch.Tensor): `[B, q_len]` fp32 scaling input.
        q_start (Optional[int]): Absolute position of the first query; defaults to
            `S - L`.
        softmax_scale (Optional[float]): Scale applied to the scores; defaults to
            `D ** -0.5`.
        eps (float): Epsilon of the GGD bias.

    Returns:
        torch.Tensor: The attention output, `[B, H, L, D]`.

    Raises:
        RuntimeError: If Triton is unavailable, or a kernel does not fit on the
            device.
        ValueError: On unsupported inputs or a negative `q_start`.
    '''
    if not HAS_TRITON:
        raise RuntimeError('there is no usable triton in the current environment')

    B, H, L, D, Hkv, S = _check_tensors(q, k, v, log_n)
    _check_params(alpha, beta, mu, ssmax, H)
    if softmax_scale is None:
        softmax_scale = D ** -0.5
    if q_start is None:
        q_start = S - L
    q_start = int(q_start)
    if q_start < 0:
        raise ValueError(f'q_start must not be negative, got {q_start}')

    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (q, k, v, alpha, beta, mu, ssmax))
    if needs_grad:
        return _TritonChordFunction.apply(
            q, k, v, alpha, beta, mu, ssmax, log_n,
            q_start, float(softmax_scale), float(eps),
        )
    if L == 0:
        return torch.empty((B, H, L, D), dtype=q.dtype, device=q.device)
    out, _ = _chord_forward_impl(
        q, k, v, alpha, beta, mu, ssmax, log_n, q_start,
        softmax_scale, eps, need_lse=False,
    )
    return out
