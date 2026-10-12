import torch
import torch.nn as nn

from motif.block.attention.base import BasicAttention, AttentionOutput
from motif.model.cache import KVLayerCache
from motif.ops import (
    apply_chord_attention,
    RMSNorm,
    HAS_TRITON
)

from typing import Optional, Union


_HALF_DTYPES = (torch.float16, torch.bfloat16)
_TRITON_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_GGD_EPS = 1e-5
_LOG_TABLE = {}


def _log_n_table(n: int, device, dtype=torch.float32) -> torch.Tensor:
    '''
    Cached table of `log(1 .. size)`, long enough to cover `n` positions.

    The table is kept per `(device, dtype)` in `_LOG_TABLE` and rounded up to a
    power of two (at least 8192 entries) so that growing sequences reuse it.

    Args:
        n (int): Number of positions that must be covered.
        device: Device the table is built on.
        dtype: Floating dtype of the table (`torch.float32` by default).

    Returns:
        torch.Tensor: `log(1), log(2), ...` with at least `n` entries.
    '''
    key = (str(device), str(dtype))
    table = _LOG_TABLE.get(key)
    if table is None or table.numel() < n:
        size = max(n, 8192)
        size = 1 << (size - 1).bit_length()
        table = torch.log(torch.arange(1, size + 1, device=device, dtype=dtype))
        _LOG_TABLE[key] = table
    return table


def _promote(t: torch.Tensor) -> torch.Tensor:
    '''
    Upcast half-precision tensors to fp32 and leave every other dtype untouched.

    Args:
        t (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: The fp32 cast for fp16 / bf16 input, otherwise `t`.
    '''
    return t.float() if t.dtype in _HALF_DTYPES else t


class ChordAttention(BasicAttention):
    '''
    Chord attention: softmax attention with a gated geometric-decay bias.

    Attributes:
        q_proj (nn.Linear): Query projection, `hidden_size -> hidden_size`.
        k_proj (nn.Linear): Key projection, `hidden_size -> kv_dim`.
        v_proj (nn.Linear): Value projection, `hidden_size -> kv_dim`.
        o_proj (nn.Linear): Output projection, `hidden_size -> hidden_size`.
        g_proj (nn.Linear): Sigmoid gate applied to the attention output.
        q_norm (RMSNorm): Per-head RMSNorm over the query head dimension.
        k_norm (RMSNorm): Per-head RMSNorm over the key head dimension.
        theta_alpha (nn.Parameter): Per-head decay scale of the GGD bias.
        theta_beta (nn.Parameter): Per-head decay exponent of the GGD bias.
        theta_mu (nn.Parameter): Per-head preferred distance of the GGD bias,
            frozen (`requires_grad=False`).
        ssmax_s (nn.Parameter): Per-head logit scale applied before the softmax
            (stabilised softmax).
    '''

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        **kwargs,
    ):
        '''
        Initialise the projections, the per-head normalisations and the GGD
        parameters.

        Args:
            hidden_size (int): Width of the input / output representation.
            num_heads (int): Number of query heads.
            num_kv_heads (Optional[int]): Number of key / value heads; None means
                one per query head.
            **kwargs: Accepted for signature compatibility and ignored.
        '''
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.g_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.theta_alpha = nn.Parameter(torch.zeros(num_heads))
        self.theta_beta = nn.Parameter(torch.zeros(num_heads))
        self.theta_mu = nn.Parameter(torch.zeros(num_heads), requires_grad=False)

        self.ssmax_s = nn.Parameter(torch.ones(num_heads))

    @classmethod
    def cache_type(cls) -> type:
        '''
        Returns:
            type: The per-layer cache class, `KVLayerCache`.
        '''
        return KVLayerCache

    def _ggd_parameters(self):
        '''
        Materialise the GGD bias parameters in fp32.

        Returns:
            tuple: `(alpha, beta, mu)` with `alpha = exp(theta_alpha)`,
            `beta = theta_beta` and `mu = exp(theta_mu) - exp(-theta_mu)`.
        '''
        ta = _promote(self.theta_alpha)
        tb = _promote(self.theta_beta)
        tm = _promote(self.theta_mu)
        mu = torch.exp(tm) - torch.exp(-tm)
        return torch.exp(ta), tb, mu

    def _compute_ggd_bias(
        self,
        q_len: int,
        kv_len: int,
        q_start: Union[int, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        '''
        GGD bias for every (query, key) pair of the current chunk.

        The bias is `-alpha * (|d - mu| + eps) ** beta`, where `d` is the signed
        distance `key_position - query_position` and `eps = _GGD_EPS` keeps the
        power well defined at `d == mu`.

        Args:
            q_len (int): Number of query positions in this call.
            kv_len (int): Number of key positions available (cache included).
            q_start (int | torch.Tensor): Absolute position of the first query in the
                sequence. A scalar tensor keeps the position out of the graph.
            device (torch.device): Device the distance grid is built on.
            dtype (torch.dtype): Output dtype (`torch.float32` by default).

        Returns:
            torch.Tensor: `[num_heads, q_len, kv_len]` additive bias.
        '''
        alpha, beta, mu = self._ggd_parameters()
        cd = alpha.dtype
        if isinstance(q_start, torch.Tensor):
            start = q_start.to(device=device, dtype=cd).reshape(-1)[0]
            q_pos = torch.arange(q_len, device=device, dtype=cd) + start
        else:
            q_pos = torch.arange(q_len, device=device, dtype=cd) + float(q_start)
        k_pos = torch.arange(kv_len, device=device, dtype=cd)
        d = k_pos[None, :] - q_pos[:, None]
        b = torch.abs(d[None] - mu[:, None, None]) + _GGD_EPS
        return (-alpha[:, None, None] * b ** beta[:, None, None]).to(dtype)

    @torch.no_grad()
    def relative_prior(
        self,
        max_distance: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        '''
        The GGD bias as a function of signed distance, for inspection / plotting.

        Args:
            max_distance (int): Largest absolute distance to evaluate.
            device (Optional[torch.device]): Device of the result; defaults to the
                device holding `theta_alpha`.
            dtype (torch.dtype): Output dtype (`torch.float32` by default).

        Returns:
            torch.Tensor: `[num_heads, 2 * max_distance + 1]` bias values, ordered
            from `-max_distance` to `+max_distance`.
        '''
        alpha, beta, mu = self._ggd_parameters()
        device = device if device is not None else self.theta_alpha.device
        d = torch.arange(-max_distance, max_distance + 1, device=device, dtype=alpha.dtype)
        b = torch.abs(d[None, :] - mu[:, None]) + _GGD_EPS
        return (-alpha[:, None] * b ** beta[:, None]).to(dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        embedding_start: Union[int, torch.Tensor] = 0,
        past_key_value: Optional[KVLayerCache] = None,
        section_log_len: Optional[torch.Tensor] = None,
    ) -> AttentionOutput:
        '''
        Run chord attention over `hidden_states`.

        The fused Triton kernel is attempted first when Triton is available, the
        input lives on CUDA, its dtype is supported, no attention mask or attention
        weights are requested and `head_dim >= 16`; any failure silently falls back
        to the eager path below.

        Args:
            hidden_states (torch.Tensor): `[B, q_len, hidden_size]` inputs; also
                used for the key / value projections when `kv_states` is None.
            kv_states (Optional[torch.Tensor]): `[B, kv_len_input, hidden_size]`
                states feeding the key / value projections.
            attention_mask (Optional[torch.Tensor]): Additive mask, or a boolean
                keep-mask, applied to the logits.
            output_attentions (bool): Also return the softmax weights and the
                relative prior in `payload`.
            embedding_start (int | torch.Tensor): Absolute position of the first
                token when no cache is used, or overrides the position the cache
                reports. Pass a scalar tensor to keep the position out of the graph
                (required to export an incremental decoding step); the fused Triton
                kernel is then skipped.
            past_key_value (Optional[KVLayerCache]): Cache to read from, and to
                append the new keys / values to. A cache may hold either
                `num_kv_heads` or `num_heads` heads; when it already holds
                `num_heads` the grouped-query expansion below is skipped.
            section_log_len (Optional[torch.Tensor]): Optional `[B, q_len]` (or
                `[B, q_len, 1]`) replacement for `log(n)`; defaults to the cached
                log table.

        Returns:
            AttentionOutput: Output tensor, optional attention weights and the
            updated cache.
        '''
        if kv_states is None:
            kv_states = hidden_states

        batch_size, q_len, _ = hidden_states.shape
        kv_len_input = kv_states.shape[1]

        G = torch.sigmoid(self.g_proj(hidden_states))

        q = self.reshape_q(self.q_proj(hidden_states), q_len, batch_size)
        k = self.reshape_kv(self.k_proj(kv_states), kv_len_input, batch_size)
        v = self.reshape_kv(self.v_proj(kv_states), kv_len_input, batch_size)

        q = self.q_norm(q)
        k = self.k_norm(k)

        use_cache = isinstance(past_key_value, KVLayerCache)
        q_start = embedding_start
        if use_cache:
            cached = past_key_value.seq_length
            if isinstance(q_start, int) and q_start == 0 and cached > 0:
                q_start = cached
            k, v = past_key_value.update(k, v, dim=2)
        current_key_value = past_key_value if use_cache else None

        kv_len = k.shape[2]

        if section_log_len is None:
            if isinstance(q_start, torch.Tensor):
                start = q_start.reshape(-1)[0].to(
                    device=hidden_states.device, dtype=torch.float32)
                log_n = torch.log(
                    start + torch.arange(1, q_len + 1, dtype=torch.float32,
                                         device=hidden_states.device))
            else:
                log_n = _log_n_table(q_start + q_len, hidden_states.device)[
                    q_start:q_start + q_len]
        else:
            sl = section_log_len
            if sl.dim() == 3:
                sl = sl.squeeze(-1)
            if sl.shape != (batch_size, q_len):
                raise ValueError(
                    f'section_log_len expects [B, q_len] or [B, q_len, 1], '
                    f'got {tuple(section_log_len.shape)}'
                )
            log_n = sl.float()

        if (HAS_TRITON and hidden_states.is_cuda
                and hidden_states.dtype in _TRITON_DTYPES
                and attention_mask is None and not output_attentions
                and not isinstance(q_start, torch.Tensor)
                and self.head_dim >= 16):
            try:
                alpha, beta, mu = self._ggd_parameters()
                out = apply_chord_attention(
                    q, k, v,
                    alpha.contiguous(), beta.contiguous(), mu.contiguous(),
                    _promote(self.ssmax_s).contiguous(),
                    log_n if log_n.dim() == 2 else log_n.expand(batch_size, q_len),
                    q_start=q_start,
                    softmax_scale=self.head_dim ** -0.5,
                    eps=_GGD_EPS,
                )
                out = out.transpose(1, 2).contiguous().view(
                    batch_size, q_len, self.hidden_size)
                return AttentionOutput(output=self.o_proj(out) * G, past_key_value=current_key_value)
            except Exception: pass

        if self.num_kv_queries > 1 and k.shape[1] != self.num_heads:
            k = k[:, :, None, :, :].expand(
                batch_size, self.num_kv_heads, self.num_kv_queries,
                kv_len, self.head_dim,
            ).reshape(batch_size, self.num_heads, kv_len, self.head_dim)
            v = v[:, :, None, :, :].expand(
                batch_size, self.num_kv_heads, self.num_kv_queries,
                kv_len, self.head_dim,
            ).reshape(batch_size, self.num_heads, kv_len, self.head_dim)

        scale = self.head_dim ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale

        logit_dtype = (torch.float32 if scores.dtype in _HALF_DTYPES else scores.dtype)
        bias = self._compute_ggd_bias(
            q_len=q_len, kv_len=kv_len, q_start=q_start,
            device=scores.device, dtype=logit_dtype,
        )
        logits = scores.to(logit_dtype) + bias.unsqueeze(0)

        log_n_4d = (log_n.view(1, 1, q_len, 1) if log_n.dim() == 1
                    else log_n.view(batch_size, 1, q_len, 1))

        logits = logits * (_promote(self.ssmax_s).view(1, -1, 1, 1)
                           * log_n_4d.to(logit_dtype))
        logits = logits.clamp_min(torch.finfo(logit_dtype).min / 4)

        q_pos = torch.arange(q_len, device=logits.device) + q_start
        k_pos = torch.arange(kv_len, device=logits.device)
        logits = logits.masked_fill(k_pos[None, :] > q_pos[:, None], float('-inf'))

        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                keep = attention_mask
                if keep.dim() == 2:
                    keep = keep[:, None, None, :]
                elif keep.dim() == 3:
                    keep = keep[:, None]
                logits = logits.masked_fill(~keep, float('-inf'))
            else:
                logits = logits + attention_mask

        attn = torch.softmax(logits, dim=-1)
        out = attn.to(v.dtype) @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_size)
        out = self.o_proj(out) * G

        payload = None
        if output_attentions:
            start = (q_start if isinstance(q_start, int)
                     else int(q_start.reshape(-1)[0]))
            payload = {
                'prior': self.relative_prior(
                    max(q_len - 1, kv_len - 1 - start, 1) + 1,
                    device=logits.device,
                ),
            }

        return AttentionOutput(
            output=out,
            attention_weights=attn if output_attentions else None,
            past_key_value=current_key_value,
            payload=payload,
        )
