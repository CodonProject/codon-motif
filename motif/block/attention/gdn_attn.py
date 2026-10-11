import torch
import torch.nn as nn
import torch.nn.functional as F

from motif.block.attention.base import BasicAttention, AttentionOutput
from motif.model.cache import GatedDeltaAttentionLayerCache
from motif.ops import (
    recurrent_gated_delta_rule,
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_triton,
    GatedRMSNorm,
    HAS_TRITON
)

import math
from typing import Optional


class GatedDeltaAttention(BasicAttention):
    '''
    Gated delta-net attention.

    Attributes:
        dk (int): Per-head key dimension (`head_k_dim`, or `head_dim` when unset).
        dv (int): Per-head value dimension, `int(dk * expand_v)`.
        conv_size (int): Kernel size of the short causal depthwise convolution.
        chunk_size (int): Chunk length used by the chunked delta-rule kernels.
        bias (bool): Whether the projections carry a bias.
        dropout (float): Dropout probability kept for configuration compatibility.
        use_triton (bool): Whether the fused Triton chunk kernel may be used.
        q_proj (nn.Linear): Query projection, `hidden_size -> num_heads * dk`.
        k_proj (nn.Linear): Key projection, `hidden_size -> num_heads * dk`.
        v_proj (nn.Linear): Value projection, `hidden_size -> num_heads * dv`.
        a_proj (nn.Linear): Per-head decay projection.
        b_proj (nn.Linear): Per-head beta (write strength) projection.
        g_proj (nn.Linear): Output gate projection, `hidden_size -> num_heads * dv`.
        o_proj (nn.Linear): Output projection, `num_heads * dv -> hidden_size`.
        q_conv / k_conv / v_conv (nn.Conv1d): Depthwise causal convolutions.
        A_log (nn.Parameter): Log of the per-head decay initialisation.
        dt_bias (nn.Parameter): Per-head bias of the decay time step.
        o_norm (GatedRMSNorm): Gated normalisation applied to the output.
    '''

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_k_dim: Optional[int] = None,
        expand_v: float = 2.0,
        conv_size: int = 4,
        chunk_size: int = 64,
        bias: bool = False,
        dropout: float = 0.0,
        use_triton: bool = True,
        **kwargs,
    ):
        '''
        Initialise the projections, the depthwise convolutions and the decay
        parameters.

        Args:
            hidden_size (int): Width of the input / output representation.
            num_heads (int): Number of attention heads; keys and values share it.
            head_k_dim (Optional[int]): Per-head key dimension; None uses the head
                dimension derived from `hidden_size / num_heads`.
            expand_v (float): Expansion factor of the value dimension relative to
                `dk`.
            conv_size (int): Kernel size of the short causal depthwise convolution.
            chunk_size (int): Chunk length for the chunked delta-rule kernels.
            bias (bool): Whether to add a bias to the projections.
            dropout (float): Dropout probability (stored for compatibility).
            use_triton (bool): Allow the fused Triton chunk kernel; it is only used
                when Triton is actually available.
            **kwargs: Accepted for signature compatibility and ignored.
        '''
        super().__init__(hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_heads)

        self.dk = head_k_dim or self.head_dim
        self.dv = int(self.dk * expand_v)
        self.conv_size = conv_size
        self.chunk_size = chunk_size
        self.bias = bias
        self.dropout = dropout
        self.use_triton = bool(use_triton and HAS_TRITON)
        dk_all = self.num_heads * self.dk
        dv_all = self.num_heads * self.dv

        self.q_proj = nn.Linear(hidden_size, dk_all, bias=bias)
        self.k_proj = nn.Linear(hidden_size, dk_all, bias=bias)
        self.v_proj = nn.Linear(hidden_size, dv_all, bias=bias)
        self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=bias)
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=bias)
        self.g_proj = nn.Linear(hidden_size, dv_all, bias=bias)
        self.o_proj = nn.Linear(dv_all, hidden_size, bias=bias)

        self.q_conv = nn.Conv1d(dk_all, dk_all, conv_size, groups=dk_all, bias=False)
        self.k_conv = nn.Conv1d(dk_all, dk_all, conv_size, groups=dk_all, bias=False)
        self.v_conv = nn.Conv1d(dv_all, dv_all, conv_size, groups=dv_all, bias=False)

        A = torch.empty(self.num_heads).uniform_(1, 16)
        self.A_log = nn.Parameter(A.log())
        dt = torch.exp(torch.rand(self.num_heads)
                       * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        self.o_norm = GatedRMSNorm(self.dv)

    @classmethod
    def cache_type(cls) -> type:
        '''
        Returns:
            type: The per-layer cache class, `GatedDeltaAttentionLayerCache`.
        '''
        return GatedDeltaAttentionLayerCache

    def _conv(self, x, conv, cache):
        '''
        Short causal depthwise convolution followed by SiLU.

        Args:
            x: `[B, L, D]` input features.
            conv (nn.Conv1d): The depthwise convolution to apply.
            cache: `[B, D, conv_size - 1]` convolution state of the previous
                chunk, or None for a fresh sequence.

        Returns:
            tuple: `(output, new_cache)` where output is `[B, L, D]` and
            new_cache is `[B, D, conv_size - 1]`.
        '''
        B, _, D = x.shape
        x = x.transpose(1, 2)
        if cache is None:
            cache = x.new_zeros(B, D, self.conv_size - 1)
        xc = torch.cat([cache, x], dim=-1)
        new_cache = xc[..., -(self.conv_size - 1):]
        return F.silu(conv(xc)).transpose(1, 2), new_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value=None,
    ) -> AttentionOutput:
        '''
        Run gated delta-net attention over `hidden_states`.

        A single-token step (`L == 1`) always uses the recurrent formulation; for
        longer chunks the fused Triton kernel runs when Triton is enabled and the
        head dimensions are supported, otherwise the pure-PyTorch chunked
        implementation is used. The delta-rule maths is done in fp32 and cast back
        to the input dtype afterwards.

        Args:
            hidden_states (torch.Tensor): `[B, L, hidden_size]` input features.
            past_key_value (Optional[GatedDeltaAttentionLayerCache]): Recurrent
                state and convolution cache to continue from; the updated cache is
                returned in the output.

        Returns:
            AttentionOutput: Output tensor, plus either the updated cache (when one
            was passed in) or a payload holding the detached recurrent state and
            convolution cache.

        Raises:
            TypeError: If `past_key_value` is not a
                `GatedDeltaAttentionLayerCache`.
        '''
        B, L, _ = hidden_states.shape

        use_cache = past_key_value is not None
        if use_cache:
            if not isinstance(past_key_value, GatedDeltaAttentionLayerCache):
                raise TypeError(
                    f'GatedDeltaAttention requires GatedDeltaAttentionLayerCache, '
                    f'got {type(past_key_value).__name__}'
                )
            conv_cache = (past_key_value.conv_state if past_key_value.conv_state is not None
                          else (None, None, None))
            S0 = past_key_value.state
        else:
            conv_cache = (None, None, None)
            S0 = None

        q, cq = self._conv(self.q_proj(hidden_states), self.q_conv, conv_cache[0])
        k, ck = self._conv(self.k_proj(hidden_states), self.k_conv, conv_cache[1])
        v, cv = self._conv(self.v_proj(hidden_states), self.v_conv, conv_cache[2])

        q = q.view(B, L, self.num_heads, self.dk).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.dk).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.dv).transpose(1, 2)
        q = F.normalize(q, dim=-1) * (self.dk ** -0.5)
        k = F.normalize(k, dim=-1)

        beta = torch.sigmoid(self.b_proj(hidden_states)).transpose(1, 2)
        g = -self.A_log.exp() * F.softplus(self.a_proj(hidden_states) + self.dt_bias)
        g = g.transpose(1, 2)

        qf, kf, vf, bf, gf = (t.float() for t in (q, k, v, beta, g))
        if L == 1:
            o, S = recurrent_gated_delta_rule(qf, kf, vf, bf, gf, S0)
        elif (self.use_triton and self.dk in (32, 64, 128)
              and self.dv % 64 == 0 and self.chunk_size >= 16):
            o, S = chunk_gated_delta_rule_triton(qf, kf, vf, bf, gf, S0, self.chunk_size)
        else:
            o, S = chunk_gated_delta_rule(qf, kf, vf, bf, gf, S0, self.chunk_size)

        o = o.to(hidden_states.dtype).transpose(1, 2)
        gate = self.g_proj(hidden_states).view(B, L, self.num_heads, self.dv)

        o = self.o_norm(torch.cat([o, gate.to(o.dtype)], dim=-1)).reshape(B, L, -1)
        output = self.o_proj(o)

        if use_cache:
            past_key_value.update(S, (cq, ck, cv), steps=L)
            return AttentionOutput(output=output, past_key_value=past_key_value)
        payload = {
            'state': S.detach(),
            'conv_state': tuple(c.detach() for c in (cq, ck, cv)),
        }
        return AttentionOutput(output=output, payload=payload)
