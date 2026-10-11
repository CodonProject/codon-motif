import torch
import torch.nn as nn

from typing import Literal, Optional, List
from dataclasses import dataclass

from motif.block.attention import (
    AttentionOutput,
    ChordAttention, GatedDeltaAttention
)
from motif.block.mlp    import SwiGLU
from motif.ops.attn_res import apply_block_attn_res
from motif.ops.rms_norm import RMSNorm
from motif.model.cache  import (
    BasicLayerCache,
    KVLayerCache,
    GatedDeltaAttentionLayerCache,
    DecoderBlockCache,
)
from motif.model.config import MotifChordConfig


@dataclass
class DecoderBlockOutput:
    '''
    Output of a `DecoderBlock`.

    Attributes:
        hidden_states (torch.Tensor): `[B, T, D]` block representation, i.e. the
            running sum of the attention-residual history (`blocks[1:]`).
        cache (DecoderBlockCache): Cache holding the attention-residual state and
            every per-decoder attention cache.
    '''

    hidden_states: torch.Tensor
    cache: DecoderBlockCache

    @property
    def blocks(self) -> List[torch.Tensor]:
        '''The attention-residual history, oldest first.'''
        return self.cache.attn_res.blocks


class Decoder(nn.Module):
    '''
    One decoder layer: pre-norm attention followed by a pre-norm SwiGLU MLP.

    Both sub-layers are added onto the incoming hidden states as residuals.

    Attributes:
        attn_type (str): 'chord' or 'gdn', lower-cased.
        attn_norm (RMSNorm): Pre-norm applied before the attention layer.
        attn (nn.Module): `ChordAttention` or `GatedDeltaAttention`.
        mlp_norm (RMSNorm): Pre-norm applied before the MLP.
        mlp (SwiGLU): The pre-norm SwiGLU MLP.
    '''

    def __init__(
        self,
        attn_type: Literal['gdn', 'chord'],
        config: MotifChordConfig
    ):
        '''
        Build the sub-layers for the requested attention type.

        Args:
            attn_type (Literal['gdn', 'chord']): Which attention to instantiate;
                'chord' selects `ChordAttention`, anything else
                `GatedDeltaAttention`.
            config (MotifChordConfig): Model hyper-parameters.
        '''
        super().__init__()

        self.attn_type = attn_type.lower()

        self.attn_norm = RMSNorm(config.model_dim)

        self.attn = ChordAttention(
            hidden_size=config.model_dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads
        ) if self.attn_type == 'chord' else GatedDeltaAttention(
            hidden_size=config.model_dim,
            num_heads=config.num_heads,
            head_k_dim=config.head_k_dim,
            expand_v=2.0,
            chunk_size=64,
            bias=False
        )

        self.mlp_norm = RMSNorm(config.model_dim)

        self.mlp = SwiGLU(
            in_features=config.model_dim
        )

    @property
    def cache_type(self) -> BasicLayerCache:
        '''
        Returns:
            BasicLayerCache: The cache class matching `attn_type`
            (`KVLayerCache` for chord, `GatedDeltaAttentionLayerCache` otherwise).
        '''
        return KVLayerCache if self.attn_type == 'chord' else GatedDeltaAttentionLayerCache


class DecoderBlock(nn.Module):
    '''
    A stack of `Decoder`s joined by cross-block attention residuals.

    Attributes:
        features (nn.ModuleList): The decoders, one per entry of `config.block`.
        pseudo_queries (nn.ParameterList): One learnable query per decoder, used by
            the attention-residual op.
        attn_res_norms (nn.ModuleList): RMSNorm applied to each attention-residual
            read before the decoder consumes it.
    '''

    def __init__(self, config: MotifChordConfig):
        '''
        Build the decoder stack.

        Args:
            config (MotifChordConfig): Model hyper-parameters; `config.block` fixes
                the number of decoders and their attention types.
        '''
        super().__init__()
        self.features = nn.ModuleList([
            Decoder(attn_type=attn_type, config=config)
            for attn_type in config.block
        ])
        self.pseudo_queries = nn.ParameterList([
            nn.Parameter(torch.zeros(config.model_dim)) for _ in self.features
        ])
        self.attn_res_norms = nn.ModuleList([
            RMSNorm(config.model_dim) for _ in self.features
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[DecoderBlockCache] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> DecoderBlockOutput:
        '''
        Run every decoder of the block, threading the attention-residual state.

        Args:
            hidden_states (torch.Tensor): `[B, T, D]` inputs of the block.
            cache (Optional[DecoderBlockCache]): Cache to reuse; a fresh one is
                created when None. Outside training mode the per-decoder attention
                caches are instantiated lazily on the first call.
            mask (Optional[torch.Tensor]): Keep-mask over the key axis, `True` on
                positions that may be attended (`[B, T]`, one row per batch row).
                Only the attention layers that declare it receive it; the recurrent
                layers (`gdn`) carry their own state and ignore it.

        Returns:
            DecoderBlockOutput: The block representation and the updated cache.

        Raises:
            ValueError: If the cache holds a different number of attention slots
                than the block has decoders.
        '''
        if cache is None:
            cache = DecoderBlockCache()

        attn_res = cache.attn_res
        attn_res.init_embedding(hidden_states)

        if len(cache.attn) == 0:
            if not self.training:
                cache.ensure_attn(
                    [decoder.cache_type for decoder in self.features]
                )
        elif len(cache.attn) != len(self.features):
            raise ValueError(
                f'cache.attn has {len(cache.attn)} entries but the block has '
                f'{len(self.features)} layers'
            )

        use_attn_cache = len(cache.attn) == len(self.features)

        for i, decoder in enumerate(self.features):
            h = apply_block_attn_res(
                attn_res,
                self.pseudo_queries[i],
                self.attn_res_norms[i],
            )

            if isinstance(decoder.attn, ChordAttention):
                attn_out: AttentionOutput = decoder.attn(
                    hidden_states=decoder.attn_norm(h),
                    past_key_value=cache.attn[i] if use_attn_cache else None,
                    attention_mask=mask,
                )
            else:
                attn_out = decoder.attn(
                    hidden_states=decoder.attn_norm(h),
                    past_key_value=cache.attn[i] if use_attn_cache else None,
                )
            mlp_out = decoder.mlp(decoder.mlp_norm(h))
            f_h = attn_out.output + mlp_out

            if use_attn_cache:
                cache.attn[i] = attn_out.past_key_value

            attn_res.update_increment(f_h)

        block_repr = attn_res.partial_block
        if block_repr is None:
            block_repr = hidden_states

        return DecoderBlockOutput(
            hidden_states=block_repr,
            cache=cache,
        )
