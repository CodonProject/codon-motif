import torch
import torch.nn as nn

from motif.model.cache import BasicLayerCache

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AttentionOutput:
    '''
    What an attention layer returns.

    Attributes:
        output (torch.Tensor): `[B, L, hidden_size]` attention output.
        attention_weights (Optional[torch.Tensor]): Softmax weights when the caller
            asked for them, None otherwise.
        past_key_value (Optional[Any]): The updated per-layer cache, when the layer
            was called with one.
        payload (Optional[dict]): Extra per-layer tensors, for example the detached
            recurrent state of the gated delta-net layer.
    '''

    output: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    past_key_value: Optional[Any] = None
    payload: Optional[dict] = None


class BasicAttention(nn.Module):
    '''
    Base class of every attention layer.

    Attributes:
        hidden_size (int): Width of the input / output representation.
        num_heads (int): Number of query heads.
        num_kv_heads (int): Number of key / value heads.
        head_dim (int): Dimension of one head, `hidden_size // num_heads`.
        kv_dim (int): Flattened dimension of the key / value heads,
            `num_kv_heads * head_dim`.
        num_kv_queries (int): How many query heads share one key / value head,
            `num_heads // num_kv_heads`.
    '''

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        **kwargs,
    ):
        '''
        Derive the head layout and validate the head counts.

        Args:
            hidden_size (int): Width of the input / output representation.
            num_heads (int): Number of query heads.
            num_kv_heads (Optional[int]): Number of key / value heads; None means
                one per query head.
            **kwargs: Accepted for signature compatibility and ignored.

        Raises:
            AssertionError: If `hidden_size` is not divisible by `num_heads`, or
                `num_heads` is not divisible by `num_kv_heads`.
        '''
        super().__init__()

        num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        assert hidden_size % num_heads == 0, \
            f'hidden_size {hidden_size} must be divisible by num_heads {num_heads}'
        assert num_heads % num_kv_heads == 0, \
            f'num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}'

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.num_kv_queries = num_heads // num_kv_heads

    @classmethod
    def cache_type(cls):
        '''
        Returns:
            type: The cache class a layer of this kind needs; the default is
            `BasicLayerCache`.
        '''
        return BasicLayerCache

    def reshape_q(self, q: torch.Tensor, q_len: int, batch_size: int) -> torch.Tensor:
        '''
        Reshape flat queries into head layout.

        Args:
            q (torch.Tensor): `[B, L, H*D]` flattened queries.
            q_len (int): Sequence length `L`.
            batch_size (int): Batch size `B`.

        Returns:
            torch.Tensor: `[B, H, L, D]`.
        '''
        return q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    def reshape_kv(self, x: torch.Tensor, kv_len: int, batch_size: int) -> torch.Tensor:
        '''
        Reshape flat keys / values into head layout.

        Args:
            x (torch.Tensor): `[B, L, H_kv*D]` flattened keys or values.
            kv_len (int): Sequence length `L`.
            batch_size (int): Batch size `B`.

        Returns:
            torch.Tensor: `[B, H_kv, L, D]`.
        '''
        return x.view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
