from typing import Optional, Tuple, List, Sequence
from dataclasses import dataclass, field

import torch


class BasicLayerCache:
    '''
    Interface that every per-layer attention cache implements.

    The members below are abstract; subclasses provide the actual storage.
    '''

    @property
    def seq_length(self) -> int:
        '''
        Returns:
            int: Number of positions already stored.
        '''
        raise NotImplementedError

    def reset(self) -> None:
        '''
        Returns:
            None: The cache is emptied in place.
        '''
        raise NotImplementedError

    def to(self, device: Optional[torch.device] = None,
           dtype: Optional[torch.dtype] = None):
        '''
        Args:
            device (Optional[torch.device]): Device to move the cached tensors to.
            dtype (Optional[torch.dtype]): Dtype to cast the cached tensors to.

        Returns:
            BasicLayerCache: The cache itself, for chaining.
        '''
        raise NotImplementedError


class KVLayerCache(BasicLayerCache):
    '''
    Key / value cache of the chord attention layer.

    The buffers are allocated with slack and grown geometrically, so appending a
    single token while decoding does not reallocate on every step.

    Attributes:
        k (Optional[torch.Tensor]): Cached keys, `[B, H, capacity, D]`.
        v (Optional[torch.Tensor]): Cached values, `[B, H, capacity, D]`.
        length (int): Number of valid positions in `k` / `v`.
        _owned (bool): True once the buffers are owned by the cache, False while they
            still alias the tensors handed in by the first `update` call.
    '''

    def __init__(self):
        '''
        Start from an empty cache.
        '''
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self.length: int = 0
        self._owned: bool = False

    @property
    def seq_length(self) -> int:
        '''
        Returns:
            int: Number of cached positions.
        '''
        return self.length

    def _grow(self, dim: int, need: int):
        '''
        Reallocate the key / value buffers larger along `dim` and copy the valid
        prefix over.

        The new capacity is `need` rounded up by at least a quarter of the current
        size (and at least 1024 positions).

        Args:
            dim (int): Sequence dimension of the buffers.
            need (int): Minimum capacity required.
        '''
        cap = max(need, self.k.shape[dim] + max(self.k.shape[dim] // 4, 1024))
        shape = list(self.k.shape)
        shape[dim] = cap
        k_buf = self.k.new_empty(shape)
        v_buf = self.v.new_empty(shape)
        sl = [slice(None)] * self.k.dim()
        sl[dim] = slice(0, self.length)
        k_buf[tuple(sl)].copy_(self.k[tuple(sl)])
        v_buf[tuple(sl)].copy_(self.v[tuple(sl)])
        self.k, self.v = k_buf, v_buf
        self._owned = True

    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        dim: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Append new keys / values and return the whole cached sequence.

        The new tensors are detached, so a cache never keeps the graph of the step
        that produced them. The first call aliases the passed tensors, later ones
        copy into the cache's own buffers.

        Args:
            k_new (torch.Tensor): New keys.
            v_new (torch.Tensor): New values.
            dim (int): Sequence dimension of the buffers (2 by default).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: All cached keys and values, including
            the new ones.
        '''
        n_new = k_new.shape[dim]
        if self.k is None:
            self.k = k_new.detach()
            self.v = v_new.detach()
            self.length = n_new
            self._owned = False
            return self.k, self.v

        if self.k.device != k_new.device:
            self.to(k_new.device)
        if not self._owned or self.length + n_new > self.k.shape[dim]:
            self._grow(dim, self.length + n_new)

        end = self.length + n_new
        sl = [slice(None)] * self.k.dim()
        sl[dim] = slice(self.length, end)
        self.k[tuple(sl)].copy_(k_new)
        self.v[tuple(sl)].copy_(v_new)
        self.length = end

        sl[dim] = slice(0, end)
        return self.k[tuple(sl)], self.v[tuple(sl)]

    def reset(self):
        '''
        Drop the cached keys / values.

        Returns:
            None: The cache is emptied in place.
        '''
        self.k = None
        self.v = None
        self.length = 0
        self._owned = False

    def to(self, device=None, dtype=None):
        '''
        Move / cast the cached tensors.

        Args:
            device (Optional[torch.device]): Target device.
            dtype (Optional[torch.dtype]): Target dtype.

        Returns:
            KVLayerCache: The cache itself, for chaining.
        '''
        if self.k is not None:
            self.k = self.k.to(device=device, dtype=dtype)
            self.v = self.v.to(device=device, dtype=dtype)
            self._owned = True
        return self

    def __repr__(self):
        '''
        Returns:
            str: A short debug representation.
        '''
        return f'KVLayerCache(seq_length={self.length}, owned={self._owned})'


class GatedDeltaAttentionLayerCache(BasicLayerCache):
    '''
    Recurrent state cache of the gated delta-net attention layer.

    Attributes:
        state (Optional[torch.Tensor]): Recurrent delta-rule state.
        conv_state (Optional[Tuple[torch.Tensor, ...]]): Convolution state of the
            query / key / value short convolutions.
        length (int): Number of positions already processed.
    '''

    def __init__(self):
        '''
        Start from an empty cache.
        '''
        self.state: Optional[torch.Tensor] = None
        self.conv_state: Optional[Tuple[torch.Tensor, ...]] = None
        self.length: int = 0

    @property
    def seq_length(self) -> int:
        '''
        Returns:
            int: Number of processed positions.
        '''
        return self.length

    def update(
        self,
        state: torch.Tensor,
        conv_state: Tuple[torch.Tensor, ...],
        steps: int = 1,
    ) -> torch.Tensor:
        '''
        Replace the recurrent state with the one produced by the last step.

        Both arguments are detached, so the cache never holds the graph of the step
        that produced them.

        Args:
            state (torch.Tensor): New recurrent state.
            conv_state (Tuple[torch.Tensor, ...]): New convolution state, one tensor
                per q / k / v convolution.
            steps (int): Number of positions covered by this update.

        Returns:
            torch.Tensor: The stored (detached) recurrent state.
        '''
        self.state = state.detach()
        self.conv_state = tuple(c.detach() for c in conv_state)
        self.length += steps
        return self.state

    def reset(self):
        '''
        Drop the recurrent and convolution state.

        Returns:
            None: The cache is emptied in place.
        '''
        self.state = None
        self.conv_state = None
        self.length = 0

    def to(self, device=None, dtype=None):
        '''
        Move / cast the cached tensors.

        Args:
            device (Optional[torch.device]): Target device.
            dtype (Optional[torch.dtype]): Target dtype.

        Returns:
            GatedDeltaAttentionLayerCache: The cache itself, for chaining.
        '''
        if self.state is not None:
            self.state = self.state.to(device=device, dtype=dtype)
        if self.conv_state is not None:
            self.conv_state = tuple(c.to(device=device, dtype=dtype) for c in self.conv_state)
        return self

    def __repr__(self):
        '''
        Returns:
            str: A short debug representation.
        '''
        return (
            f'GatedDeltaAttentionLayerCache(seq_length={self.length}, '
            f'has_state={self.state is not None})'
        )


@dataclass
class AttnResCache:
    '''
    Cross-block attention (attn-res) cache of a single `DecoderBlock`.

    Attributes:
        blocks (List[torch.Tensor]): The attention-residual history of the current
            chunk, oldest first: `blocks[0]` is the block input and `blocks[j]`
            (`j >= 1`) is the increment `f_h` left behind by the `(j - 1)`-th
            decoder, so the `i`-th decoder reads `i + 1` sources. Every entry is
            `[B, q_len, D]`, aligned to the current chunk.
        partial_block (Optional[torch.Tensor]): Running sum of those increments,
            shaped `[B, q_len, D]` - the representation the block hands to the next
            block.
        seq_len (int): Total number of positions this block has processed.
        chunk_len (int): Number of positions fed by the current forward (`q_len`);
            during inference it aligns the cached representations with the tail of
            the current chunk.
    '''

    blocks: List[torch.Tensor] = field(default_factory=list)
    partial_block: Optional[torch.Tensor] = None
    seq_len: int = 0
    chunk_len: int = 0

    def _cat_or_add(self, cached: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        '''
        Combine a cached representation with a new contribution.

        Args:
            cached (torch.Tensor): Existing accumulated representation.
            new (torch.Tensor): New contribution.

        Returns:
            torch.Tensor: `cached + new` when both cover the same number of
            positions, otherwise the two concatenated on `dim=1`.
        '''
        if cached.shape[1] == new.shape[1]:
            return cached + new
        return torch.cat([cached, new], dim=1)

    def _align(self, entries: List[torch.Tensor]) -> List[torch.Tensor]:
        '''
        Trim every entry to the last `q_len` positions of the current chunk.

        The eager path and the fused Triton kernel share this alignment logic: when
        an inference run continues, `blocks[0]` covers the whole history while the
        per-decoder increments only cover the current chunk.

        Args:
            entries (List[torch.Tensor]): Non-empty `[B, T, D]` candidates.

        Returns:
            List[torch.Tensor]: One `[B, q_len, D]` tensor per entry.
        '''
        q_len = self.chunk_len
        if q_len <= 0:
            q_len = min(e.shape[1] for e in entries)
        if any(e.shape[1] != q_len for e in entries):
            entries = [
                e if e.shape[1] == q_len else e[:, max(e.shape[1] - q_len, 0):, :]
                for e in entries
            ]
        return entries

    def aligned_values(self) -> List[torch.Tensor]:
        '''
        Return the whole attention-residual history, aligned to the last `q_len`
        positions of the current chunk (without stacking it).

        The `i`-th decoder of a block reads all `i + 1` entries: the block input plus
        the increment of every decoder before it.

        Returns:
            List[torch.Tensor]: One `[B, q_len, D]` tensor per source.

        Raises:
            RuntimeError: If the cache holds no source at all.
        '''
        if len(self.blocks) == 0:
            raise RuntimeError('AttnResCache is empty, there is no value to read')
        return self._align(list(self.blocks))

    def _stack_values(self) -> torch.Tensor:
        '''
        Returns:
            torch.Tensor: The aligned sources stacked on a new leading dimension.
        '''
        return torch.stack(self.aligned_values(), dim=0)

    def init_embedding(self, hidden_states: torch.Tensor):
        '''
        Register the block input representation.

        On the first forward the input becomes `blocks[0]`; on a continued run
        (decode, or a new chunk) the new positions are appended to `blocks[0]` and
        every representation left behind by the previous forward is dropped.

        Only the historical part is detached: the input of the current chunk keeps
        its graph, so continuing from a cache during training still propagates
        gradients back to the current input, while the stale history graph is never
        held by the cache.

        Args:
            hidden_states (torch.Tensor): `[B, q_len, D]` block input.
        '''
        q_len = hidden_states.shape[1]
        if len(self.blocks) == 0:
            self.blocks.append(hidden_states)
        else:
            base = self.blocks[0]
            self.blocks = [torch.cat([base.detach(), hidden_states], dim=1)]
            self.partial_block = None
        self.chunk_len = q_len
        self.seq_len = self.blocks[0].shape[1]

    def update_increment(self, f_h: torch.Tensor):
        '''
        Register one decoder's increment: append it to the history and fold it into
        the running partial block.

        Appending per decoder (rather than once per block) is what makes the next read
        see every previous increment as its own source. The fold is a plain
        left-to-right sum, so `partial_block` stays numerically identical to the block
        output the cache used to commit.

        Args:
            f_h (torch.Tensor): `[B, q_len, D]` attention plus MLP output of one
                decoder.
        '''
        self.blocks.append(f_h)
        if self.partial_block is None:
            self.partial_block = f_h
        else:
            self.partial_block = self._cat_or_add(self.partial_block, f_h)

        self.seq_len = max(self.seq_len, f_h.shape[1])

    def reset(self):
        '''
        Drop every cached representation.
        '''
        self.blocks.clear()
        self.partial_block = None
        self.seq_len = 0
        self.chunk_len = 0

    def to(self, device: Optional[torch.device] = None,
           dtype: Optional[torch.dtype] = None):
        '''
        Move the whole attn-res cache to the given device / dtype.

        Args:
            device (Optional[torch.device]): Target device.
            dtype (Optional[torch.dtype]): Target dtype.

        Returns:
            AttnResCache: The cache itself, for chaining.
        '''
        self.blocks = [b.to(device=device, dtype=dtype) for b in self.blocks]
        if self.partial_block is not None:
            self.partial_block = self.partial_block.to(device=device, dtype=dtype)
        return self

    def values(self) -> torch.Tensor:
        '''
        Returns:
            torch.Tensor: All sources stacked, aligned to the current chunk.
        '''
        return self._stack_values()

    def num_sources(self) -> int:
        '''
        Returns:
            int: Number of sources the next read sees (`1 +` the number of decoders
            that already ran), or `0` while the cache is empty.
        '''
        return len(self.blocks)

    @property
    def is_empty(self) -> bool:
        '''
        Returns:
            bool: True when no full block representation is cached yet.
        '''
        return len(self.blocks) == 0

    @property
    def current_seq_len(self) -> int:
        '''
        Returns:
            int: Sequence length of the newest cached representation.
        '''
        if self.partial_block is not None:
            return self.partial_block.shape[1]
        if len(self.blocks) > 0:
            return self.blocks[-1].shape[1]
        return 0

    def __repr__(self):
        '''
        Returns:
            str: A short debug representation.
        '''
        return (
            f'AttnResCache(num_blocks={len(self.blocks)}, '
            f'has_partial={self.partial_block is not None}, '
            f'seq_len={self.seq_len})'
        )


@dataclass
class DecoderBlockCache:
    '''
    Complete cache of one `DecoderBlock`.

    Attributes:
        attn_res (AttnResCache): Cross-block attention cache (the history of the
            current chunk plus the running partial sum).
        attn (List[Optional[BasicLayerCache]]): Attention cache of every `Decoder`
            inside the block; once instantiated its length is `len(config.block)` and
            each entry is a `BasicLayerCache` subclass instance (`KVLayerCache` /
            `GatedDeltaAttentionLayerCache`).
    '''

    attn_res: AttnResCache = field(default_factory=AttnResCache)
    attn: List[Optional[BasicLayerCache]] = field(default_factory=list)

    @classmethod
    def create(cls, cache_types: Sequence[type] = ()) -> "DecoderBlockCache":
        '''
        Instantiate every attention cache of this block from the given types.

        Args:
            cache_types (Sequence[type]): Usually the `cache_type` of every
                `Decoder` (`KVLayerCache` / `GatedDeltaAttentionLayerCache`); outside
                training mode its length equals `len(config.block)`.

        Returns:
            DecoderBlockCache: A cache holding one layer cache per type.
        '''
        return cls(attn=[cache_type() for cache_type in cache_types])

    def ensure_attn(self, cache_types: Sequence[type]) -> None:
        '''
        Lazily fill the layer-cache slots (only an empty list is filled; existing
        caches are never overwritten).

        Args:
            cache_types (Sequence[type]): Cache class for every layer.

        Returns:
            None: `self.attn` is populated in place.
        '''
        if len(self.attn) == 0:
            self.attn = [cache_type() for cache_type in cache_types]

    def reset(self):
        '''
        Clear every cache, for starting a new sequence.

        Returns:
            None: The caches are emptied in place.
        '''
        self.attn_res.reset()
        self.attn.clear()

    def is_empty(self) -> bool:
        '''
        Returns:
            bool: True when neither the attn-res cache nor any layer cache holds
            state.
        '''
        return self.attn_res.is_empty and len(self.attn) == 0

    def to(self, device: Optional[torch.device] = None,
           dtype: Optional[torch.dtype] = None):
        '''
        Move this block's attn-res cache and every layer cache together.

        Args:
            device (Optional[torch.device]): Target device.
            dtype (Optional[torch.dtype]): Target dtype.

        Returns:
            DecoderBlockCache: The cache itself, for chaining.
        '''
        self.attn_res.to(device, dtype)
        for layer_cache in self.attn:
            if layer_cache is not None:
                layer_cache.to(device, dtype)
        return self

    @property
    def blocks(self) -> List[torch.Tensor]:
        '''
        Returns:
            List[torch.Tensor]: The attn-res history of this block.
        '''
        return self.attn_res.blocks

    @property
    def seq_len(self) -> int:
        '''
        Returns:
            int: Number of positions this block has processed.
        '''
        return self.attn_res.seq_len

    def __repr__(self):
        '''
        Returns:
            str: A short debug representation.
        '''
        return (
            f"DecoderBlockCache(attn_res={self.attn_res!r}, "
            f"num_attn={len(self.attn)})"
        )


@dataclass
class ModelCache:
    '''
    Cache of the whole `MotifChordModel`, grouped by `DecoderBlock`.

    Attributes:
        blocks (List[DecoderBlockCache]): One entry per block; the length equals
            `config.block_n`.
    '''

    blocks: List[DecoderBlockCache] = field(default_factory=list)

    @classmethod
    def create(cls, num_blocks: int) -> "ModelCache":
        '''
        Create `num_blocks` empty `DecoderBlockCache`s (layer caches are instantiated
        later).

        Args:
            num_blocks (int): Number of decoder blocks in the model.

        Returns:
            ModelCache: The empty cache.
        '''
        return cls(blocks=[DecoderBlockCache() for _ in range(num_blocks)])

    @classmethod
    def create_for(
        cls,
        cache_types_per_block: Sequence[Sequence[type]],
    ) -> "ModelCache":
        '''
        Instantiate every cache at once from the per-block layer types (the attn-res
        cache plus each layer's KV / GD cache); used outside training mode.

        Args:
            cache_types_per_block (Sequence[Sequence[type]]): Cache class per layer,
                for every block.

        Returns:
            ModelCache: The fully instantiated cache.
        '''
        return cls(blocks=[
            DecoderBlockCache.create(cache_types)
            for cache_types in cache_types_per_block
        ])

    def reset(self):
        '''
        Reset every block cache.

        Returns:
            None: The caches are emptied in place.
        '''
        for b in self.blocks:
            b.reset()

    @property
    def seq_len(self) -> int:
        '''
        Number of positions already processed (the attn-res count of the first
        block).

        A continued run uses it to split the input: the first `seq_len` positions of
        `input_ids` are already in the cache and only the rest are new tokens that
        have to be fed.

        Returns:
            int: The number of cached positions.
        '''
        return self.blocks[0].seq_len if self.blocks else 0

    def to(self, device: Optional[torch.device] = None,
           dtype: Optional[torch.dtype] = None):
        '''
        Move the whole model cache to the given device / dtype.

        Args:
            device (Optional[torch.device]): Target device.
            dtype (Optional[torch.dtype]): Target dtype.

        Returns:
            ModelCache: The cache itself, for chaining.
        '''
        for b in self.blocks:
            b.to(device, dtype)
        return self

    def __len__(self):
        '''
        Returns:
            int: Number of block caches.
        '''
        return len(self.blocks)

    def __getitem__(self, idx: int) -> DecoderBlockCache:
        '''
        Args:
            idx (int): Block index.

        Returns:
            DecoderBlockCache: The cache of that block.
        '''
        return self.blocks[idx]

    def __iter__(self):
        '''
        Returns:
            Iterator[DecoderBlockCache]: Iterator over the block caches.
        '''
        return iter(self.blocks)

    def __repr__(self):
        '''
        Returns:
            str: A short debug representation.
        '''
        return f"ModelCache(num_blocks={len(self.blocks)})"
