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


def _widen_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    '''
    Repeat every key / value head so the tensor carries one copy per query head.

    The repeat is the one `ChordAttention` would do on the way out (head `h` of a
    grouped projection serves `num_heads // num_kv_heads` consecutive query heads),
    applied here to the incoming token instead.

    Args:
        x (torch.Tensor): Grouped keys or values, `[B, H_kv, T, D]`.
        num_heads (int): Number of query heads to store per position.

    Returns:
        torch.Tensor: `[B, num_heads, T, D]`, unchanged when already wide.

    Raises:
        ValueError: If `num_heads` is not a multiple of the grouped head count.
    '''
    batch_size, kv_heads, length, head_dim = x.shape
    if kv_heads == num_heads:
        return x
    if num_heads % kv_heads:
        raise ValueError(
            f'cannot store {num_heads} heads from {kv_heads} grouped heads: the '
            f'query-head count must be a multiple of the key / value head count')
    queries = num_heads // kv_heads
    return x[:, :, None].expand(
        batch_size, kv_heads, queries, length, head_dim,
    ).reshape(batch_size, num_heads, length, head_dim)


class KVLayerCache(BasicLayerCache):
    '''
    Key / value cache of the chord attention layer.

    The buffers are allocated with slack and grown geometrically, so appending a
    single token while decoding does not reallocate on every step.

    Passing `num_heads` makes the cache **wide**: the grouped keys / values are
    expanded to every query head as they are written, so the read side sees a cache
    that is already expanded and skips the grouped-query expansion. That costs
    `num_heads / num_kv_heads` times the memory and buys nothing on eager torch - it
    exists for traced graphs, where the expansion across a *dynamic* sequence length
    is the one operation `torch.export` cannot guard. The expansion itself is cheap
    to trace because it only ever sees the new tokens, whose count is static.

    Attributes:
        k (Optional[torch.Tensor]): Cached keys, `[B, H, capacity, D]`.
        v (Optional[torch.Tensor]): Cached values, `[B, H, capacity, D]`.
        length (int): Number of valid positions in `k` / `v`.
        _owned (bool): True once the buffers are owned by the cache, False while they
            still alias the tensors handed in by the first `update` call.
        num_heads (Optional[int]): Heads stored per position, or None to keep the
            grouped layout of the projection and expand on read.
    '''

    def __init__(self, num_heads: Optional[int] = None):
        '''
        Start from an empty cache.

        Args:
            num_heads (Optional[int]): Query heads to expand every key / value to, or
                None for the narrow (grouped) layout.
        '''
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self.length: int = 0
        self._owned: bool = False
        self.num_heads: Optional[int] = None if num_heads is None else int(num_heads)

    @property
    def wide(self) -> bool:
        '''
        Returns:
            bool: True when the cache stores one copy of every head.
        '''
        return self.num_heads is not None

    def _prepare(self, k_new: torch.Tensor, v_new: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Expand new keys / values to every head when this cache is wide.

        Args:
            k_new (torch.Tensor): New keys, `[B, H_kv, T, D]`.
            v_new (torch.Tensor): New values, `[B, H_kv, T, D]`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: `[B, num_heads, T, D]` when wide, the
            inputs unchanged when narrow.
        '''
        if self.num_heads is None:
            return k_new, v_new
        return (_widen_heads(k_new, self.num_heads),
                _widen_heads(v_new, self.num_heads))

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
        copy into the cache's own buffers. A wide cache expands the new keys /
        values to every head before storing them.

        Args:
            k_new (torch.Tensor): New keys.
            v_new (torch.Tensor): New values.
            dim (int): Sequence dimension of the buffers (2 by default).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: All cached keys and values, including
            the new ones.
        '''
        k_new, v_new = self._prepare(k_new, v_new)
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
        heads = '' if self.num_heads is None else f', heads={self.num_heads}'
        return f'KVLayerCache(seq_length={self.length}, owned={self._owned}{heads})'


class FunctionalKVLayerCache(KVLayerCache):
    '''
    Key / value cache that never writes its buffers in place.

    `update` returns a concatenation instead of copying into a growing buffer, so the
    cached keys / values are plain function inputs and outputs: the same call can be
    traced into a graph (and exported) and replayed with any sequence length, which
    the in-place `KVLayerCache` cannot do.
    '''

    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        dim: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Append new keys / values and return the whole cached sequence.

        Args:
            k_new (torch.Tensor): New keys.
            v_new (torch.Tensor): New values.
            dim (int): Sequence dimension of the buffers (2 by default).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: All cached keys and values, including
            the new ones.
        '''
        k_new, v_new = self._prepare(k_new, v_new)
        if self.k is None:
            self.k = k_new.detach()
            self.v = v_new.detach()
        else:
            if self.k.device != k_new.device:
                self.to(k_new.device)
            self.k = torch.cat([self.k, k_new.detach()], dim=dim)
            self.v = torch.cat([self.v, v_new.detach()], dim=dim)
        self.length = self.k.shape[dim]
        self._owned = True
        return self.k, self.v

    def __repr__(self):
        '''
        Returns:
            str: A short debug representation.
        '''
        heads = '' if self.num_heads is None else f', heads={self.num_heads}'
        return f'FunctionalKVLayerCache(seq_length={self.length}{heads})'


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
        (decode, or a new chunk) it replaces it and every representation left behind
        by the previous forward is dropped. `seq_len` keeps counting the total number
        of positions the block has seen, because the historical part is never read
        again: `_align` only ever looks at the tail of the current chunk, so keeping
        the history would only cost memory and a per-step copy.

        The input keeps its graph, so continuing from a cache during training still
        propagates gradients back to the current input, while nothing of the previous
        forward is held by the cache.

        Args:
            hidden_states (torch.Tensor): `[B, q_len, D]` block input.
        '''
        q_len = hidden_states.shape[1]
        if len(self.blocks) == 0:
            self.seq_len = q_len
        else:
            self.seq_len = self.seq_len + q_len
            self.partial_block = None
        self.blocks = [hidden_states]
        self.chunk_len = q_len

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


def build_cache(
    model,
    use_attn: Optional[bool] = None,
    functional: bool = False,
    wide_kv: bool = False,
) -> ModelCache:
    '''
    Build the cache of a whole model.

    Args:
        model: The model to build for; only its `training` flag, its blocks and the
            cache type of every layer are read, so any object with the same attributes
            works.
        use_attn (Optional[bool]): Whether the per-layer attention caches are
            instantiated; None instantiates them outside training mode, which is what
            generation and export want.
        functional (bool): Keep keys / values by concatenation
            (`FunctionalKVLayerCache`) instead of copying into a growing buffer, so the
            cache can be an input and an output of a traced graph.
        wide_kv (bool): Expand every key / value to all query heads as it is written
            (`KVLayerCache(num_heads=...)`), which removes the grouped-query expansion
            from the graph - the one operation a dynamic sequence length cannot trace -
            at `num_heads / num_kv_heads` times the cache memory.

    Returns:
        ModelCache: One `DecoderBlockCache` per block, with the layer caches
        instantiated when `use_attn` is true.
    '''
    if use_attn is None:
        use_attn = not getattr(model, 'training', False)
    if not use_attn:
        return ModelCache.create(len(model.blocks))

    blocks = []
    for block in model.blocks:
        attn: List[BasicLayerCache] = []
        for decoder in block.features:
            cache_type = decoder.cache_type
            if functional and issubclass(cache_type, KVLayerCache):
                cache_type = FunctionalKVLayerCache
            if wide_kv and issubclass(cache_type, KVLayerCache):
                attn.append(cache_type(num_heads=decoder.attn.num_heads))
            else:
                attn.append(cache_type())
        blocks.append(DecoderBlockCache(attn=attn))
    return ModelCache(blocks=blocks)
