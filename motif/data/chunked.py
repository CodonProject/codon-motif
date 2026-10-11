from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

import torch

from motif.data.base import MotifDataset, MotifIterableDataset
from motif.utils.sharding import row_slice


class ChunkedTokenStream(MotifIterableDataset):
    '''
    Concatenates variable-length token records into a buffer, separated by a
    **document boundary token**, and cuts a chunk out as soon as `chunk_len` tokens
    have accumulated. Each chunk is reshaped to `[batch_size, seq_len + 1]` and split
    into `(input_ids, labels)`: `input_ids` takes the first `seq_len` columns and
    `labels` the last `seq_len` columns (i.e. shifted right by one).

    What is appended after every record is `boundary_token_id` (the pipeline resolves
    it to `<|end_of_text|>` by default). It plays two roles at once: it marks "a new
    document starts here", and - disabled by default - it can be counted in the loss
    as a prediction target.

    With `mask_boundary=True` the label positions sitting on the boundary token are
    written as -100: the model still sees the separator (it is part of `input_ids`)
    but **only the body text contributes to the loss**. On the pipeline side this is
    controlled by `PretrainConfig.mask_boundary_token` (True by default).

    Resume semantics: `state_dict()` stores the "token buffer not yet consumed" and
    the "upstream record cursor". After restoring, the leftover tokens in the buffer
    are consumed first and the upstream continues from the cursor - records before
    the checkpoint are not tokenized again, and records after it are not skipped.

    The boundary token itself is part of `state_dict()` as well: switching separators
    mid-run would leave old records in the buffer with the old separator and new
    records with the new one (the boundary semantics the model sees would break), so
    `load_state_dict()` raises when the two disagree instead of silently mixing them.

    Data parallelism: with `shard_world > 1` every rank still walks the **same**
    stream and consumes the **same** buffer, and only the chunk it hands out is
    narrowed to its own contiguous row slice. Two consequences are worth being
    explicit about:

      * the shard is a pure function of the current chunk, so `state_dict()` does not
        have to record the rank layout - a checkpoint written by 1 rank resumes on 4
        ranks and vice versa without skipping or replaying a token;
      * the upstream work (reading and tokenizing records) is repeated by every rank,
        which is the price of keeping the cursors identical. It buys a smaller batch
        per rank, i.e. the actual speed-up.

    Attributes:
        data: Upstream record source. A `MotifDataset` is seeked by index (no cost);
            any other iterable degrades to skipping with `next()`.
        chunk_len (int): Number of tokens per chunk; must equal
            `batch_size * (seq_len + 1)`.
        batch_size (int): Number of rows per micro-batch.
        seq_len (int): Sequence length of a row.
        boundary_token_id (int): Document boundary token appended after every record.
        mask_boundary (bool): True = write -100 at the boundary-token positions of
            the labels (only the body text is trained on).
        ignore_index (int): Label value used for masking (-100 by default, matching
            `F.cross_entropy`).
        eos_token_id (int): Fallback used when no boundary token is given explicitly
            (equivalent to the older behaviour).
        shard_rank (int): Index of the row slice this stream hands out.
        shard_world (int): Number of ranks the chunk is split over; 1 disables the
            split.
    '''

    def __init__(
        self,
        data: Union[Iterable[Any], MotifDataset],
        chunk_len: int,
        batch_size: int,
        seq_len: int,
        eos_token_id: int,
        boundary_token_id: Optional[int] = None,
        mask_boundary: bool = True,
        ignore_index: int = -100,
        shard_rank: int = 0,
        shard_world: int = 1,
    ) -> None:
        '''
        Build the stream and validate the chunk geometry.

        Args:
            data (Union[Iterable[Any], MotifDataset]): Upstream records.
            chunk_len (int): Tokens per chunk.
            batch_size (int): Rows per micro-batch.
            seq_len (int): Sequence length of a row.
            eos_token_id (int): Fallback boundary token.
            boundary_token_id (Optional[int]): Document boundary token; defaults to
                `eos_token_id`.
            mask_boundary (bool): Mask the boundary positions in the labels.
            ignore_index (int): Label value used for masking.
            shard_rank (int): Row slice handed out by this stream (data parallelism);
                0 when the chunk is not split.
            shard_world (int): Number of ranks the chunk is split over; 1 keeps the
                chunk whole.

        Raises:
            ValueError: If `chunk_len != batch_size * (seq_len + 1)`, if `shard_world`
                is below 1 or `shard_rank` is outside `[0, shard_world)`, or if the
                chunk has fewer rows than there are ranks (so some rank would get
                nothing to train on).
        '''
        expected = batch_size * (seq_len + 1)
        if chunk_len != expected:
            raise ValueError(
                f'chunk_len ({chunk_len}) must equal batch_size * (seq_len + 1) '
                f'= {expected}.'
            )
        if shard_world < 1 or not 0 <= shard_rank < shard_world:
            raise ValueError(
                f'shard_rank/shard_world must satisfy 0 <= rank < world, got '
                f'{shard_rank}/{shard_world}.'
            )
        if batch_size < shard_world:
            raise ValueError(
                f'batch_size ({batch_size}) is smaller than the number of ranks '
                f'({shard_world}), so some rank would receive an empty chunk and the '
                f'ranks would fall out of step. Raise `global_batch_tokens` to at '
                f'least {shard_world * (seq_len + 1)} tokens per step (a '
                f'`PretrainPipeline` does that automatically while '
                f'`auto_widen_batch` is on), or use fewer ranks.'
            )
        self.data = data
        self.chunk_len = chunk_len
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.eos_token_id = int(eos_token_id)
        self.boundary_token_id = (
            self.eos_token_id if boundary_token_id is None else int(boundary_token_id)
        )
        self.mask_boundary = bool(mask_boundary)
        self.ignore_index = int(ignore_index)
        self.shard_rank = int(shard_rank)
        self.shard_world = int(shard_world)

        self._token_buffer: List[int] = []
        self._upstream_offset: int = 0

    def state_dict(self) -> Dict[str, Any]:
        '''
        Save the token buffer, the upstream cursor and the boundary token, so that
        chunks do not shift on a resume.

        Returns:
            Dict[str, Any]: The picklable state.
        '''
        return {
            'token_buffer': list(self._token_buffer),
            'upstream_offset': int(self._upstream_offset),
            'boundary_token_id': int(self.boundary_token_id),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        '''
        Restore the token buffer and the upstream cursor.

        Args:
            state (Dict[str, Any]): State produced by `state_dict()`.

        Raises:
            ValueError: If the checkpoint was written with a different boundary
                token, which would silently mix two separator semantics.
        '''
        saved = state.get('boundary_token_id')
        if saved is not None and int(saved) != self.boundary_token_id:
            raise ValueError(
                f'the checkpoint uses document boundary token {int(saved)} while '
                f'this stream uses {self.boundary_token_id}. Switching separators '
                f'mid-run would leave old records in the buffer using a different '
                f'boundary token than new records (the boundary semantics get '
                f'confused); align boundary_token_id to {int(saved)} before '
                f'resuming.'
            )
        self._token_buffer = list(state.get('token_buffer', []))
        self._upstream_offset = int(state.get('upstream_offset', 0))

    @staticmethod
    def _extract_ids(record: Any) -> List[int]:
        '''
        Turn a single record into a list of token ids (a dict is read through
        `input_ids`, anything else is treated as a sequence).

        Args:
            record (Any): One upstream record.

        Returns:
            List[int]: The token ids of that record.
        '''
        if isinstance(record, dict):
            return list(record['input_ids'])
        return list(record)

    @staticmethod
    def _normalize_item(item: Any) -> Iterator[List[int]]:
        '''
        Turn one upstream item into one or more token-id lists.

        The upstream may hand over a single record or a whole batch of them (for
        example `TextFileDataset` returns several segments when a tokenizer is
        configured). Record boundaries have to be preserved, because a boundary token
        is appended after every record.

        Args:
            item (Any): One item produced by the upstream iterator.

        Yields:
            List[int]: The token ids of one record.
        '''
        if isinstance(item, dict):
            yield list(item['input_ids'])
            return
        if (
            isinstance(item, (list, tuple))
            and len(item) > 0
            and isinstance(item[0], (list, tuple, dict))
        ):
            for record in item:
                yield ChunkedTokenStream._extract_ids(record)
            return
        yield list(item)

    def _drain_buffer(self) -> Iterator[Any]:
        '''
        Cut one `(input_ids, labels)` tensor pair out of the buffer per full chunk.

        With `mask_boundary=True`, label positions sitting on a **boundary token** are
        written as -100 (`ignore_index` of `F.cross_entropy`), i.e. "only the body
        text is trained on": the boundary token is still part of the input (the model
        sees the context boundary) but is not required as a prediction target. This
        masking is derived directly from the chunk tensor (by comparing ids), so it
        needs **no extra state** and the resume semantics are unchanged.

        Under data parallelism the chunk is narrowed to this rank's contiguous row
        slice first, so `shard_world` ranks together still see every row exactly once.
        The buffer and the cursor are advanced identically on every rank regardless.

        Yields:
            tuple: `(input_ids, labels)`, both `[rows, seq_len]` where `rows` is
            `batch_size` (or this rank's share of it).
        '''
        while len(self._token_buffer) >= self.chunk_len:
            chunk = self._token_buffer[:self.chunk_len]
            self._token_buffer = self._token_buffer[self.chunk_len:]
            tensor = torch.tensor(chunk, dtype=torch.long).view(
                self.batch_size, self.seq_len + 1
            )
            if self.shard_world > 1:
                start, end = row_slice(tensor.size(0), self.shard_rank, self.shard_world)
                tensor = tensor[start:end]
            labels = tensor[:, 1:].contiguous()
            if self.mask_boundary:
                labels = labels.masked_fill(labels == self.boundary_token_id,
                                            self.ignore_index)
            yield tensor[:, :-1].contiguous(), labels

    def iter_from(self, offset: int) -> Iterator[Any]:
        '''
        Stream `(input_ids, labels)` pairs, continuing from the internal cursor
        (max'ed with `offset`).

        `offset` is measured in upstream items (records for a map-style dataset, the
        number of `next()` calls otherwise). A token buffer brought back by this
        object's own `state_dict()` is reused as it is.

        Args:
            offset (int): Lower bound of the upstream item offset.

        Yields:
            tuple: `(input_ids, labels)` tensor pairs.
        '''
        self._upstream_offset = max(self._upstream_offset, int(offset))

        if isinstance(self.data, MotifDataset):
            total = len(self.data)
            i = self._upstream_offset
            while i < total:
                for ids in self._normalize_item(self.data[i]):
                    self._token_buffer.extend(ids)
                    self._token_buffer.append(self.boundary_token_id)
                self._upstream_offset = i + 1
                i += 1
                yield from self._drain_buffer()
            return

        upstream = iter(self.data)
        for _ in range(self._upstream_offset):
            try:
                next(upstream)
            except StopIteration:
                return
        for item in upstream:
            for ids in self._normalize_item(item):
                self._token_buffer.extend(ids)
                self._token_buffer.append(self.boundary_token_id)
            self._upstream_offset += 1
            yield from self._drain_buffer()
