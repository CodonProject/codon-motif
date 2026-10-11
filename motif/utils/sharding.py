from typing import Any, Dict, Optional, Tuple

import torch

__all__ = ['row_slice', 'slice_rows', 'shard_batch']


def row_slice(total_rows: int, rank: int, world_size: int) -> Tuple[int, int]:
    '''
    Contiguous split of `total_rows` rows over `world_size` ranks.

    Args:
        total_rows (int): Rows to distribute.
        rank (int): Rank asking for its slice.
        world_size (int): Number of ranks; 1 keeps every row.

    Returns:
        Tuple[int, int]: The half-open row interval `[start, end)` of this rank.
    '''
    if world_size <= 1:
        return 0, int(total_rows)
    per_rank, remainder = divmod(int(total_rows), int(world_size))
    start = rank * per_rank + min(rank, remainder)
    count = per_rank + (1 if rank < remainder else 0)
    return start, start + count


def slice_rows(value: Any, start: int, end: int, total_rows: Optional[int] = None) -> Any:
    '''
    Slice (possibly nested) tensors and sequences along their batch dimension.

    Only the parts that are demonstrably per-row are touched: a tensor is sliced when
    its leading dimension is the batch (`total_rows`), and a list is sliced when it
    holds one entry per row. Anything else - a scalar, a string of metadata - is kept as
    it is, because guessing wrong is worse than not sharding an auxiliary field.

    Args:
        value (Any): Tensor, list, tuple, or anything else.
        start (int): First row kept.
        end (int): One past the last row kept.
        total_rows (Optional[int]): Row count of the batch; None slices every tensor
            with a leading dimension and every sequence.

    Returns:
        Any: The same structure, restricted to the selected rows.
    '''
    if isinstance(value, torch.Tensor):
        if value.dim() == 0 or (total_rows is not None and value.size(0) != total_rows):
            return value
        return value[start:end]
    if isinstance(value, (list, tuple)):
        if total_rows is not None and len(value) == total_rows:
            return type(value)(value[start:end])
        return type(value)(
            slice_rows(item, start, end, total_rows) for item in value
        )
    return value


def shard_batch(batch: Dict[str, Any], rank: int, world_size: int) -> Dict[str, Any]:
    '''
    Restrict a whole batch dict to the rows of one rank.

    Used by pipelines whose batches are already materialised on every rank: they walk
    the same dataset and slice it afterwards, which keeps every rank's resumable cursor
    identical (`chunked.py` instead slices the chunk before it becomes a batch).

    Nothing is dropped and nothing is duplicated, so the union over the ranks is exactly
    the original batch - which is what makes a multi-GPU step compute the same thing as
    a single-GPU step (the pipelines compensate for the uneven token counts by weighting
    the loss, see `BasicPipeline._scale_loss()`).

    Args:
        batch (Dict[str, Any]): One batch; the batch dimension is measured on the first
            tensor that has one.
        rank (int): Rank asking for its slice.
        world_size (int): Number of ranks.

    Returns:
        Dict[str, Any]: The batch of this rank; `batch` itself when there is one rank
        or when the batch holds no tensor to measure the row count with.
    '''
    if world_size <= 1:
        return batch
    total = None
    for key in ('input_ids', 'labels'):
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            total = int(value.size(0))
            break
    if total is None:
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                total = int(value.size(0))
                break
    if total is None:
        return batch
    start, end = row_slice(total, rank, world_size)
    return {key: slice_rows(value, start, end, total) for key, value in batch.items()}
