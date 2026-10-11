import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch
from tqdm import tqdm

from motif.pipeline.callback import Callback


@dataclass
class SnapshotConfig:
    '''
    Weight-snapshot configuration.

    The three `PretrainConfig.snapshot_*` fields are shorthand for this.

    Attributes:
        every_percent (float): Save every this percentage of the total steps; <= 0
            disables snapshots.
        dir (Optional[str]): Output directory; None means `<ckpt_dir>/snapshots`.
        also_at_end (bool): Write one more snapshot when training winds down
            (early stop and interruption included).
        atomic (bool): Write a `.tmp` file first and rename it, so a half-written
            file never appears.
        required (bool): True raises when a snapshot cannot be saved instead of
            degrading gracefully.
    '''

    every_percent: float = 5.0
    dir: Optional[str] = None
    also_at_end: bool = True
    atomic: bool = True
    required: bool = False


def snapshot_targets(total_steps: int, every_percent: float) -> List[int]:
    '''
    Steps at which a snapshot should be written:
    `step = ceil(total_steps * k * pct / 100)`, for `k = 1..floor(100 / pct)`.

    It only depends on `total_steps` and the percentage, so resuming, running on
    several machines and re-running all produce the same target set.

    Args:
        total_steps (int): Total number of optimiser steps of the plan.
        every_percent (float): Snapshot interval, in percent of `total_steps`.

    Returns:
        List[int]: Strictly increasing, de-duplicated target steps; empty when the
        schedule or the percentage is not positive.
    '''
    total = int(total_steps or 0)
    pct = float(every_percent or 0.0)
    if total <= 0 or pct <= 0:
        return []
    targets: List[int] = []
    for k in range(1, int(math.floor(100.0 / pct)) + 1):
        step = min(total, int(math.ceil(total * k * pct / 100.0)))
        if not targets or step > targets[-1]:
            targets.append(step)
    return targets


class SnapshotCallback(Callback):
    '''
    Save bare model weights into `dir` at percentage-based steps.

    It is an ordinary `Callback` and does not touch the training loop:

        PretrainConfig(
            snapshot_every_percent=5,             # every 5% of the total steps
            snapshot_dir='runs/motif/weights',    # None -> <ckpt_dir>/snapshots
        )

    How it differs from a checkpoint - and why it exists at all:

    * **weights only**: `torch.save(model.state_dict(), path)`, with no optimizer,
      scheduler or data-flow state, so the files are far smaller and `torch.load()`
      returns a plain state_dict that analysis scripts can consume directly;
    * **percentage-based targets**: for a plan of 176,672 steps and 5%, the targets are
      step 8834 / 17667 / ... of that set of 20, no matter how training was resumed in
      between (the set follows from `total_steps` and is reproducible);
    * **safe to resume**: a target file that already exists is skipped, so a resumed run
      never rewrites a snapshot that is already on disk.

    The targets are fully determined by `total_steps` and `every_percent`; with
    `also_at_end=True` one more snapshot of the current weights is written when training
    winds down (normal end, early stop or interruption).

    Args:
        config (Optional[Any]): A `SnapshotConfig`, or a number (equivalent to
            `SnapshotConfig(every_percent=n)`), or None for the defaults.
        dir (Optional[str]): Overrides `config.dir`.
        sink: Optional `(tag, text, step)` callback; every saved snapshot reports one
            `Snapshots/save` entry, so snapshots also show up in TensorBoard / the
            run log.

    Raises:
        TypeError: If `config` is neither a `SnapshotConfig`, a number nor None.
    '''

    def __init__(
        self,
        config: Optional[Any] = None,
        dir: Optional[str] = None,
        sink: Optional[Callable[[str, str, int], None]] = None,
    ) -> None:
        '''
        Build the callback.

        Args:
            config (Optional[Any]): `SnapshotConfig`, a number, or None.
            dir (Optional[str]): Directory override.
            sink (Optional[Callable[[str, str, int], None]]): Text sink for the
                snapshot reports.

        Raises:
            TypeError: If `config` has an unsupported type.
        '''
        if config is None:
            config = SnapshotConfig()
        elif isinstance(config, (int, float)) and not isinstance(config, bool):
            config = SnapshotConfig(every_percent=float(config))
        elif not isinstance(config, SnapshotConfig):
            raise TypeError(
                f'SnapshotCallback needs a SnapshotConfig / a number / None, '
                f'got {type(config).__name__}.'
            )
        self.config = config
        self._dir_override = dir
        self.sink = sink

        self._targets: List[int] = []
        self._next = 0
        self._dir: Optional[str] = None
        self._total = 0
        self._disabled = False
        self._saved: List[str] = []

    @property
    def dir(self) -> Optional[str]:
        '''
        Returns:
            Optional[str]: The resolved snapshot directory, once training started.
        '''
        return self._dir

    @property
    def saved(self) -> List[str]:
        '''
        Returns:
            List[str]: Paths of the snapshots written so far, in write order.
        '''
        return list(self._saved)

    def _resolve_dir(self, pipeline) -> str:
        '''
        Resolve the output directory: the explicit override, then `config.dir`, then
        `<ckpt_dir>/snapshots` (or `./snapshots` when the pipeline has no ckpt_dir).

        Args:
            pipeline: The training pipeline.

        Returns:
            str: An absolute directory path.
        '''
        target = self._dir_override or self.config.dir
        if target:
            return os.path.abspath(target)
        ckpt_dir = getattr(pipeline, 'ckpt_dir', None) or '.'
        return os.path.abspath(os.path.join(ckpt_dir, 'snapshots'))

    @staticmethod
    def _weights(pipeline) -> dict:
        '''
        Bare model weights, unwrapping the `torch.compile` and DDP wrappers that would
        prefix the keys with `_orig_mod.` / `module.`.

        Args:
            pipeline: The training pipeline.

        Returns:
            dict: The model state_dict.
        '''
        from motif.pipeline.distributed import unwrap_model

        model = getattr(pipeline, 'raw_model', None)
        if model is None:
            model = unwrap_model(pipeline.model)
        return model.state_dict()

    def _save(self, pipeline, path: str) -> None:
        '''
        Write one bare-weight file, atomically by default.

        A failure either raises (when `config.required`) or disables the plug-in and
        reports it, depending on the configuration.

        Args:
            pipeline: The training pipeline.
            path (str): Destination file.

        Raises:
            RuntimeError: If saving fails and `config.required` is set.
        '''
        directory = os.path.dirname(os.path.abspath(path))
        state = self._weights(pipeline)
        try:
            os.makedirs(directory, exist_ok=True)
            if self.config.atomic:
                fd, tmp = tempfile.mkstemp(dir=directory, suffix='.tmp')
                try:
                    with os.fdopen(fd, 'wb') as f:
                        torch.save(state, f)
                    os.replace(tmp, path)
                except BaseException:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                    raise
            else:
                torch.save(state, path)
        except Exception as exc:
            message = f'[snapshot] failed to save the weight snapshot ({path}): {exc}'
            if self.config.required:
                raise RuntimeError(message) from exc
            self._disabled = True
            tqdm.write(message + '; snapshots disabled, training continues.')
            return
        self._saved.append(path)

    def _emit(self, path: str, step: int, percent: float, written: bool) -> None:
        '''
        Report one snapshot to the console and, when it was actually written, to the
        sink.

        Args:
            path (str): Snapshot file.
            step (int): Step the snapshot belongs to.
            percent (float): Position of that step in the schedule, in percent.
            written (bool): False when the file already existed and was skipped.
        '''
        mark = '' if written else ' (already exists, skipped)'
        message = f'snapshot @ {percent:g}% step {step:,} -> {path}{mark}'
        tqdm.write(f'[*] {message}')
        if self.sink is not None and written:
            try:
                self.sink('Snapshots/save', message, step)
            except Exception:
                pass

    def on_train_start(self, pipeline) -> None:
        '''
        Validate the percentage and compute the target steps for this run.

        Args:
            pipeline: The training pipeline.

        Raises:
            ValueError: If `every_percent` is outside [0, 100].
        '''
        pct = float(self.config.every_percent or 0.0)
        if pct < 0 or pct > 100:
            raise ValueError(f'snapshot every_percent must be within [0, 100], got {pct}')

        self._total = int(getattr(pipeline, 'total_steps', 0) or 0)
        self._dir = self._resolve_dir(pipeline)

        if pct == 0:
            tqdm.write('[snapshot] every_percent=0: only the final snapshot will be saved.')
            return
        if self._total <= 0:
            tqdm.write('[snapshot] total_steps is unavailable, only the final snapshot will be saved.')
            return

        self._targets = snapshot_targets(self._total, pct)
        step = int(pipeline.state.global_step)
        self._next = next((i for i, t in enumerate(self._targets) if t > step),
                          len(self._targets))
        if self._next < len(self._targets):
            tail = f'next target step {self._targets[self._next]:,}'
        else:
            tail = 'all targets already passed'
        tqdm.write(f'[*] weight snapshots: every {pct:g}%, {len(self._targets)} files -> '
                   f'{self._dir} ({tail})')

    def on_step_end(self, pipeline, step_metrics) -> None:
        '''
        Write every target that the current step has reached.

        Args:
            pipeline: The training pipeline.
            step_metrics: Metrics of this step (unused).
        '''
        if self._disabled or self._dir is None:
            return
        step = int(pipeline.state.global_step)
        while self._next < len(self._targets) and step >= self._targets[self._next]:
            target = self._targets[self._next]
            self._next += 1
            path = os.path.join(self._dir, f'step_{target:09d}.pt')
            percent = 100.0 * target / self._total
            if os.path.exists(path):
                self._emit(path, target, percent, written=False)
                continue
            self._save(pipeline, path)
            self._emit(path, target, percent, written=True)

    def on_train_end(self, pipeline) -> None:
        '''
        Write the final snapshot, unless it already exists or `also_at_end` is off.

        Args:
            pipeline: The training pipeline.
        '''
        if self._disabled or not self.config.also_at_end:
            return
        if self._dir is None:
            self._dir = self._resolve_dir(pipeline)
        step = int(pipeline.state.global_step)
        if step <= 0 or (self._saved and self._saved[-1].endswith(f'step_{step:09d}.pt')):
            return
        path = os.path.join(self._dir, f'step_{step:09d}.pt')
        written = not os.path.exists(path)
        if written:
            self._save(pipeline, path)
        percent = 100.0 * step / self._total if self._total else 0.0
        self._emit(path, step, percent, written=written)


def build_snapshot(
    every_percent: Optional[float],
    dir: Optional[str] = None,
    also_at_end: bool = True,
) -> Optional[SnapshotCallback]:
    '''
    Turn the three `PretrainConfig.snapshot_*` fields into a callback.

    Args:
        every_percent (Optional[float]): Snapshot interval in percent; None or <= 0
            switches the plug-in off.
        dir (Optional[str]): Output directory; None means `<ckpt_dir>/snapshots`.
        also_at_end (bool): Also snapshot when training winds down.

    Returns:
        Optional[SnapshotCallback]: The callback, or None when the interval is
        unset / not positive.

    Raises:
        ValueError: If `every_percent` is greater than 100.
    '''
    if every_percent is None or float(every_percent) <= 0:
        return None
    pct = float(every_percent)
    if pct > 100:
        raise ValueError(f'snapshot every_percent must be <= 100, got {pct}')
    return SnapshotCallback(SnapshotConfig(
        every_percent=pct, dir=dir, also_at_end=also_at_end))
