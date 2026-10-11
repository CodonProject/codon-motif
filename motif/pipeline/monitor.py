import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from tqdm import tqdm

from motif.pipeline.callback import Callback, DEFAULT_TAG_MAP


#: Keys written to the per-step progress line by default (in this order; metrics that
#: are not listed here are appended after them).
DEFAULT_PROGRESS_KEYS: Sequence[str] = (
    'loss/train', 'lr', 'grad_norm', 'progress/percent',
    'speed/tokens_per_sec', 'tokens', 'seq_len',
)


def _fmt(value: Any) -> str:
    '''
    Format one metric value so that a log line is both readable and greppable.

    Args:
        value (Any): The value to format.

    Returns:
        str: Booleans and ints as they are, floats in `%.4e` or `%.6g` depending on
        their magnitude, everything else through `str()`.
    '''
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0:
            return '0'
        if abs(value) >= 1e5 or abs(value) < 1e-4:
            return f'{value:.4e}'
        return f'{value:.6g}'
    return str(value)


def _stamp() -> str:
    '''
    Returns:
        str: The current local time as `YYYY-mm-dd HH:MM:SS`.
    '''
    return time.strftime('%Y-%m-%d %H:%M:%S')


@dataclass
class FileLogConfig:
    '''
    Run-log configuration.

    Assigning it to `PretrainConfig.log_file` / `SFTConfig.log_file` switches the log
    on.

    By default the log is **split into two files**:

    * `path`: the progress / event log (per-step progress, checkpoints, probe text,
      the final summary);
    * `sanity_path`: the sanity check (when unset, `sanity.log` next to `path`).

    Set `split_sanity=False` to write everything into one file again.

    TensorBoard scalars and this plain-text run log are ordinary `Callback`s: attach
    them to `callbacks` and the training loop stays untouched.

        PretrainConfig(
            tensorboard=TensorBoardConfig(log_dir='runs/motif'),  # or log_dir='runs/motif'
            log_file='logs/pretrain.log',                         # or FileLogConfig(...)
            probe_every_steps=2000,                                # probe text reaches both
        )

    Three design constraints:

    1. **No hard dependency** - `torch.utils.tensorboard` is imported lazily; when it is
       missing, one hint line is printed and the plug-in degrades silently while
       training continues. Only `TensorBoardConfig(required=True)` turns that into an
       error.
    2. **Never drag training down** - any problem while writing events or log lines only
       disables that plug-in and never propagates.
    3. **The log can be tailed live** - the file is line-buffered and every line is
       `flush()`ed as soon as it is written, so `tail -f logs/pretrain.log` shows the
       progress in real time instead of only at the end of training.

    The log is split by default into a **`sanity.log`** (written once at the start:
    `pipeline.describe()`, whether the run resumes, the starting step) and the progress
    log (default `progress.log`: per-step progress plus checkpoints / evaluate / probe
    text / the final metric snapshot, with a pointer line to `sanity.log`).

    Attributes:
        path (str): Progress / event log.
        sanity_path (Optional[str]): Sanity log; None means `sanity.log` next to
            `path`.
        split_sanity (bool): False writes the sanity check into `path` as well.
        every_steps (int): Write one progress line every this many optimiser steps.
        flush (bool): Flush after every line, so `tail -f` follows it live.
        append (bool): Append on resume by default instead of clearing the old log.
        sanity_check (bool): Write `describe()` plus the curriculum at the start.
        log_eval (bool): Log the results of `evaluate()`.
        log_checkpoints (bool): One line per checkpoint.
        log_text (bool): Log probe / external text (`add_text`).
        progress_keys (Sequence[str]): Key order of the progress line.
        required (bool): True raises when the log file cannot be opened.
    '''

    path: str = 'progress.log'
    sanity_path: Optional[str] = None
    split_sanity: bool = True
    every_steps: int = 1
    flush: bool = True
    append: bool = True
    sanity_check: bool = True
    log_eval: bool = True
    log_checkpoints: bool = True
    log_text: bool = True
    progress_keys: Sequence[str] = field(default_factory=lambda: tuple(DEFAULT_PROGRESS_KEYS))
    required: bool = False


class FileLogger(Callback):
    '''
    Append the sanity check, the per-step progress and events to text files, line by
    line.

    The sanity check goes into `sanity.log` by default (in a long run it only occupies
    the first few lines, and mixed into the progress it is hard to find), while the
    progress and the events go into `path`.

    Args:
        config (Optional[Any]): A `FileLogConfig`, or a plain path string, or None for
            the defaults.
        file: A ready writable handle (for tests or for reusing an open file); when it
            is given, no progress file is opened.
        sanity_file: A ready writable handle for the sanity log, same as above.

    Raises:
        TypeError: If `config` is neither a `FileLogConfig`, a string nor None.
    '''

    def __init__(
        self,
        config: Optional[Any] = None,
        file: Any = None,
        sanity_file: Any = None,
    ) -> None:
        '''
        Build the logger.

        Args:
            config (Optional[Any]): `FileLogConfig`, a path string, or None.
            file (Any): Ready handle for the progress log.
            sanity_file (Any): Ready handle for the sanity log.

        Raises:
            TypeError: If `config` has an unsupported type.
        '''
        if config is None:
            config = FileLogConfig()
        elif isinstance(config, str):
            config = FileLogConfig(path=config)
        elif not isinstance(config, FileLogConfig):
            raise TypeError(
                f'FileLogger needs a FileLogConfig / a path string / None, '
                f'got {type(config).__name__}.'
            )
        self.config = config
        self._file = file
        self._sanity = sanity_file
        self._owned = file is None
        self._owned_sanity = sanity_file is None
        self._disabled = False

    @property
    def file(self):
        '''
        Returns:
            The progress-log handle, once one is open.
        '''
        return self._file

    @property
    def sanity_file(self):
        '''
        Returns:
            The sanity-log handle, once one is open.
        '''
        return self._sanity

    @property
    def path(self) -> str:
        '''
        Returns:
            str: The configured progress-log path.
        '''
        return self.config.path

    @property
    def sanity_path(self) -> Optional[str]:
        '''
        The real path of the sanity log (the progress log itself when
        `split_sanity=False`).

        Returns:
            Optional[str]: The sanity-log path.
        '''
        if not self.config.split_sanity:
            return self.config.path
        if self.config.sanity_path:
            return self.config.sanity_path
        return os.path.join(
            os.path.dirname(os.path.abspath(self.config.path)), 'sanity.log'
        )

    @staticmethod
    def _open_file(path: str, append: bool, flush: bool):
        '''
        Open one log file, creating its directory when needed.

        Args:
            path (str): File to open.
            append (bool): Append instead of truncating.
            flush (bool): Use line buffering so every line reaches the disk at once.

        Returns:
            The open text handle.
        '''
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        return open(path, 'a' if append else 'w', encoding='utf-8',
                    buffering=1 if flush else -1)

    def _open(self) -> bool:
        '''
        Open the progress file and the sanity file.

        A failure degrades gracefully (or raises when `required=True`).

        Returns:
            bool: True when the progress log is available.

        Raises:
            RuntimeError: If opening fails and `config.required` is set.
        '''
        if self._file is None and not self._disabled:
            try:
                self._file = self._open_file(
                    self.config.path, self.config.append, self.config.flush)
            except Exception as exc:
                self._disabled = True
                message = f'[log] cannot open the log file {self.config.path!r} ({exc}).'
                if self.config.required:
                    raise RuntimeError(message) from exc
                tqdm.write(message + ' This run will not write a log.')
                return False

        path = self.sanity_path
        if (self._sanity is None and path and path != self.config.path
                and self._file is not None):
            try:
                self._sanity = self._open_file(path, self.config.append, self.config.flush)
            except Exception as exc:
                message = f'[log] cannot open the sanity log {path!r} ({exc}).'
                if self.config.required:
                    raise RuntimeError(message) from exc
                tqdm.write(message + ' The sanity check goes to the progress log instead.')
                self._owned_sanity = False
                self._sanity = self._file
        elif self._sanity is None and self._file is not None:
            self._owned_sanity = False
            self._sanity = self._file
        return self._file is not None

    def close(self) -> None:
        '''
        Close only the files this logger opened; handles passed in from outside are
        left alone (the sanity and the progress log may share one handle).

        Returns:
            None: The owned handles are flushed and closed.
        '''
        file, self._file = self._file, None
        sanity, self._sanity = self._sanity, None
        if sanity is not None and self._owned_sanity and sanity is not file:
            self._close_one(sanity)
        if file is not None and self._owned:
            self._close_one(file)

    @staticmethod
    def _close_one(handle: Any) -> None:
        '''
        Flush and close one handle, reporting rather than raising on failure.

        Args:
            handle (Any): The handle to close.
        '''
        try:
            handle.flush()
            handle.close()
        except Exception as exc:
            tqdm.write(f'[!] failed to close the log file: {exc}')

    def _guard(self, action: Callable[[], None]) -> None:
        '''
        Run one write action, disabling the logger on any failure instead of
        propagating it.

        Args:
            action (Callable[[], None]): The write call to run.
        '''
        if not self._open():
            return
        try:
            action()
        except Exception as exc:
            self._disabled = True
            tqdm.write(f'[!] log write failed ({exc}), logging disabled, training continues.')

    def log(self, text: str = '') -> None:
        '''
        Write a block of text line by line (each line flushed immediately) with a
        timestamp prefix.

        Args:
            text (str): The text to write; empty lines are written as bare newlines.
        '''
        def action() -> None:
            for line in str(text).splitlines() or ['']:
                self._file.write(f'[{_stamp()}] {line}\n' if line else '\n')
            if self.config.flush:
                self._file.flush()
        self._guard(action)

    def write_raw(self, text: str, sanity: bool = False) -> None:
        '''
        Write text verbatim (no timestamp, no re-wrapping), for block-like tables such
        as the sanity check.

        Args:
            text (str): The text to write.
            sanity (bool): Write to the sanity log instead of the progress log.
        '''
        def action() -> None:
            handle = self._sanity if sanity else self._file
            handle.write(text if text.endswith('\n') else text + '\n')
            if self.config.flush:
                handle.flush()
        self._guard(action)

    def log_sanity(self, text: str = '') -> None:
        '''
        Write to the sanity log line by line (with timestamps).

        Args:
            text (str): The text to write.
        '''
        def action() -> None:
            handle = self._sanity if self._sanity is not None else self._file
            for line in str(text).splitlines() or ['']:
                handle.write(f'[{_stamp()}] {line}\n' if line else '\n')
            if self.config.flush:
                handle.flush()
        self._guard(action)

    def add_text(self, tag: str, text: str, step: int) -> None:
        '''
        A `(tag, text, step)` text sink, usable directly as `Probe(sink=...)`.

        Args:
            tag (str): Text tag, written as a header line.
            text (str): The text to write verbatim.
            step (int): Optimiser step the text belongs to.
        '''
        if not self.config.log_text:
            return
        self.log(f'--- {tag} @ step {step} ---')
        self.write_raw(text)

    def on_train_start(self, pipeline) -> None:
        '''
        Open the files and write the sanity header.

        Args:
            pipeline: The training pipeline.
        '''
        if not self._open():
            return
        self._guard(lambda: self._write_header(pipeline))

    def _write_header(self, pipeline) -> None:
        '''
        Write the sanity-check block and, when the logs are split, a pointer to it in
        the progress log.

        Args:
            pipeline: The training pipeline.
        '''
        lines: List[str] = [
            '=' * 70,
            f'sanity check | {pipeline.name} | {_stamp()}',
        ]
        if pipeline.state.global_step:
            lines.append(
                f'resumed at step {pipeline.state.global_step:,} '
                f'(epoch {pipeline.state.epoch})'
            )
        lines.append('-' * 70)
        lines += list(pipeline.describe()) if self.config.sanity_check else []
        if lines[-1] != '=' * 70:
            lines.append('=' * 70)
        self.write_raw('\n'.join(lines), sanity=True)
        if self.config.split_sanity and self.sanity_path != self.config.path:
            self.log(f'sanity check -> {os.path.abspath(self.sanity_path)}')

    def on_step_end(self, pipeline, step_metrics: Dict[str, float]) -> None:
        '''
        Write one progress line per configured step interval.

        Args:
            pipeline: The training pipeline.
            step_metrics (Dict[str, float]): Metrics of this step.
        '''
        step = pipeline.state.global_step
        if self.config.every_steps > 1 and step % self.config.every_steps:
            return
        self.log(f'step={step} {self._metrics(step_metrics)}')

    def on_evaluate(self, pipeline, eval_metrics: Dict[str, float]) -> None:
        '''
        Write the evaluation metrics when enabled.

        Args:
            pipeline: The training pipeline.
            eval_metrics (Dict[str, float]): Metrics of the evaluation.
        '''
        if not self.config.log_eval:
            return
        self.log(f'eval @ step {pipeline.state.global_step} {self._metrics(eval_metrics)}')

    def on_epoch_end(self, pipeline) -> None:
        '''
        Write one line when an epoch finishes.

        Args:
            pipeline: The training pipeline.
        '''
        self.log(f'epoch {pipeline.state.epoch} done @ step '
                 f'{pipeline.state.global_step}')

    def on_checkpoint(self, pipeline, path: str) -> None:
        '''
        Write one line per checkpoint when enabled.

        Args:
            pipeline: The training pipeline.
            path (str): Path of the checkpoint just written.
        '''
        if not self.config.log_checkpoints:
            return
        self.log(f'checkpoint @ step {pipeline.state.global_step} -> {path}')

    def on_interrupt(self, pipeline) -> None:
        '''
        Write a line when training is interrupted.

        Args:
            pipeline: The training pipeline.
        '''
        self.log(f'[!] interrupted @ step {pipeline.state.global_step}; saving an emergency checkpoint')

    def on_train_end(self, pipeline) -> None:
        '''
        Write the final metric snapshot and close the log files.

        Args:
            pipeline: The training pipeline.
        '''
        if self._file is None and not self._open():
            return

        def action() -> None:
            snapshot = pipeline.metrics.snapshot()
            lines = [
                '-' * 70,
                f'run end | {pipeline.state.phase.name} | step '
                f'{pipeline.state.global_step:,} | epoch {pipeline.state.epoch} | '
                f'elapsed {pipeline.state.elapsed:.1f}s',
            ]
            lines += [f'{key} = {_fmt(value)}' for key, value in sorted(snapshot.items())]
            lines.append('=' * 70)
            self.write_raw('\n'.join(lines))
        self._guard(action)
        self.close()

    def _metrics(self, metrics: Dict[str, float]) -> str:
        '''
        Lay the metrics of one step out as a `key=value` string.

        The keys listed in `progress_keys` come first, in that order, and everything
        else follows alphabetically.

        Args:
            metrics (Dict[str, float]): Metrics of one step.

        Returns:
            str: The formatted metric string; empty when there is nothing to write.
        '''
        if not metrics:
            return ''
        preferred = [k for k in self.config.progress_keys if k in metrics]
        rest = [k for k in sorted(metrics) if k not in preferred]
        return ' '.join(f'{key}={_fmt(metrics[key])}' for key in preferred + rest)


@dataclass
class TensorBoardConfig:
    '''
    TensorBoard reporting configuration.

    Assigning it to `PretrainConfig.tensorboard` / `SFTConfig.tensorboard` switches it
    on.

    Attributes:
        log_dir (str): Directory the event files are written to.
        every_steps (int): Write the scalars every this many optimiser steps.
        flush_secs (int): Automatic flush interval of the `SummaryWriter`, in seconds.
        log_eval (bool): Log the results of `evaluate()`.
        log_checkpoints (bool): Add a text entry per checkpoint.
        log_run_info (bool): Write the sanity check once at the start of training.
        log_text (bool): Log probe / external text (`add_text`).
        tag_map (Optional[Dict[str, str]]): Overrides applied on top of
            `DEFAULT_TAG_MAP`.
        comment (str): Suffix appended to the `log_dir` of the `SummaryWriter`.
        required (bool): True turns an unavailable tensorboard into an error.
    '''

    log_dir: str = 'runs/motif'
    every_steps: int = 1
    flush_secs: int = 30
    log_eval: bool = True
    log_checkpoints: bool = True
    log_run_info: bool = True
    log_text: bool = True
    tag_map: Optional[Dict[str, str]] = None
    comment: str = ''
    required: bool = False


class TensorBoardCallback(Callback):
    '''
    Write the `pipeline` metrics as TensorBoard events.

    Args:
        config (Optional[Any]): A `TensorBoardConfig`, or a plain `log_dir` string, or
            None for the defaults.
        writer: A ready writer, as long as it implements `add_scalar` / `add_text` /
            `flush` / `close`. Passing one bypasses the `torch.utils.tensorboard`
            import completely, which is handy for tests.

    Raises:
        TypeError: If `config` is neither a `TensorBoardConfig`, a string nor None.
    '''

    def __init__(self, config: Optional[Any] = None, writer: Any = None) -> None:
        '''
        Build the callback.

        Args:
            config (Optional[Any]): `TensorBoardConfig`, a `log_dir` string, or None.
            writer (Any): Ready writer to use instead of importing tensorboard.

        Raises:
            TypeError: If `config` has an unsupported type.
        '''
        if config is None:
            config = TensorBoardConfig()
        elif isinstance(config, str):
            config = TensorBoardConfig(log_dir=config)
        elif not isinstance(config, TensorBoardConfig):
            raise TypeError(
                f'TensorBoardCallback needs a TensorBoardConfig / a log_dir string / None, '
                f'got {type(config).__name__}.'
            )
        self.config = config
        self._writer = writer
        self._disabled = False
        self.tag_map = {**DEFAULT_TAG_MAP, **(config.tag_map or {})}

    @property
    def writer(self):
        '''
        Returns:
            The underlying `SummaryWriter`, once one has been opened.
        '''
        return self._writer

    def _open(self) -> bool:
        '''
        Import `torch.utils.tensorboard` lazily and open the writer.

        Returns:
            bool: False when this run does not write TensorBoard events.

        Raises:
            RuntimeError: If opening fails and `config.required` is set.
        '''
        if self._writer is not None:
            return True
        if self._disabled:
            return False
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:
            self._disabled = True
            message = (f'[tensorboard] cannot import torch.utils.tensorboard ({exc}); '
                       f'this run will not write TensorBoard events. Install it and '
                       f'retry: pip install tensorboard')
            if self.config.required:
                raise RuntimeError(message) from exc
            tqdm.write(message)
            return False
        try:
            self._writer = SummaryWriter(
                log_dir=self.config.log_dir,
                comment=self.config.comment,
                flush_secs=self.config.flush_secs,
            )
        except Exception as exc:
            self._disabled = True
            message = f'[tensorboard] failed to create the SummaryWriter ({exc}); this run will not write TensorBoard events.'
            if self.config.required:
                raise RuntimeError(message) from exc
            tqdm.write(message)
            return False
        return True

    def close(self) -> None:
        '''
        Flush and close the writer; a failure is only reported and never affects
        training.

        Returns:
            None: The writer is closed when one is open.
        '''
        writer, self._writer = self._writer, None
        if writer is None:
            return
        try:
            flush = getattr(writer, 'flush', None)
            if callable(flush):
                flush()
            writer.close()
        except Exception as exc:
            tqdm.write(f'[!] tensorboard close failed: {exc}')

    def _guard(self, action: Callable[[], None]) -> None:
        '''
        Run one write action, disabling the plug-in on any failure instead of
        propagating it.

        Args:
            action (Callable[[], None]): The write call to run.
        '''
        if not self._open():
            return
        try:
            action()
        except Exception as exc:
            self._disabled = True
            tqdm.write(f'[!] tensorboard write failed ({exc}), reporting disabled, training continues.')

    def add_scalar(self, key: str, value: Any, step: int) -> None:
        '''
        Write one scalar through the tag map; non-numeric values are ignored.

        Args:
            key (str): Pipeline metric name; an unknown key is used as the tag as it
                is.
            value (Any): Value to write.
            step (int): Optimiser step the value belongs to.
        '''
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return
        tag = self.tag_map.get(key, key)
        self._guard(lambda: self._writer.add_scalar(tag, float(value), int(step)))

    def add_text(self, tag: str, text: str, step: int) -> None:
        '''
        Write a piece of text.

        The signature matches `Probe(sink=...)`, so this method can be used as a sink
        directly.

        Args:
            tag (str): Text tag.
            text (str): The text to write.
            step (int): Optimiser step the text belongs to.
        '''
        if not self.config.log_text:
            return
        self._guard(lambda: self._writer.add_text(tag, str(text), int(step)))

    def on_train_start(self, pipeline) -> None:
        '''
        Open the writer and write the sanity check and the start entry.

        Args:
            pipeline: The training pipeline.
        '''
        if not self._open():
            return
        if self.config.log_run_info and hasattr(pipeline, 'describe'):
            self.add_text('Run/config', '\n'.join(pipeline.describe()), 0)
        self.add_text('Run/start', f'{pipeline.name} started on {pipeline.device}', 0)

    def on_step_end(self, pipeline, step_metrics: Dict[str, float]) -> None:
        '''
        Write the step metrics at the configured interval.

        Args:
            pipeline: The training pipeline.
            step_metrics (Dict[str, float]): Metrics of this step.
        '''
        if self._disabled or self._writer is None:
            return
        step = pipeline.state.global_step
        if self.config.every_steps > 1 and step % self.config.every_steps:
            return
        for key, value in step_metrics.items():
            self.add_scalar(key, value, step)

    def on_evaluate(self, pipeline, eval_metrics: Dict[str, float]) -> None:
        '''
        Write the evaluation metrics when enabled.

        Args:
            pipeline: The training pipeline.
            eval_metrics (Dict[str, float]): Metrics of the evaluation.
        '''
        if not self.config.log_eval:
            return
        step = pipeline.state.global_step
        for key, value in eval_metrics.items():
            self.add_scalar(key, value, step)

    def on_epoch_end(self, pipeline) -> None:
        '''
        Write the epoch index.

        Args:
            pipeline: The training pipeline.
        '''
        self.add_scalar('progress/epoch', pipeline.state.epoch, pipeline.state.global_step)

    def on_checkpoint(self, pipeline, path: str) -> None:
        '''
        Write the checkpoint path as text when enabled.

        Args:
            pipeline: The training pipeline.
            path (str): Path of the checkpoint just written.
        '''
        if not self.config.log_checkpoints:
            return
        self.add_text('Run/checkpoint', path, pipeline.state.global_step)

    def on_train_end(self, pipeline) -> None:
        '''
        Write the final metric summary and close the writer.

        Args:
            pipeline: The training pipeline.
        '''
        if self._writer is not None:
            step = pipeline.state.global_step
            snapshot = pipeline.metrics.snapshot()
            summary = '\n'.join(f'{key} = {_fmt(value)}' for key, value in sorted(snapshot.items()))
            self.add_text(
                'Run/finish',
                f'{pipeline.name} {pipeline.state.phase.name} @ step {step}\n{summary}',
                step,
            )
        self.close()


def build_monitor(spec: Any) -> Optional[Callback]:
    '''
    Normalise a monitoring field of a config into a callback:

        None / False                     -> None (not attached)
        str / os.PathLike                -> FileLogger(FileLogConfig(path))
        TensorBoardConfig                -> TensorBoardCallback(config)
        FileLogConfig                    -> FileLogger(config)
        TensorBoardCallback / FileLogger -> returned as it is (for callers that
                                            configure their own writer)

    Args:
        spec (Any): The `tensorboard` / `log_file` field value.

    Returns:
        Optional[Callback]: The callback, or None when monitoring is off.

    Raises:
        TypeError: If `spec` has an unsupported type.
    '''
    if spec is None or spec is False:
        return None
    if isinstance(spec, (TensorBoardCallback, FileLogger)):
        return spec
    if isinstance(spec, TensorBoardConfig):
        return TensorBoardCallback(spec)
    if isinstance(spec, FileLogConfig):
        return FileLogger(spec)
    if isinstance(spec, (str, os.PathLike)):
        return FileLogger(FileLogConfig(path=os.fspath(spec)))
    raise TypeError(
        f'a monitoring plug-in needs None / a path / TensorBoardConfig / FileLogConfig / '
        f'TensorBoardCallback / FileLogger, got {type(spec).__name__}.'
    )
