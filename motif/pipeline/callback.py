from typing import Any, Callable, Dict, Iterable, List, Optional

from tqdm import tqdm


#: Pipeline metric name -> dashboard tag, shared by every reporting plug-in so that one
#: dashboard can read all of them.
DEFAULT_TAG_MAP: Dict[str, str] = {
    'loss/train': 'Loss/train_avg',
    'loss/eval': 'Loss/eval',
    'lr': 'LR',
    'grad_norm': 'Grad_norm',
    'speed/steps_per_sec': 'Throughput',
    'speed/tokens_per_sec': 'Throughput/tokens',
    'progress/percent': 'Progress/percent',
    'seq_len': 'Seq_len',
    'tokens': 'Tokens',
}


class Callback:
    '''
    Base class of every callback: the hooks are empty by default, override as needed.

    A callback is an optional side observer - the pipeline calls the same-named hook at
    every key point, and a callback only reads the pipeline state, never taking part in
    the training loop. A callback that has to be persisted implements `state_dict()` /
    `load_state_dict()` and the pipeline writes it into the checkpoint along with
    everything else. `combine_sinks()` solves "one piece of text has to reach several
    plug-ins": `Probe` has a single `sink` entry point, while TensorBoard and the log
    file both want that text.
    '''

    #: Once set to True the pipeline winds down at the end of the current step.
    stop_requested: bool = False

    #: Run the hooks on rank 0 only. True by default, because a callback is a side
    #: effect - a log file, a dashboard, a progress bar, a probe, a snapshot - and N
    #: ranks writing the same path only produce garbage. The pipeline still agrees on
    #: stopping with every rank (`_sync_stop`), so an early-stopping callback works
    #: from rank 0 alone. Set it to False for a callback that takes part in training
    #: (it must then keep every rank in the same collective order).
    main_process_only: bool = True

    def on_train_start(self, pipeline) -> None:
        '''
        Called once before the training loop starts.

        Args:
            pipeline: The training pipeline.
        '''
        ...

    def on_epoch_start(self, pipeline) -> None:
        '''
        Called at the beginning of every epoch.

        Args:
            pipeline: The training pipeline.
        '''
        ...

    def on_step_end(self, pipeline, step_metrics: Dict[str, float]) -> None:
        '''
        Called after every optimiser step.

        Args:
            pipeline: The training pipeline.
            step_metrics (Dict[str, float]): Metrics of this step.
        '''
        ...

    def on_epoch_end(self, pipeline) -> None:
        '''
        Called at the end of every epoch.

        Args:
            pipeline: The training pipeline.
        '''
        ...

    def on_evaluate(self, pipeline, eval_metrics: Dict[str, float]) -> None:
        '''
        Called after an evaluation pass.

        Args:
            pipeline: The training pipeline.
            eval_metrics (Dict[str, float]): Metrics of the evaluation.
        '''
        ...

    def on_train_end(self, pipeline) -> None:
        '''
        Called once when the training loop finishes.

        Args:
            pipeline: The training pipeline.
        '''
        ...

    def on_checkpoint(self, pipeline, path: str) -> None:
        '''
        Called after a checkpoint has been written.

        Args:
            pipeline: The training pipeline.
            path (str): Path of the checkpoint file.
        '''
        ...

    def on_interrupt(self, pipeline) -> None:
        '''
        Called when training is interrupted (Ctrl+C).

        Args:
            pipeline: The training pipeline.
        '''
        ...

    def state_dict(self) -> Dict[str, Any]:
        '''
        Returns:
            Dict[str, Any]: The callback state; empty by default.
        '''
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        '''
        Args:
            state (Dict[str, Any]): State previously produced by `state_dict()`.

        Returns:
            None: The callback restores its own state in place.
        '''
        return None


def combine_sinks(
    sinks: Iterable[Optional[Callable[[str, str, int], None]]],
) -> Optional[Callable[[str, str, int], None]]:
    '''
    Combine several `(tag, text, step)` sinks into one (fan-out).

    * all of them None -> None (`Probe` then treats it as "no sink");
    * exactly one -> returned unchanged, without an extra wrapper;
    * a sink that raises only disables itself, the others keep receiving text, and
      the failure is reported once.

    Args:
        sinks (Iterable[Optional[Callable[[str, str, int], None]]]): Candidate sinks.

    Returns:
        Optional[Callable[[str, str, int], None]]: The combined sink, or None when
        every candidate was None.
    '''
    live = [sink for sink in sinks if sink is not None]
    if not live:
        return None
    if len(live) == 1:
        return live[0]

    disabled: List[Callable] = []

    def fan_out(tag: str, text: str, step: int) -> None:
        for sink in live:
            if sink in disabled:
                continue
            try:
                sink(tag, text, step)
            except Exception as exc:
                disabled.append(sink)
                tqdm.write(f'[!] text sink {type(sink).__name__} failed to write ({exc}), disabled.')

    return fan_out


class ProgressBar(Callback):
    '''tqdm progress bar hooked onto `on_step_end`.'''

    def __init__(self, keys: Optional[List[str]] = None) -> None:
        '''
        Args:
            keys (Optional[List[str]]): Metrics shown as the postfix; defaults to
                `['loss/train', 'lr']`.
        '''
        self.keys = keys
        self.pbar = None

    def on_train_start(self, pipeline) -> None:
        '''
        Open the progress bar.

        Args:
            pipeline: The training pipeline.
        '''
        self.pbar = tqdm(
            total=pipeline.total_steps,
            initial=pipeline.state.global_step,
            desc=pipeline.name,
            dynamic_ncols=True,
        )

    def on_step_end(self, pipeline, step_metrics: Dict[str, float]) -> None:
        '''
        Refresh the postfix with the running averages and advance the bar.

        Args:
            pipeline: The training pipeline.
            step_metrics (Dict[str, float]): Metrics of this step (unused; the
                running averages are read from the pipeline instead).
        '''
        if self.pbar is None:
            return
        keys = self.keys or ['loss/train', 'lr']
        postfix = {}
        for key in keys:
            if key not in pipeline.metrics.keys():
                continue
            value = pipeline.metrics.running_avg(key)
            postfix[key] = f'{value:.4g}' if 'lr' not in key else f'{value:.2e}'
        if postfix:
            self.pbar.set_postfix(postfix)
        self.pbar.update(1)

    def on_train_end(self, pipeline) -> None:
        '''
        Close the progress bar.

        Args:
            pipeline: The training pipeline.
        '''
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
