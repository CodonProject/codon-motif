import contextlib
import dataclasses
import json
import os
import random
import tempfile
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from motif.model.checkpoint import load_checkpoint as read_any_checkpoint
from motif.pipeline.callback import Callback, ProgressBar, combine_sinks
from motif.pipeline.distributed import (
    DistributedContext,
    setup_distributed,
    unwrap_model,
    wrap_model,
)
from motif.utils.sharding import shard_batch


class PipelinePhase(Enum):
    '''
    Phase of the training pipeline.

    The value is stored inside `TrainState` and therefore ends up in the checkpoint;
    restoring a checkpoint resets it to `INIT`.

    Attributes:
        INIT (PipelinePhase): Set up, not training yet.
        SETUP (PipelinePhase): Running `setup()`.
        TRAINING (PipelinePhase): Inside the training loop.
        EVALUATING (PipelinePhase): Running `evaluate()`.
        FINISHED (PipelinePhase): The training loop ended normally.
        INTERRUPTED (PipelinePhase): Stopped by Ctrl+C.
        FAILED (PipelinePhase): Aborted by an exception.
    '''

    INIT = auto()
    SETUP = auto()
    TRAINING = auto()
    EVALUATING = auto()
    FINISHED = auto()
    INTERRUPTED = auto()
    FAILED = auto()


@dataclass
class TrainState:
    '''
    Training progress; written into the checkpoint in full.

    Attributes:
        global_step (int): Number of optimiser steps taken so far.
        epoch (int): Index of the current epoch.
        best_metric (float): Best evaluation metric seen so far.
        best_metric_name (Optional[str]): Name of the metric being tracked.
        phase (PipelinePhase): Current phase of the pipeline.
        interrupted (bool): True when the run was stopped by Ctrl+C.
        start_time (float): Wall-clock time the run started.
        elapsed (float): Accumulated training time in seconds.
    '''
    global_step: int = 0
    epoch: int = 0
    best_metric: float = float('-inf')
    best_metric_name: Optional[str] = None
    phase: PipelinePhase = PipelinePhase.INIT
    interrupted: bool = False
    start_time: float = 0.0
    elapsed: float = 0.0

    def advance_step(self) -> int:
        '''
        Increment the global step counter.

        Returns:
            int: The new global step.
        '''
        self.global_step += 1
        return self.global_step


class MetricsTracker:
    '''
    Metric statistics with history; keys are conventionally `namespace/key`
    (for example `loss/train`).

    Attributes:
        _history (Dict[str, List[float]]): Recorded values per key.
        _maxlen (int): Maximum number of values kept per key.
    '''

    def __init__(self, maxlen: int = 10000):
        '''
        Args:
            maxlen (int): Maximum number of values kept per key (older ones are
                dropped).
        '''
        self._history: Dict[str, List[float]] = {}
        self._maxlen = maxlen

    def log(self, metrics: Dict[str, float]) -> None:
        '''
        Record one step of metrics; non-numeric values are skipped.

        Args:
            metrics (Dict[str, float]): Metric name to value.
        '''
        for key, value in metrics.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            self._history.setdefault(key, []).append(float(value))
            if len(self._history[key]) > self._maxlen:
                self._history[key] = self._history[key][-self._maxlen:]

    def last(self, key: str, default: float = 0.0) -> float:
        '''
        Args:
            key (str): Metric name.
            default (float): Value returned when the metric was never logged.

        Returns:
            float: The most recent value of that metric.
        '''
        history = self._history.get(key)
        return history[-1] if history else default

    def get(self, key: str) -> List[float]:
        '''
        Args:
            key (str): Metric name.

        Returns:
            List[float]: A copy of the whole history of that metric.
        '''
        return list(self._history.get(key, []))

    def running_avg(self, key: str, window: int = 100) -> float:
        '''
        Args:
            key (str): Metric name.
            window (int): Number of most recent values averaged.

        Returns:
            float: The running average, or 0.0 when the metric was never logged.
        '''
        history = self._history.get(key, [])
        tail = history[-window:]
        return sum(tail) / len(tail) if tail else 0.0

    def keys(self) -> List[str]:
        '''
        Returns:
            List[str]: Every recorded metric name, sorted.
        '''
        return sorted(self._history.keys())

    def snapshot(self, window: int = 100) -> Dict[str, float]:
        '''
        Args:
            window (int): Number of most recent values averaged per metric.

        Returns:
            Dict[str, float]: The running average of every recorded metric.
        '''
        return {key: self.running_avg(key, window) for key in self.keys()}

    def dump_jsonl(self, path: str) -> None:
        '''
        Write the per-step metrics to a jsonl file (for reviewing the curves later).

        Args:
            path (str): Destination file; its directory is created when needed.
        '''
        length = max((len(v) for v in self._history.values()), default=0)
        keys = self.keys()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for i in range(length):
                row = {k: self._history[k][i] for k in keys if i < len(self._history[k])}
                f.write(json.dumps(row) + '\n')


def resolve_device(model: nn.Module, device: Union[str, torch.device, None] = 'auto') -> torch.device:
    '''
    Decide which device to train on. Priority:

      1. an explicitly passed `device` (`'cpu'` / `'cuda'` / `'cuda:1'` /
         `torch.device`);
      2. a model that is **already** on a non-CPU device - follow it;
      3. `'auto'` (the default): use CUDA when it is available, otherwise CPU.

    Point 3 matters: a model sitting on the CPU does not mean "I want to train on the
    CPU". If this fell back to the CPU by default, a script that forgot
    `model.cuda()` would silently train on the CPU for hours before anyone noticed
    (and `setup()` would faithfully move the model to that wrong device). Ask for the
    CPU explicitly with `device='cpu'`.

    A codon-style model carries its own `.device` attribute, which is point 2.

    Args:
        model (nn.Module): The model about to be trained.
        device (Union[str, torch.device, None]): Requested device, or 'auto' / None.

    Returns:
        torch.device: The device to train on.
    '''
    if device is not None and str(device) != 'auto':
        return torch.device(device)

    declared = getattr(model, 'device', None)
    if isinstance(declared, torch.device) and declared.type != 'cpu':
        return declared
    if isinstance(declared, str) and declared != 'cpu':
        return torch.device(declared)

    current = None
    parameters = getattr(model, 'parameters', None)
    if callable(parameters):
        try:
            current = next(parameters()).device
        except StopIteration:
            current = None
    if current is None:
        buffers = getattr(model, 'buffers', None)
        if callable(buffers):
            try:
                current = next(buffers()).device
            except StopIteration:
                current = None
    if current is None:
        current = torch.device('cpu')

    if current.type != 'cpu':
        return current
    if torch.cuda.is_available():
        return torch.device('cuda')
    return current


def move_to_device(value: Any, device: torch.device) -> Any:
    '''
    Move (possibly nested) tensors to `device`, recursively.

    A field may be a flat list or one sub-list per batch row; flattening would lose
    the batch ownership, so the containers are rebuilt level by level instead.

    Args:
        value (Any): Tensor, or a nested list / tuple of tensors.
        device (torch.device): Target device.

    Returns:
        Any: The same structure with every tensor moved.
    '''
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


def resolve_token_spec(tokenizer, spec: Any, what: str = 'token') -> int:
    '''
    Normalise "a token id" or "token text" into an id.

    Both an id (`eos_token_id=9`) and a logical name or the literal vocabulary text
    (`eos_token_id='im_end'` / `'<|im_end|>'`) are accepted: the latter is far more
    readable and does not silently encode the wrong id when the vocabulary changes.

    Args:
        tokenizer: Tokenizer used to resolve token text; may be None when an id is
            used.
        spec (Any): Token id, or token text (logical name or literal text).
        what (str): Field name used in the error messages.

    Returns:
        int: The resolved token id.

    Raises:
        TypeError: If `spec` is neither an int nor a string.
        ValueError: If token text is given without a tokenizer, or if the text is not
            in the vocabulary.
    '''
    if isinstance(spec, bool) or not isinstance(spec, (int, str)):
        raise TypeError(
            f'{what} needs a token id (int) or token text (str), got {type(spec).__name__}.'
        )
    if isinstance(spec, int):
        return int(spec)
    if tokenizer is None:
        raise ValueError(
            f'{what}={spec!r} is token text, but there is no tokenizer here to parse '
            f'it; please pass the token id directly.'
        )
    from motif.utils.session import resolve_token_id

    token_id = resolve_token_id(tokenizer, str(spec))
    if token_id is None:
        raise ValueError(
            f'the vocabulary has no {spec!r} ({what}): pass a logical name (such as '
            f'"im_end" / "end_of_text"), the literal vocabulary text (such as '
            f'"<|im_end|>"), or a token id directly.'
        )
    return int(token_id)


def resolve_eos_token_id(tokenizer, explicit: Optional[Any] = None) -> Optional[int]:
    '''
    Find the EOS id used as the record separator / early-stop token.

    Tried in order: the explicit argument (id or token text) ->
    `fast_tokenizer.eos_token_id` -> `<|im_end|>` in the vocabulary. When none of
    them exists it returns None, and `ChunkedTokenStream` then needs the caller to
    supply an eos id itself.

    Args:
        tokenizer: Tokenizer to inspect.
        explicit (Optional[Any]): Explicit token id or token text.

    Returns:
        Optional[int]: The resolved EOS id, or None.
    '''
    if explicit is not None:
        return resolve_token_spec(tokenizer, explicit, 'eos_token_id')
    fast = getattr(tokenizer, 'fast_tokenizer', None)
    eos = getattr(fast, 'eos_token_id', None) if fast is not None else None
    if isinstance(eos, int):
        return eos
    try:
        from motif.utils.session import resolve_token_id
        return resolve_token_id(tokenizer, 'im_end')
    except Exception:
        return None


def resolve_boundary_token_id(tokenizer, explicit: Optional[Any] = None) -> Optional[int]:
    '''
    Find the **document boundary token** inserted after every record while packing
    the pre-training corpus.

    Tried in order: the explicit argument (id or token text) -> `<|end_of_text|>` in
    the vocabulary. When neither works it returns None and the caller falls back to
    eos (the older behaviour).

    `resolve_eos_token_id` is deliberately not reused here: on the generation side eos
    is usually `<|im_end|>` (the end of a conversation), while the document boundary
    wants `<|end_of_text|>` (the end of a document). Mixing them would fill the
    pre-training corpus with `im_end` and make it impossible for the SFT stage to tell
    "the conversation ended" from "the document ended".

    Args:
        tokenizer: Tokenizer to inspect.
        explicit (Optional[Any]): Explicit token id or token text.

    Returns:
        Optional[int]: The resolved boundary token id, or None.
    '''
    if explicit is not None:
        return resolve_token_spec(tokenizer, explicit, 'boundary_token_id')
    if tokenizer is None:
        return None
    try:
        from motif.utils.session import resolve_token_id
        return resolve_token_id(tokenizer, 'end_of_text')
    except Exception:
        return None


def forward_accepted_kwargs(model: nn.Module) -> Optional[set]:
    '''
    Names of the parameters `forward` accepts.

    None is returned when `forward` takes `**kwargs` or its signature cannot be
    inspected, which means "it accepts everything" - that way a custom model is never
    hit by a `TypeError` just because a batch carried a field it does not declare.

    The distributed and the compiled wrapper are unwrapped first: DDP's own `forward`
    is `(*inputs, **kwargs)`, so inspecting the wrapper would answer "accepts
    everything" and defeat the filtering that keeps a model from receiving fields it
    does not declare.

    Args:
        model (nn.Module): The model to inspect.

    Returns:
        Optional[set]: The accepted parameter names, or None when unknown.
    '''
    import inspect

    target = unwrap_model(model)
    forward = getattr(target, 'forward', None)
    if forward is None:
        return None
    try:
        params = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    return set(params)


def select_model_kwargs(model: nn.Module, batch: Dict[str, Any]) -> Dict[str, Any]:
    '''
    Pick the fields of the batch dict that `forward` really accepts (renaming
    `attention_mask` to `mask`).

    A field a model does not declare never reaches it, so it never gets a `TypeError`
    from one.

    Args:
        model (nn.Module): The model whose `forward` signature decides.
        batch (Dict[str, Any]): One batch produced by the dataset.

    Returns:
        Dict[str, Any]: The keyword arguments to pass on.
    '''
    accepted = forward_accepted_kwargs(model)
    picked: Dict[str, Any] = {}
    for key, value in batch.items():
        if key == 'labels':
            continue
        name = 'mask' if key == 'attention_mask' else key
        if accepted is not None and name not in accepted and key not in accepted:
            continue
        picked[name] = value
    return picked


class BasicPipeline:
    '''
    Base class of every training pipeline.

    A subclass **must** implement:

        compute_loss(batch) -> (torch.Tensor, Dict[str, float])
        build_optimizer()   -> torch.optim.Optimizer
        state_payload()     -> Dict[str, Any]

    and may override:

        setup() / teardown()
        iterate_epochs(dataset) -> Iterator[any "one epoch" iterable]
        evaluate(dataset) -> Dict[str, float]
        _load_payload(payload) -> None
        _scale_loss(loss, metrics) -> torch.Tensor

    A subclass that moves the model in `setup()` calls `_start_ddp()` right afterwards;
    everything else about data parallelism (device per rank, rank-0-only checkpoints
    and callbacks, metric reduction, stop synchronisation, token-weighted loss) is
    handled here, see `motif/pipeline/distributed.py`.

    A subclass answers four questions:

        compute_loss(batch)          -> (loss, metrics)     how to compute this step's loss
        build_optimizer()            -> optimizer           how to group the parameters
        iterate_epochs(dataset)      -> Iterator[epoch_data] what each epoch feeds
        state_payload()              -> dict                 which tensors go into the checkpoint

    Everything else is handled by the base class: AMP, gradient accumulation, gradient
    clipping, optimizer/scheduler stepping, metric tracking, callbacks, periodic
    checkpoints, Ctrl+C winding down, and **resuming the data-flow state**.

    Stateful resume contract: the pipeline keeps a table of `{name: dataset}`
    (`register_stateful`). When saving a checkpoint it stores
    `payload['data'] = {name: dataset.state_dict()}`. Restoring pushes the state back
    in reverse: `load_checkpoint()` first caches the payload -> `setup()` builds the
    datasets and the optimizer -> then `_restore_data_state()` lands it on the data
    flow. When the data-flow state is missing it **raises immediately** instead of
    feeding from the start again: a silent replay turns "resume" into "train another
    stretch over data that was already trained"; the loss curve looks normal, but the
    actual semantics are already wrong.
    '''

    #: Checkpoint format version; bumped when the payload layout changes.
    __pipeline_version__: str = '1.1'

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device, None] = None,
        callbacks: Optional[List[Callback]] = None,
        seed: Optional[int] = None,
        ckpt_dir: str = './ckpt',
        save_every_steps: int = 0,
        keep_last: int = 3,
        amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
        auto_resume: bool = True,
        use_progress: bool = True,
        distributed: Optional[Any] = None,
    ):
        '''
        Build the pipeline around a model.

        Args:
            model (nn.Module): Model to train.
            device (Union[str, torch.device, None]): Training device; None goes
                through `resolve_device`. Under a distributed launch this is
                overridden by the device of this rank (`cuda:<local_rank>`).
            callbacks (Optional[List[Callback]]): Extra callbacks.
            seed (Optional[int]): Seed for Python and torch; None skips seeding.
            ckpt_dir (str): Directory the checkpoints are written to.
            save_every_steps (int): Write a periodic checkpoint every this many steps
                (0 disables the periodic save).
            keep_last (int): Number of periodic checkpoints kept on disk.
            amp (bool): Default for autocast, effective on CUDA only.
            amp_dtype (torch.dtype): Dtype used by autocast (bf16 by default).
            auto_resume (bool): Load `<ckpt_dir>/last.pt` automatically in `train()`.
            use_progress (bool): Attach the default progress bar.
            distributed (Optional[Any]): `DistributedConfig`, or None for its defaults.
                A launch environment (`RANK` / `WORLD_SIZE`) is always joined - that is
                a correctness requirement, not a preference, because N uncoordinated
                ranks would write the same checkpoint file.
        '''
        self._model = model
        self._model_compiled: Optional[nn.Module] = None
        self._ddp: Optional[nn.Module] = None

        self.device = resolve_device(model, device)
        self._dist: DistributedContext = setup_distributed(distributed, device=self.device)
        if self._dist.active and self._dist.device is not None:
            self.device = self._dist.device

        self.state = TrainState()
        self.metrics = MetricsTracker()

        self.ckpt_dir = ckpt_dir
        self.save_every_steps = save_every_steps
        self.keep_last = keep_last
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.auto_resume = auto_resume

        self.seed = seed
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.callbacks: List[Callback] = list(callbacks or [])
        if use_progress and self.is_main:
            self.callbacks.append(ProgressBar())

        self._stateful: Dict[str, Any] = {}
        self._pending_payload: Optional[Dict[str, Any]] = None
        self._pending_data: Optional[Dict[str, Any]] = None
        self._saved: List[str] = []

        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler: Optional[Any] = None
        self._total_steps: int = 0
        #: Batches accumulated per optimizer step; `train()` fills this in before
        #: `setup()`, so a subclass can turn its batch count into optimizer steps.
        self._grad_accum_steps: int = 1
        self._setup_done = False
        self._last_step_time = time.time()
        self._spawn_hint_shown = False

    @property
    def name(self) -> str:
        '''
        Returns:
            str: The class name, used as the pipeline identity in checkpoints.
        '''
        return type(self).__name__

    @property
    def model(self) -> nn.Module:
        '''
        Returns:
            nn.Module: The module the forward pass runs through: the DDP wrapper when
            the run is distributed, otherwise the compiled model when one exists,
            otherwise the raw model.

        Note that `state_payload()` deliberately reads the **raw** model
        (`self._model`), so a checkpoint never carries DDP's `module.` prefix and one
        written on 4 GPUs loads on 1.
        '''
        if self._ddp is not None:
            return self._ddp
        return self._model_compiled if self._model_compiled is not None else self._model

    @property
    def raw_model(self) -> nn.Module:
        '''
        Returns:
            nn.Module: The unwrapped model, without DDP / `torch.compile` wrappers.
            Used by the plug-ins (probe, snapshots) so that rank 0 does not run a
            collective or a prefixed `state_dict()` by accident.
        '''
        return unwrap_model(self.model)

    @property
    def distributed(self) -> bool:
        '''
        Returns:
            bool: True when this process is one rank of a data-parallel group.
        '''
        return self._dist.active

    @property
    def rank(self) -> int:
        '''
        Returns:
            int: Global rank of this process; 0 for a single-process run.
        '''
        return self._dist.rank

    @property
    def local_rank(self) -> int:
        '''
        Returns:
            int: Index of the device this process owns; 0 for a single-process run.
        '''
        return self._dist.local_rank

    @property
    def world_size(self) -> int:
        '''
        Returns:
            int: Number of ranks; 1 for a single-process run.
        '''
        return self._dist.world_size

    @property
    def is_main(self) -> bool:
        '''
        Returns:
            bool: True on rank 0, the only rank that writes checkpoints and reports.
        '''
        return self._dist.is_main

    def _start_ddp(self) -> None:
        '''
        Wrap the model in DDP; a no-op on a single process.

        Call it from `setup()` right after the model has been moved to
        `self.device`: DDP records where the parameters live when it is built, so
        wrapping earlier would replicate the model on the wrong device.
        '''
        if not self._dist.active or self._ddp is not None:
            return
        self._ddp = wrap_model(self._model_compiled or self._model, self._dist)

    def shard_config(self) -> Tuple[int, int]:
        '''
        The row-slicing layout a distributed pipeline should use for its data.

        Returns:
            Tuple[int, int]: `(shard_rank, shard_world)`; `(0, 1)` means "no sharding",
            which is also what `DistributedConfig.shard_data = False` asks for (every
            rank then computes the same batch).
        '''
        if not self._dist.active or not self._dist.shard_data:
            return 0, 1
        return self.rank, self.world_size

    def _shard_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        '''
        Restrict a materialised batch to the rows of this rank.

        For pipelines whose data flow hands the whole batch to every rank: the
        cursors stay identical (so the checkpoint remains world-size agnostic) and the
        step's work is still divided over the ranks instead of being duplicated.

        Args:
            batch (Dict[str, Any]): One batch produced by the dataset.

        Returns:
            Dict[str, Any]: The batch of this rank, or `batch` unchanged when the run
            is not sharded.
        '''
        shard_rank, shard_world = self.shard_config()
        if shard_world <= 1:
            return batch
        return shard_batch(batch, shard_rank, shard_world)

    @property
    def total_steps(self) -> int:
        '''
        Returns:
            int: Total number of optimiser steps of the run; 0 until `setup()` ran.
        '''
        return self._total_steps

    @property
    def current_lr(self) -> float:
        '''
        Returns:
            float: The learning rate of the first parameter group, or 0.0 before the
            scheduler exists.
        '''
        return self._scheduler.get_last_lr()[0] if self._scheduler is not None else 0.0

    def optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        '''
        Returns:
            Dict[str, torch.optim.Optimizer]: The named optimizers of this pipeline.
        '''
        return {'main': self._optimizer} if self._optimizer is not None else {}

    def compute_loss(self, batch: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        '''
        Compute the loss of this step.

        Args:
            batch (Any): One batch as produced by `iterate_epochs`.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: The loss and any extra metrics.

        Raises:
            NotImplementedError: Always; subclasses must implement this.
        '''
        raise NotImplementedError

    def build_optimizer(self) -> torch.optim.Optimizer:
        '''
        Build the optimizer and attach the scheduler to `self._scheduler`.

        Returns:
            torch.optim.Optimizer: The optimizer that was created.

        Raises:
            NotImplementedError: Always; subclasses must implement this.
        '''
        raise NotImplementedError

    def state_payload(self) -> Dict[str, Any]:
        '''
        Tensors that go into the checkpoint, usually the state_dicts of the model,
        optimizer and scheduler.

        Returns:
            Dict[str, Any]: The payload; empty by default.
        '''
        return {}

    def _load_payload(self, payload: Dict[str, Any]) -> None:
        '''
        Push the content of `state_payload()` back into the objects.

        Args:
            payload (Dict[str, Any]): Payload produced by `state_payload()`.

        Returns:
            None: The subclass restores its own state in place.
        '''
        return None

    def setup(self) -> None:
        '''
        Lazy initialisation: move to the device, build the optimizer and register the
        data flows.
        '''

    def teardown(self) -> None:
        '''
        Release the cached CUDA memory, when CUDA is in use.

        Returns:
            None: The CUDA cache is emptied in place.
        '''
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def iterate_epochs(self, dataset: Any) -> Iterator[Any]:
        '''
        Default behaviour: hand the dataset back once per epoch (the caller iterates
        it).

        Args:
            dataset (Any): The dataset to hand out.

        Yields:
            Any: The dataset itself.
        '''
        while True:
            yield dataset

    def evaluate(self, dataset: Any = None) -> Dict[str, float]:
        '''
        Evaluate the model.

        Args:
            dataset (Any): Evaluation dataset; None returns no metrics.

        Returns:
            Dict[str, float]: The evaluation metrics; empty by default.
        '''
        return {}

    def register_stateful(self, name: str, obj: Any) -> None:
        '''
        Register a resumable object; its `state_dict()` is stored with the checkpoint.

        Args:
            name (str): Key the object is stored under.
            obj (Any): Object implementing `state_dict()` / `load_state_dict()`.

        Raises:
            TypeError: If `obj` does not implement both methods and therefore could
                not be resumed.
        '''
        if not (hasattr(obj, 'state_dict') and hasattr(obj, 'load_state_dict')):
            raise TypeError(
                f'{name}: both state_dict() and load_state_dict() are required to '
                f'resume, got {type(obj).__name__}.'
            )
        self._stateful[name] = obj

    def data_state(self) -> Dict[str, Any]:
        '''
        Returns:
            Dict[str, Any]: The `state_dict()` of every registered resumable object.
        '''
        return {name: obj.state_dict() for name, obj in self._stateful.items()}

    def _restore_data_state(self) -> None:
        '''
        Push the data-flow state of the checkpoint onto the registered objects.

        Raises:
            RuntimeError: If the checkpoint names a data flow that is not registered,
                which would silently misalign the samples.
        '''
        if not self._pending_data:
            return
        pending, self._pending_data = self._pending_data, None

        missing = [name for name in pending if name not in self._stateful]
        if missing:
            raise RuntimeError(
                f'the checkpoint holds data-flow state for {missing}, but no dataset '
                f'with those names is registered. Resuming has to use the same data '
                f'configuration as the checkpoint, otherwise the samples are silently '
                f'misaligned.'
            )
        for name, state in pending.items():
            self._stateful[name].load_state_dict(state)

    def _callback(self, hook: str, *args) -> None:
        '''
        Call one hook on every registered callback.

        Callbacks marked `main_process_only` (the default) are skipped on every rank
        but 0, so the log file, TensorBoard, the progress bar, the probe and
        the weight snapshots are written once instead of once per rank. The one
        decision that is taken from a callback - early stopping - is broadcast by
        `_sync_stop()`, which is why skipping the others is safe.

        Args:
            hook (str): Name of the hook.
            *args: Extra positional arguments passed after the pipeline itself.
        '''
        for cb in self.callbacks:
            if not self.is_main and getattr(cb, 'main_process_only', True):
                continue
            getattr(cb, hook)(self, *args)

    def _should_stop(self) -> bool:
        '''
        Returns:
            bool: True when a callback of this rank requested a stop.
        '''
        return any(getattr(cb, 'stop_requested', False) for cb in self.callbacks)

    def _sync_stop(self) -> bool:
        '''
        Agree on "wind down now" with every rank.

        The hooks only run on rank 0, so an early-stopping callback sets
        `stop_requested` there and nowhere else. Stopping has to be agreed on:
        a rank that left the loop while its peers are waiting in a gradient all-reduce
        would hang the whole job, so rank 0's decision is broadcast.

        Returns:
            bool: The agreed decision.
        '''
        return self._dist.broadcast_flag(self._should_stop())

    def _scale_loss(self, loss: torch.Tensor, batch_metrics: Dict[str, float]) -> torch.Tensor:
        '''
        Weight one rank's loss so that the DDP gradient average is the global one.

        DDP sums the per-rank gradients and divides by the world size, while
        `F.cross_entropy(..., reduction='mean')` averaged over **this rank's** unmasked
        positions - so a rank that holds fewer trainable tokens would count less.
        Multiplying by `world_size * local_tokens / global_tokens` cancels both, and
        the same factor makes the plain average of the per-rank losses equal to the
        global mean loss, which is what `_reduce_step_metrics()` then reports.

        The factor is exactly 1 whenever every rank holds the same number of tokens
        (always true with one rank), so the single-GPU path is untouched.

        Args:
            loss (torch.Tensor): The loss of one micro-batch, as returned by
                `compute_loss()`.
            batch_metrics (Dict[str, float]): Its metrics; `tokens` is the number of
                positions that contributed to the loss.

        Returns:
            torch.Tensor: The weighted loss.
        '''
        if not self._dist.active or self.world_size <= 1:
            return loss
        local_tokens = float(batch_metrics.get('tokens', 0.0) or 0.0)
        if local_tokens <= 0.0:
            return loss

        sizes = torch.tensor([local_tokens], device=self.device)
        sizes = self._dist.reduce_sum_(sizes)
        global_tokens = float(sizes.item())
        if global_tokens <= 0.0:
            return loss
        return loss * (local_tokens * self.world_size / global_tokens)

    def _reduce_step_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        '''
        Combine the step metrics of every rank into the global ones.

        `loss/train` is averaged (the per-rank losses were token-weighted by
        `_scale_loss()`, so their plain mean *is* the global mean loss) while the token
        counters are summed. Throughput, learning rate and progress are properties of
        this rank or identical everywhere, so they are left alone.

        A **fixed** key list is packed into one tensor: every rank has to contribute the
        same shape in the same order, and a rank that happens to have no trainable
        token in a batch would otherwise send a differently shaped message.

        Args:
            metrics (Dict[str, float]): Metrics of this rank.

        Returns:
            Dict[str, float]: The same dict with the shared keys replaced.
        '''
        if not self._dist.active or self.world_size <= 1:
            return metrics

        summed = ('tokens', 'tokens/masked')
        averaged = ('loss/train',)
        keys = summed + averaged
        pack = torch.tensor([float(metrics.get(key, 0.0)) for key in keys],
                            device=self.device)
        pack = self._dist.reduce_sum_(pack)
        for index, key in enumerate(averaged):
            pack[len(summed) + index] = pack[len(summed) + index] / self.world_size
        for index, key in enumerate(keys):
            if key in metrics:
                metrics[key] = float(pack[index])
        return metrics

    def attach_monitor(self, spec: Any):
        '''
        Attach a monitoring plug-in to `callbacks` and return the callback itself.

        `spec` may be a path string / `FileLogConfig` / `TensorBoardConfig`, or a
        ready `FileLogger` / `TensorBoardCallback`; None attaches nothing. TensorBoard
        is an optional dependency: when it cannot be imported the plug-in degrades on
        its own and nothing is raised here.

        Args:
            spec (Any): The `tensorboard` / `log_file` field value.

        Returns:
            Optional[Callback]: The attached callback, or None.
        '''
        from motif.pipeline.monitor import build_monitor

        callback = build_monitor(spec)
        if callback is None:
            return None
        self.callbacks.append(callback)
        return callback

    def attach_snapshot(self, config: Any = None):
        '''
        Attach a weight-snapshot callback from the three `PretrainConfig.snapshot_*`
        fields and return it.

        None is returned when the switch is off (a percentage that is None / <= 0).
        Snapshots are a research side product: failing to save one only disables the
        plug-in and never affects training (`SnapshotConfig(required=True)` is what
        raises).

        Args:
            config (Any): Configuration object carrying the `snapshot_*` fields.

        Returns:
            Optional[SnapshotCallback]: The attached callback, or None.
        '''
        from motif.pipeline.snapshot import build_snapshot

        callback = build_snapshot(
            getattr(config, 'snapshot_every_percent', None),
            getattr(config, 'snapshot_dir', None),
            getattr(config, 'snapshot_at_end', True),
        )
        if callback is None:
            return None
        self.callbacks.append(callback)
        return callback

    def attach_text_sinks(self) -> Optional[Callable[[str, str, int], None]]:
        '''
        Combine every callback that has `add_text(tag, text, step)` into one sink.

        That way a piece of text generated by `Probe` reaches TensorBoard and the log
        file at once, without `Probe` having to know those plug-ins
        exist.

        Returns:
            Optional[Callable[[str, str, int], None]]: The combined sink, or None when
            no callback accepts text.
        '''
        return combine_sinks(
            getattr(cb, 'add_text', None) for cb in self.callbacks
        )

    def describe(self) -> List[str]:
        '''
        Sanity check: the run information of this pipeline (for the log, TensorBoard
        and environment debugging).

        Subclasses may append their own content (pre-training, for instance, appends
        the context curriculum table).

        Returns:
            List[str]: The description lines.
        '''
        from motif.utils.units import count_params, format_count

        total_params = count_params(self.model)
        return [
            f'pipeline    : {self.name}',
            f'device      : {self.device}',
            f'distributed : {self._dist.describe()}',
            f'params      : {format_count(total_params)} ({total_params:,})',
            f'total_steps : {self.total_steps:,}',
            f'amp         : {self.amp} ({self.amp_dtype})',
            f'ckpt_dir    : {os.path.abspath(self.ckpt_dir)}',
            f'save_every  : {self.save_every_steps}',
            f'config      : {self.config!r}',
        ]

    def _hint_about_idle_gpus(self) -> None:
        '''
        Print once, on rank 0, that several GPUs are visible but only one is used.

        Adding cards is the difference between a two-day run and a two-hour run, and
        the failure mode is silent (a correct but single-process run), so it is worth
        one line of console output.
        '''
        if self._spawn_hint_shown or not self.is_main or self._dist.active:
            return
        self._spawn_hint_shown = True
        if not self._dist.config.auto_spawn or not torch.cuda.is_available():
            return
        available = torch.cuda.device_count()
        if available > 1:
            print(f'[i] {available} GPUs are visible but this run uses one: wrap the '
                  f"entry point in motif.pipeline.launch(build) (or torchrun) to use "
                  f'all of them.')

    def save_checkpoint(self, path: str) -> None:
        '''
        Write a checkpoint atomically: a temporary file first, then a rename, so a
        crash midway never leaves half a checkpoint behind.

        Only rank 0 writes: the payload is identical on every rank (DDP keeps the
        replicas in step and the data-flow cursors start from the same checkpoint), so
        N ranks writing the same path would only race.

        Args:
            path (str): Destination file.

        Raises:
            RuntimeError: If a data-flow state has not been restored yet, because
                writing then would store an unrestored cursor and break the resume
                semantics.
        '''
        if not self.is_main:
            return
        if self._pending_data:
            raise RuntimeError(
                f'a checkpoint is being saved while the data-flow state '
                f'{sorted(self._pending_data)} is still unrestored - the data objects '
                f'were built but `_restore_data_state()` was never called. Saving here '
                f'would write the unrestored cursor into a new checkpoint and break '
                f'the resume semantics.'
            )
        payload = {
            '__pipeline__': self.name,
            '__version__': self.__pipeline_version__,
            'state': dataclasses.asdict(self.state),
            'metrics': self.metrics.snapshot(),
            'callbacks': [cb.state_dict() for cb in self.callbacks],
            'payload': self.state_payload(),
            'data': self.data_state(),
        }
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as f:
                torch.save(payload, f)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        self._callback('on_checkpoint', path)

    def load_weights(self, path: str, strict: bool = False,
                     verbose: bool = True) -> Dict[str, List[str]]:
        '''
        Warm-start the model from a checkpoint, leaving the training state alone.

        This is the difference between *resuming* and *continuing*: `load_checkpoint()`
        restores the step counter, the optimizer, the learning-rate schedule and the
        data-flow cursors together with the weights, while this method takes **only the
        weights**. Continuing pre-training on a new corpus wants exactly that, because
        the AdamW moments, the schedule position and the corpus cursor of the old run
        all describe the old data - keeping them would restart the new corpus in the
        middle of the warmup and step the optimizer as if it had seen the new tokens.

        The file is read through `motif.model.checkpoint.load_checkpoint`, so every
        layout understood there works: a bare `model.state_dict()` (what
        `motif/pipeline/snapshot.py` writes), a pipeline checkpoint (the `payload['model']`
        of a `last.pt` / `step_*.pt`) and safetensors. Wrapper prefixes (`module.`,
        `_orig_mod.`) are stripped on the way in, so a checkpoint written under DDP or
        `torch.compile` loads verbatim.

        The tensors land in the raw model, so this may be called before `setup()` (as
        `PretrainPipeline` does, on the CPU, before DDP replicates the weights) or after
        it (DDP replicas and a compiled module share the same parameter tensors).

        Args:
            path (str): Checkpoint file or directory to read the weights from.
            strict (bool): Forwarded to `load_state_dict`. False (the default) reports
                the mismatches and keeps the tensors that did line up, which is what a
                warm start wants when the source config is not byte-identical; True
                raises instead.
            verbose (bool): Print the detected layout and the mismatch report on the
                main rank.

        Returns:
            Dict[str, List[str]]: `{'missing': [...], 'unexpected': [...]}`, both empty
            when the state dict lined up exactly.

        Raises:
            FileNotFoundError: If `path` does not exist.
            ValueError: If the file cannot be interpreted as a checkpoint.
            RuntimeError: If `strict` is set and the keys do not match.
        '''
        loaded = read_any_checkpoint(path, map_location='cpu')
        report = self._model.load_state_dict(loaded.state_dict, strict=strict)
        missing = list(getattr(report, 'missing_keys', None) or [])
        unexpected = list(getattr(report, 'unexpected_keys', None) or [])

        if verbose and self.is_main:
            print(f'[*] weights <- {path} ({loaded.source}, '
                  f'{len(loaded.state_dict)} tensors)')
            if missing or unexpected:
                print(f'[!] {len(missing)} missing / {len(unexpected)} unexpected '
                      f'key(s); the tensors behind them keep their current value:')
                for name in (unexpected + missing)[:8]:
                    print(f'      {name}')
                extra = len(missing) + len(unexpected) - 8
                if extra > 0:
                    print(f'      ... and {extra} more')

        return {'missing': missing, 'unexpected': unexpected}

    def load_checkpoint(self, path: str, strict: bool = True) -> None:
        '''
        Restore the training progress, the weights and the data-flow cursors.

        Args:
            path (str): Checkpoint file.
            strict (bool): Verify that the checkpoint belongs to this pipeline class.

        Raises:
            ValueError: If `strict` is set and the checkpoint was written by a
                different pipeline.
        '''
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if strict and ckpt.get('__pipeline__') != self.name:
            raise ValueError(
                f'the checkpoint belongs to {ckpt.get("__pipeline__")!r} while this pipeline is {self.name!r}.'
            )

        self.state = TrainState(**ckpt['state'])
        self.state.phase = PipelinePhase.INIT
        self._pending_payload = ckpt.get('payload')
        self._pending_data = dict(ckpt.get('data') or {})

        for cb, saved in zip(self.callbacks, ckpt.get('callbacks', [])):
            cb.load_state_dict(saved)

        if self._setup_done:
            self._apply_pending()
            self._on_resume_ready()

    def _apply_pending(self) -> None:
        '''
        Land the cached checkpoint payload (weights / optimizer / scheduler) once
        `setup()` has finished.

        The data-flow state is not restored here: a subclass calls
        `_restore_data_state()` itself after building its datasets, because the
        pre-training executor only exists once the dataset has been handed over.
        '''
        if self._pending_payload is not None:
            payload, self._pending_payload = self._pending_payload, None
            self._load_payload(payload)

    def _on_resume_ready(self) -> None:
        '''
        Hook called after `load_checkpoint()` in the case where `setup()` has already
        run.

        By default it assumes the data objects of the subclass are still alive and
        simply pushes the state back. A subclass that rebuilds its data objects (the
        pre-training plan executor is recreated every round, for instance) overrides
        this and restores once the new object exists.
        '''
        self._restore_data_state()

    def _save_tagged(self, tag: str, rotate: bool = True) -> str:
        '''
        Save a checkpoint under a tag name and optionally rotate the old ones.

        Args:
            tag (str): File tag, i.e. `<ckpt_dir>/<tag>.pt`.
            rotate (bool): Keep at most `keep_last` tagged checkpoints.

        Returns:
            str: The path that was written.
        '''
        path = os.path.join(self.ckpt_dir, f'{tag}.pt')
        self.save_checkpoint(path)
        if not self.is_main:
            return path
        print(f'[*] checkpoint -> {path}')
        if rotate:
            self._saved.append(path)
            while len(self._saved) > max(1, self.keep_last):
                victim = self._saved.pop(0)
                try:
                    os.remove(victim)
                except OSError:
                    pass
        return path

    def _maybe_periodic_save(self) -> None:
        '''
        Write the periodic checkpoints when the current step hits
        `save_every_steps`.
        '''
        if self.save_every_steps and self.state.global_step % self.save_every_steps == 0:
            self._save_tagged(f'step_{self.state.global_step}')
            self._save_tagged('last', rotate=False)

    def _autocast(self, enabled: bool):
        '''
        Build the autocast context; when it is disabled an empty context is returned,
        so no dtype is passed to an unsupported device.

        Args:
            enabled (bool): Whether autocast should be active.

        Returns:
            A context manager to wrap the forward pass in.
        '''
        if not enabled:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def _optimizer_step(
        self,
        window: List[Any],
        max_grad_norm: Optional[float],
        amp_enabled: bool,
    ) -> Dict[str, float]:
        '''
        Run one gradient-accumulation window: N forward passes, N backward passes and
        a single optimizer step.

        Args:
            window (List[Any]): The batches accumulated into this step.
            max_grad_norm (Optional[float]): Gradient-norm clipping threshold; None
                computes the norm without clipping it.
            amp_enabled (bool): Whether the forward passes run under autocast.

        Returns:
            Dict[str, float]: The metrics of this step.
        '''
        assert self._optimizer is not None
        self._optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_tokens = 0.0
        metrics: Dict[str, float] = {}
        for batch in window:
            with self._autocast(amp_enabled):
                loss, batch_metrics = self.compute_loss(batch)
            loss = self._scale_loss(loss, batch_metrics)
            (loss / len(window)).backward()
            total_loss += float(loss.detach().float())
            total_tokens += float(batch_metrics.get('tokens', 0.0))
            metrics.update(batch_metrics)

        params = [p for p in self._model.parameters() if p.requires_grad]
        metrics['grad_norm'] = float(
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm or float('inf'))
        )

        self._optimizer.step()
        if self._scheduler is not None:
            self._scheduler.step()

        self.state.advance_step()
        now = time.time()
        elapsed = now - self._last_step_time
        self._last_step_time = now

        metrics['loss/train'] = total_loss / len(window)
        if total_tokens:
            metrics['tokens'] = total_tokens
        metrics['lr'] = self.current_lr
        if elapsed > 0:
            metrics['speed/steps_per_sec'] = 1.0 / elapsed
            if total_tokens:
                metrics['speed/tokens_per_sec'] = total_tokens / elapsed
        if self._total_steps:
            metrics['progress/percent'] = 100.0 * self.state.global_step / self._total_steps

        metrics = self._reduce_step_metrics(metrics)
        self.log(metrics)
        self._callback('on_step_end', metrics)
        self._maybe_periodic_save()
        return metrics

    def _run_epoch(
        self,
        batches: Iterator[Any],
        grad_accum_steps: int,
        max_grad_norm: Optional[float],
        amp_enabled: bool,
        steps_per_epoch: Optional[int],
    ) -> str:
        '''
        Run one epoch and return why it ended:

            'exhausted' the data was consumed completely
            'steps'     `steps_per_epoch` was reached (the epoch is simply that long)
            'stopped'   early stopping interrupted it - the epoch did **not** finish,
                        so the caller must not advance the epoch counter

        Args:
            batches (Iterator[Any]): The batches of this epoch.
            grad_accum_steps (int): Batches accumulated per optimizer step.
            max_grad_norm (Optional[float]): Gradient-norm clipping threshold.
            amp_enabled (bool): Whether the forward passes run under autocast.
            steps_per_epoch (Optional[int]): Maximum number of optimizer steps; None
                means unlimited.

        Returns:
            str: The reason the epoch ended.
        '''
        window: List[Any] = []
        steps = 0
        for batch in batches:
            window.append(batch)
            if len(window) < grad_accum_steps:
                continue
            self._optimizer_step(window, max_grad_norm, amp_enabled)
            window = []
            steps += 1
            if steps_per_epoch is not None and steps >= steps_per_epoch:
                return 'steps'
            if self._sync_stop():
                return 'stopped'
        if window:
            self._optimizer_step(window, max_grad_norm, amp_enabled)
        return 'exhausted'

    def log(self, metrics: Dict[str, float]) -> None:
        '''
        Record one step of metrics in the tracker.

        Args:
            metrics (Dict[str, float]): Metric name to value.
        '''
        self.metrics.log(metrics)

    def _track_best(self, eval_metrics: Dict[str, float]) -> None:
        '''
        Update the best-metric bookkeeping from an evaluation.

        The tracked metric is chosen once, as the key with the largest value of the
        first non-empty evaluation, and reused afterwards.

        Args:
            eval_metrics (Dict[str, float]): Metrics of the evaluation.
        '''
        if not eval_metrics:
            return
        name = self.state.best_metric_name
        if name is None or name not in eval_metrics:
            name = max(eval_metrics, key=lambda k: eval_metrics[k])
            self.state.best_metric_name = name
        current = eval_metrics[name]
        if current > self.state.best_metric:
            self.state.best_metric = current

    def train(
        self,
        dataset: Any = None,
        num_epochs: int = 1,
        steps_per_epoch: Optional[int] = None,
        batch_fn: Optional[Callable[[Any], Iterator[Any]]] = None,
        grad_accum_steps: int = 1,
        use_amp: Optional[bool] = None,
        max_grad_norm: Optional[float] = None,
        eval_every_epochs: Optional[int] = None,
        eval_dataset: Any = None,
        resume_from: Optional[str] = None,
    ) -> Dict[str, float]:
        '''
        Run the whole training job.

        Args:
            dataset: Data source; how it is interpreted is up to the subclass's
                `iterate_epochs`.
            num_epochs (int): Epoch budget.
            steps_per_epoch (Optional[int]): Maximum number of **optimizer steps** per
                epoch (None = unlimited).
            batch_fn (Optional[Callable[[Any], Iterator[Any]]]): Splits one epoch of
                data into batches; the default iterates it directly.
            grad_accum_steps (int): Number of batches accumulated before one
                optimizer step.
            use_amp (Optional[bool]): Whether to enable autocast; None uses the
                pipeline's `amp` (True by default, effective on CUDA only).
            max_grad_norm (Optional[float]): Global gradient-norm clipping threshold
                (None disables it).
            eval_every_epochs (Optional[int]): Run `evaluate()` every this many
                epochs.
            eval_dataset (Any): Evaluation dataset.
            resume_from (Optional[str]): Checkpoint path; None plus `auto_resume`
                looks for `last.pt` automatically.

        Returns:
            Dict[str, float]: The final metric snapshot.

        Raises:
            ValueError: If `grad_accum_steps` is smaller than 1.
        '''
        if grad_accum_steps < 1:
            raise ValueError(f'grad_accum_steps must be >= 1, got {grad_accum_steps}')

        self._hint_about_idle_gpus()

        if resume_from is None and self.auto_resume:
            auto = os.path.join(self.ckpt_dir, 'last.pt')
            if os.path.exists(auto):
                resume_from = auto
                if self.is_main:
                    print(f'[*] auto-resume from {auto}')

        if resume_from is not None:
            self.load_checkpoint(resume_from)
            if self.is_main:
                print(f'[*] resumed: step={self.state.global_step}, epoch={self.state.epoch}')

        if not self._setup_done:
            self._grad_accum_steps = int(grad_accum_steps)
            self.setup()
            self._setup_done = True

        amp_enabled = (self.amp if use_amp is None else use_amp) and self.device.type == 'cuda'

        self.state.phase = PipelinePhase.TRAINING
        self.state.start_time = time.time() - self.state.elapsed
        self._last_step_time = time.time()
        self._callback('on_train_start')

        stop = False
        try:
            epoch_iter = self.iterate_epochs(dataset)
            while self.state.epoch < num_epochs and not stop:
                self._callback('on_epoch_start')
                try:
                    data = next(epoch_iter)
                except StopIteration:
                    break
                batches = batch_fn(data) if batch_fn else iter(data)

                reason = self._run_epoch(batches, grad_accum_steps, max_grad_norm,
                                         amp_enabled, steps_per_epoch)

                if reason == 'stopped':
                    self._save_tagged('last', rotate=False)
                    stop = True

                else:
                    self.state.epoch += 1
                    self._callback('on_epoch_end')
                    self._save_tagged('last', rotate=False)

                    if (eval_every_epochs and eval_dataset is not None
                            and self.state.epoch % eval_every_epochs == 0):
                        self.state.phase = PipelinePhase.EVALUATING
                        eval_metrics = self.evaluate(eval_dataset)
                        self._track_best(eval_metrics)
                        self.log(eval_metrics)
                        self._callback('on_evaluate', eval_metrics)
                        self.state.phase = PipelinePhase.TRAINING

                    if self._sync_stop():
                        stop = True

        except KeyboardInterrupt:
            self.state.interrupted = True
            self.state.phase = PipelinePhase.INTERRUPTED
            if self.is_main:
                print(f'\n[!] interrupted at step {self.state.global_step}, saving an emergency checkpoint...')
            self._callback('on_interrupt')
            self._save_tagged('interrupted', rotate=False)
            self._save_tagged('last', rotate=False)
        except BaseException:
            self.state.phase = PipelinePhase.FAILED
            self._save_tagged('emergency', rotate=False)
            raise
        else:
            self.state.phase = PipelinePhase.FINISHED
        finally:
            self.state.elapsed = time.time() - self.state.start_time
            self._callback('on_train_end')
            self._dist.barrier()
            self.teardown()

        return self.metrics.snapshot()

    def __enter__(self) -> 'BasicPipeline':
        '''
        Run `setup()` once and return the pipeline, so it can be used as a context
        manager.

        Returns:
            BasicPipeline: This pipeline.
        '''
        if not self._setup_done:
            self.setup()
            self._setup_done = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        '''
        Tear the pipeline down when leaving the `with` block.

        Args:
            exc_type: Exception type, when one was raised.
            exc: The exception instance, when one was raised.
            tb: Traceback, when an exception was raised.
        '''
        self.teardown()

    def __repr__(self) -> str:
        '''
        Returns:
            str: A short debug representation with the device, step, epoch and phase.
        '''
        return (f'{self.name}(device={self.device}, step={self.state.global_step}, '
                f'epoch={self.state.epoch}, phase={self.state.phase.name})')
