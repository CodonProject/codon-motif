import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from motif.pipeline.base import (
    BasicPipeline,
    resolve_boundary_token_id,
    resolve_eos_token_id,
)
from motif.pipeline.callback import Callback
from motif.pipeline.distributed import widen_global_batch_tokens
from motif.pipeline.probe import Probe
from motif.utils.plan import ContextTrainingPlanner, StatefulPlanRunner, TrainingPlan


@dataclass
class PretrainConfig:
    '''
    Pre-training configuration: curriculum, optimizer, checkpointing and probe.

    Attributes:
        compiled (bool): Compile the model with `torch.compile` before training.
        base_context (int): Context length of the first curriculum stage.
        target_context (int): Context length the curriculum expands to.
        global_batch_tokens (int): Tokens consumed per optimiser step.
        step_mode (str): Budget preset, one of 'min' / 'recommended' / 'overfit'.
        total_tokens (Optional[int]): Explicit token budget for the whole run; None
            derives it from `step_mode` and the parameter count. It replaces that preset
            (the stage weights stay the same, so the curriculum only gets longer or
            shorter), and it is ignored when `total_steps` is set.
        total_steps (Optional[int]): Explicit step count of the whole run; None derives
            it from the token budget. It wins over both `total_tokens` and `step_mode`:
            the warmup and the cosine schedule are laid out over exactly these steps, and
            the weights only decide how they are split across the stages.
        full_data (bool): Take the budget from the corpus instead: the run consumes the
            **whole training corpus once**, so `total_tokens` becomes the body tokens
            `TextFileDataset.count_tokens()` reports (after `drop_lang`, `dataset_type`
            and `drop_last` have had their say). Counting is a full tokenization pass done
            in `setup()`, before the first step; every rank of a distributed run walks the
            corpus itself, and because the answer only depends on the corpus, the ranks
            agree. A resume does not count again: it reads the budget recorded in the
            checkpoint being resumed. False by default. Giving `total_tokens` together
            with it is contradictory and raises; `total_steps` wins and leaves `full_data`
            a no-op.
            The per-stage split still applies, so "one pass" means the *stages together*
            consume one pass: Foundation and Stabilization each see their share of the
            corpus, not all of it.
        learning_rate (float): Peak learning rate.
        weight_decay (float): Weight decay applied to the decayed parameter group.
        beta1 (float): AdamW beta1.
        beta2 (float): AdamW beta2.
        eps (float): AdamW epsilon.
        grad_clip_norm (float): Gradient-norm clipping threshold.
        min_lr_ratio (float): Final learning rate as a fraction of the peak.
        warmup_steps (Optional[int]): Warmup steps; None derives them from the total
            step count.
        ckpt_dir (str): Checkpoint directory.
        save_every_steps (int): Write a periodic checkpoint every this many steps.
        keep_last (int): Number of periodic checkpoints kept on disk.
        auto_resume (bool): Load `<ckpt_dir>/last.pt` automatically in `train()`.
        init_from (Optional[str]): Checkpoint the **warm start** reads the model weights
            from; None (the default) disables it. Only the weights are taken - the
            optimizer, the learning-rate schedule, the step counter and the data cursor
            of this run all start from zero, which is what continuing pre-training on a
            new corpus wants. Any layout `motif.model.checkpoint.load_checkpoint`
            understands works (pipeline checkpoint, bare `state_dict()`, safetensors).
            Ignored when this run resumes its own `<ckpt_dir>/last.pt`, so later
            invocations never overwrite newer weights with older ones.
        init_strict (bool): Forwarded to `load_state_dict` by the warm start. False (the
            default) reports the missing / unexpected keys and keeps the tensors that
            matched; True raises instead.
        amp (bool): fp32 weights with bf16 autocast. Activations run in bf16 (saving
            memory) and the matrix multiplies use the bf16 tensor cores, while the
            parameters and the AdamW moments stay fp32 (so there is no low-precision
            optimizer pitfall). The custom autograd Functions in `motif/ops` are
            covered by `autocast_safe` in `motif/ops/autocast.py`: they compute in the
            input precision internally and never mix dtypes.
        use_progress (bool): Show the progress bar.
        probe_prompts (List[str]): Prompts used by the generation probe.
        probe_every_steps (int): Run the probe every this many steps; 0 disables it.
        probe_max_new_tokens (int): Generation budget of a probe.
        probe_temperature (float): Sampling temperature of a probe.
        tensorboard (Optional[Any]): Monitoring plug-in (TensorBoard). None disables
            it, `TensorBoardConfig(log_dir=...)` switches it on, and a ready
            `TensorBoardCallback` is accepted too.
        log_file (Optional[Any]): Run log. None disables it, `FileLogConfig(path=...)`
            or a plain path string switches it on, and a ready `FileLogger` is
            accepted too. Both `torch.utils.tensorboard` and the file writer degrade
            gracefully ("use it when available, otherwise carry on") and never affect
            training. When the log file is on, the sanity check (describe + curriculum
            table) is written first and the per-step progress is flushed as it goes.
        snapshot_every_percent (Optional[float]): Weight snapshots; None or 0 disables
            them. A percentage (5, say) saves a **bare model state** (only
            `model.state_dict()`, no optimizer / scheduler / data-flow state) every 5%
            of the total steps, by default into
            `<ckpt_dir>/snapshots/step_000012345.pt`. The targets follow from
            `total_steps` alone, so they are reproducible across resumes.
        snapshot_dir (Optional[str]): Snapshot directory; None means
            `<ckpt_dir>/snapshots`.
        snapshot_at_end (bool): Write one more snapshot when training winds down.
        eos_token_id (Optional[Union[int, str]]): End-of-sequence token; None derives
            it from the tokenizer (`fast_tokenizer.eos_token_id` -> `<|im_end|>`). It
            accepts an id or token text such as 'im_end' / '<|im_end|>'.
        boundary_token_id (Optional[Union[int, str]]): Document boundary token inserted
            after every record while packing; None takes `<|end_of_text|>` from the
            tokenizer and falls back to `eos_token_id`. Also accepts an id or token
            text.
        mask_boundary_token (bool): True (the default) treats the boundary token as a
            context separator only and writes -100 at its label positions, so **no
            loss is computed on it**; False makes it a prediction target as well (the
            older behaviour, where the model had to learn to emit the document end
            marker itself).
        distributed (Optional[Any]): Automatic multi-GPU mode; None uses the defaults
            of `DistributedConfig`, a `DistributedConfig(...)` tunes it, and
            `DistributedConfig(auto_spawn=False)` / `follow_env=False` keep the run
            single-process. A launch environment (`torchrun`, `motif.pipeline.launch`)
            is joined either way, because N uncoordinated ranks would write the same
            checkpoint file.
        auto_widen_batch (bool): When the ranks of a distributed run cannot split a
            stage's batch (fewer rows than ranks), widen `global_batch_tokens` to
            `world_size * (target_context + 1)` instead of failing. True by default;
            False raises a `ValueError` instead.

    The pipeline drills the context-length curriculum (Foundation -> Expansion_* ->
    Stabilization):

        from motif.data import TextFileDataset

        pipe = PretrainPipeline(model, tokenizer, PretrainConfig(
            target_context=4096, base_context=512, global_batch_tokens=8192,
            ckpt_dir='./ckpt',
        ))
        pipe.train(TextFileDataset('D:/corpus', recursive=True, tokenizer=tokenizer,
                                   seq_len=4096, drop_last=False))

    Resuming is the default behaviour: `train()` looks for `<ckpt_dir>/last.pt` on its
    own and restores the model weights, the optimizer, the learning-rate schedule and
    the stage position plus token buffer of the `StatefulPlanRunner`, so an interruption
    never replays corpus that was already trained on.

    Continuing pre-training on a new corpus (warm start) is the other direction:
    `init_from` names a checkpoint whose **weights only** initialise the run, while the
    optimizer, the learning-rate schedule, the step counter and the data cursor all
    start from scratch - the old AdamW moments and the old corpus position describe the
    old data. Any layout `motif.model.checkpoint.load_checkpoint` understands is
    accepted (a full pipeline checkpoint, a bare `state_dict()` as written by the
    snapshot plug-in, or safetensors), and wrapper prefixes (`module.`, `_orig_mod.`)
    are stripped. Give the continued run a **new** `ckpt_dir` so the old run stays
    untouched. `init_from` is read **once**: the second invocation of that command finds
    `<ckpt_dir>/last.pt` and resumes the continued run properly (weights *and* optimizer
    *and* cursor), and the stale `init_from` is ignored with a note - otherwise every
    restart would throw away everything trained since the first checkpoint. Because the
    new corpus is a different length distribution from the old one, this is also where
    `learning_rate` and `warmup_steps` usually want a value of their own.

    How long the run is: `step_mode` sizes the run from the parameter count alone
    ('min' / 'recommended' / 'overfit'). Three switches replace that number and only
    change the *length* - the shape of the curriculum stays the same either way:
    `total_tokens=...` (an explicit token budget), `total_steps=...` (an explicit step
    count) or `full_data=True` (exactly one pass over the corpus). `total_steps` wins
    over `total_tokens`, which wins over `step_mode`; `total_steps` also lays out the
    warmup and the cosine schedule over exactly those steps, while the stage weights
    only decide how they are split across the stages (largest remainder, so no stage
    ever gets zero steps). `full_data=True` takes the budget from the corpus instead:
    `TextFileDataset.count_tokens()` walks it once, after `drop_lang` / `dataset_type`
    have had their say, so the stages **together** consume one pass; it is counted at
    the start of the run, before the first step, which is why the dataset has to be
    handed to `train()`. Combining it with `total_tokens` is contradictory and raises a
    `ValueError`, while `total_steps` simply wins and leaves `full_data` a no-op.

    While packing, a **document boundary token** (`<|end_of_text|>`, see
    `boundary_token_id`) is appended after every record: it is the explicit "a new
    document starts here" signal and, unless `mask_boundary_token=True`, also a
    prediction target counted in the loss - so even when many short documents are packed
    into one sequence the model still has a boundary to rely on. To study how the
    parameters evolve, switch on `snapshot_every_percent`: every n percent of the total
    steps a **bare model state** (only `model.state_dict()`) is saved to
    `<ckpt_dir>/snapshots/`.

    Automatic multi-GPU: the same script runs on one GPU or on eight. Put the
    construction into a `build` function and hand it to `launch()`, which spawns one
    worker per visible GPU (or simply runs it inline when there is only one); see
    `DistributedContext`.

        from motif.pipeline import DistributedConfig, launch

        def build(rank, world_size, local_rank):
            model = MotifChordModel(MotifChordConfig())
            pipe = PretrainPipeline(model, tokenizer, PretrainConfig(
                target_context=4096, base_context=512, global_batch_tokens=8192,
                ckpt_dir='./ckpt', distributed=DistributedConfig(),
            ))
            pipe.train(TextFileDataset(...))

        if __name__ == '__main__':          # spawn re-imports this module
            launch(build)

    `torchrun --nproc_per_node=8` is equivalent - `launch` notices the launch
    environment. What is preserved is the meaning of the configuration:
    `global_batch_tokens` stays the batch of **one optimiser step over all ranks**, so
    the curriculum, the step count and the learning-rate schedule are the same as on one
    GPU, and each rank computes its own contiguous row slice of every chunk. When that
    batch is too small to give every rank a row (a 4096-token stage with
    `global_batch_tokens=8192` holds only 2 rows), it is widened to
    `world_size * (target_context + 1)` - turn that off with `auto_widen_batch=False` to
    get a `ValueError` instead. The checkpoint of a distributed run is an ordinary
    single-model checkpoint (rank 0 writes it, weights are unwrapped from DDP), and it
    is **world-size agnostic**: start on one GPU, resume on four.
    '''

    compiled: bool = False

    base_context: int = 512
    target_context: int = 4096
    global_batch_tokens: int = 8192 * 2
    step_mode: str = 'recommended'

    total_tokens: Optional[int] = None
    total_steps: Optional[int] = None
    full_data: bool = False

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip_norm: float = 1.0
    min_lr_ratio: float = 0.1
    warmup_steps: Optional[int] = None

    ckpt_dir: str = './ckpt'
    save_every_steps: int = 2000
    keep_last: int = 3
    auto_resume: bool = True

    init_from: Optional[str] = None
    init_strict: bool = False

    amp: bool = True
    use_progress: bool = True

    probe_prompts: List[str] = field(default_factory=lambda: ['The', '这'])
    probe_every_steps: int = 2000
    probe_max_new_tokens: int = 64
    probe_temperature: float = 0.8

    tensorboard: Optional[Any] = None
    log_file: Optional[Any] = None

    snapshot_every_percent: Optional[float] = None
    snapshot_dir: Optional[str] = None
    snapshot_at_end: bool = True

    eos_token_id: Optional[Union[int, str]] = None
    boundary_token_id: Optional[Union[int, str]] = None
    mask_boundary_token: bool = True

    distributed: Optional[Any] = None
    auto_widen_batch: bool = True


def cal_warmup(total_steps: int) -> int:
    '''
    Derive the number of warmup steps from the total step count (5% for long runs,
    8% for short ones).

    Args:
        total_steps (int): Total number of optimiser steps.

    Returns:
        int: The warmup step count.
    '''
    if total_steps >= 20000:
        return max(2000, min(10000, int(total_steps * 0.05)))
    return max(1, int(total_steps * 0.08))


class WarmupCosineSchedule(torch.optim.lr_scheduler.LRScheduler):
    '''Linear warmup followed by a cosine decay down to `min_lr_ratio * peak`.'''

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1, last_epoch=-1):
        '''
        Build the schedule.

        Args:
            optimizer: The optimizer whose learning rate is scheduled.
            warmup_steps (int): Number of linear warmup steps (at least 1).
            total_steps (int): Total number of steps of the run (at least 1).
            min_lr_ratio (float): Final learning rate as a fraction of the peak.
            last_epoch (int): Index of the last epoch, as required by `LRScheduler`.
        '''
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        '''
        Learning rate for the current step.

        Returns:
            list: One learning rate per parameter group.
        '''
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            scale = step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos
        return [base * scale for base in self.base_lrs]


class PretrainPipeline(BasicPipeline):
    '''Runs the context curriculum described by a `TrainingPlan`; one epoch equals one full pass over that plan.'''

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[PretrainConfig] = None,
        device: Union[str, torch.device, None] = None,
        callbacks: Optional[List[Callback]] = None,
        seed: Optional[int] = None,
    ):
        '''
        Assemble the pipeline: resolve the end / boundary tokens, attach every
        plug-in and create the probe.

        Args:
            model (nn.Module): Model to train.
            tokenizer: Tokenizer used to resolve the end / boundary tokens.
            config (Optional[PretrainConfig]): Pre-training configuration; defaults
                to `PretrainConfig()`.
            device (Union[str, torch.device, None]): Device to train on; None lets
                the base class pick one.
            callbacks (Optional[List[Callback]]): Extra callbacks.
            seed (Optional[int]): Random seed.

        Raises:
            ValueError: If no end-of-sequence token can be determined.
        '''
        self.config = config or PretrainConfig()
        cfg = self.config

        self.tokenizer = tokenizer
        self._eos_id = resolve_eos_token_id(tokenizer, cfg.eos_token_id)
        if self._eos_id is None:
            raise ValueError(
                'cannot determine eos_token_id: the tokenizer has no eos token, '
                'please set eos_token_id explicitly in PretrainConfig.'
            )

        boundary = resolve_boundary_token_id(tokenizer, cfg.boundary_token_id)
        if boundary is None:
            boundary = self._eos_id
            if cfg.boundary_token_id is None and tokenizer is not None:
                print(f'[!] the tokenizer has no <|end_of_text|>, the document '
                      f'boundary falls back to eos ({self._eos_id}).')
        self._boundary_id = boundary

        super().__init__(
            model=model,
            device=device,
            callbacks=callbacks,
            seed=seed,
            ckpt_dir=cfg.ckpt_dir,
            save_every_steps=cfg.save_every_steps,
            keep_last=cfg.keep_last,
            amp=cfg.amp,
            auto_resume=cfg.auto_resume,
            use_progress=cfg.use_progress,
            distributed=cfg.distributed,
        )

        self.attach_monitor(cfg.tensorboard)
        self.attach_monitor(cfg.log_file)
        snapshot = self.attach_snapshot(cfg)
        sink = self.attach_text_sinks()
        if snapshot is not None:
            snapshot.sink = sink
        if cfg.probe_every_steps:
            self.callbacks.append(Probe(
                prompts=cfg.probe_prompts,
                every_steps=cfg.probe_every_steps,
                max_new_tokens=cfg.probe_max_new_tokens,
                temperature=cfg.probe_temperature,
                eos_token_id=self._eos_id,
                sink=sink,
            ))

        self._plan: Optional[TrainingPlan] = None
        self._dataset: Any = None
        self._global_batch_tokens: int = cfg.global_batch_tokens
        self._ignore_index = -100

    def describe(self) -> List[str]:
        '''
        Run information followed by the context curriculum table (used by the sanity
        check written to the log / TensorBoard).

        Returns:
            List[str]: The description lines.
        '''
        lines = super().describe()
        if self.world_size > 1:
            per_rank = self._global_batch_tokens // self.world_size
            lines.append(
                f'global batch : {self._global_batch_tokens:,} tokens/step over '
                f'{self.world_size} ranks (~{per_rank:,}/rank)'
            )
        if self._plan is not None:
            lines.append('')
            lines += self._plan.report_lines()
        return lines

    def setup(self) -> None:
        '''
        Move the model to the device, build the curriculum plan and the optimizer, and
        apply any pending checkpoint payload.

        Under a distributed launch the model is wrapped in DDP first and
        `global_batch_tokens` is widened when a stage's batch holds fewer rows than
        there are ranks (see `widen_global_batch_tokens`): the plan has to be identical
        on every rank, and a rank without a row would step out of the collective order.

        How long the run is comes from `step_mode`, unless `total_tokens` /
        `total_steps` / `full_data` say otherwise (see `_resolve_token_budget`). A
        `full_data` budget is counted right here, so the training dataset must already be
        known - `train()` stashes it before calling this.

        Raises:
            ValueError: If the configured batch cannot be split over the ranks and
                `auto_widen_batch` is off, or if the token budget is inconsistent (see
                `_resolve_token_budget`).
        '''
        self._model = self._model.to(self.device)
        self._warm_start()
        self._start_ddp()

        global_batch_tokens = widen_global_batch_tokens(
            self.config.global_batch_tokens, self.world_size, self.config.target_context
        )
        if global_batch_tokens != self.config.global_batch_tokens:
            if not self.config.auto_widen_batch:
                raise ValueError(
                    f'global_batch_tokens={self.config.global_batch_tokens} gives a '
                    f'batch of only '
                    f'{self.config.global_batch_tokens // (self.config.target_context + 1)} '
                    f'rows on the {self.config.target_context}-token stage, which cannot '
                    f'be split over {self.world_size} ranks. Raise it to at least '
                    f'{self.world_size * (self.config.target_context + 1)}, use fewer '
                    f'ranks, or leave auto_widen_batch on to have it raised '
                    f'automatically.'
                )
            if self.is_main:
                print(f'[i] global_batch_tokens widened from '
                      f'{self.config.global_batch_tokens:,} to {global_batch_tokens:,} '
                      f'so that all {self.world_size} ranks get a share of every '
                      f'batch.')
        self._global_batch_tokens = global_batch_tokens

        total_tokens, budget_source = self._resolve_token_budget()
        planner = ContextTrainingPlanner(
            self._model,
            step_mode=self.config.step_mode,
            base_context=self.config.base_context,
            target_context=self.config.target_context,
            global_batch_tokens=global_batch_tokens,
            total_tokens=total_tokens,
            total_steps=self.config.total_steps,
            budget_source=budget_source,
        )
        self._plan = planner.generate_plan()
        self._total_steps = self._plan.total_steps
        if self.is_main:
            self._plan.print_report()

        self.build_optimizer()

        self._apply_pending()

    def _warm_start(self) -> None:
        '''
        Load `config.init_from` unless this run is resuming its own checkpoint.

        A resume always wins: `<ckpt_dir>/last.pt` holds the weights *and* the
        optimizer, the schedule position and the corpus cursor of the run being
        continued, so overwriting its weights with the older `init_from` would throw
        away every step taken since the first checkpoint. The warm start therefore only
        runs when there is no pending payload, i.e. on the first invocation of a
        continue-training run - which is exactly the one that must read the source
        weights.

        The weights are loaded with `BasicPipeline.load_weights` semantics (weights
        only); `init_strict` decides whether a shape-compatible but structurally
        different checkpoint is reported or raised.
        '''
        path = self.config.init_from
        if not path:
            return
        if self._pending_payload is not None:
            if self.is_main:
                print(f'[i] this run resumes from its own checkpoint: init_from '
                      f'({path}) is ignored, its weights are older than the ones being '
                      f'resumed.')
            return
        self.load_weights(path, strict=self.config.init_strict)

    def _resolve_token_budget(self) -> Tuple[Optional[int], Optional[str]]:
        '''
        Decide how long this run is, and which config field decided it.

        The precedence is `total_steps` > `full_data` > `total_tokens` > `step_mode`.
        The first two are horizons in their own right, so `total_steps` makes `full_data`
        a no-op rather than a conflict, while `full_data` and `total_tokens` both claim to
        *be* the budget and therefore contradict each other. Whatever is returned only
        sizes the run: the curriculum weights (Foundation / Expansion / Stabilization) are
        the same in every mode.

        Returns:
            Tuple[Optional[int], Optional[str]]: The token budget (None = the `step_mode`
            preset of the parameter count) and the label the plan should report (None =
            let `step_mode` / `total_steps` name it).

        Raises:
            ValueError: If the budget is not positive, if `full_data` is combined with
                `total_tokens`, or if `full_data` cannot count the corpus.
        '''
        cfg = self.config
        if cfg.total_tokens is not None and cfg.total_tokens <= 0:
            raise ValueError(f'total_tokens must be positive, got {cfg.total_tokens!r}')
        if cfg.total_steps is not None and cfg.total_steps <= 0:
            raise ValueError(f'total_steps must be positive, got {cfg.total_steps!r}')

        if cfg.total_steps is not None:
            if cfg.full_data and self.is_main:
                print(f'[i] full_data is ignored: total_steps={cfg.total_steps:,} fixes '
                      f'the horizon of the run by itself.')
            return cfg.total_tokens, None
        if cfg.full_data:
            if cfg.total_tokens is not None:
                raise ValueError(
                    f'full_data=True computes the budget from the corpus, so total_tokens '
                    f'({cfg.total_tokens!r}) is contradictory: leave it None, or turn '
                    f'full_data off to train for a fixed number of tokens.'
                )
            resumed = self._resumed_budget()
            if resumed is not None:
                return resumed, 'full_data'
            return self._corpus_token_budget(), 'full_data'
        return cfg.total_tokens, None

    def _resumed_budget(self) -> Optional[int]:
        '''
        The token budget the checkpoint being resumed was built with, when it recorded
        one (`state_payload()['budget']`).

        Returning it keeps the plan identical across restarts - the corpus cannot have
        changed, or the data cursor of the checkpoint would be meaningless - and spares a
        full tokenization pass on every restart of a long run. A checkpoint written before
        this field existed simply has none, and the corpus is counted again.

        Returns:
            Optional[int]: The recorded `full_data` budget, or None when there is nothing
            to reuse.
        '''
        payload = self._pending_payload
        if not isinstance(payload, dict):
            return None
        budget = payload.get('budget')
        if not isinstance(budget, dict) or budget.get('step_mode') != 'full_data':
            return None
        tokens = budget.get('total_tokens')
        if not isinstance(tokens, int) or tokens <= 0:
            return None
        if self.is_main:
            print(f'[i] full-data budget taken from the checkpoint being resumed: '
                  f'{tokens:,} tokens (the corpus is not counted again).')
        return tokens

    def _corpus_token_budget(self) -> int:
        '''
        Count the body tokens of the whole training corpus (`full_data=True`).

        The dataset must be reachable already - i.e. handed to `train()`, which stashes it
        before `setup()` builds the plan - and it must be able to count itself
        (`TextFileDataset.count_tokens()`). Every rank counts on its own: the answer is a
        function of the corpus alone, so the ranks agree and no collective is involved.

        Returns:
            int: The number of body tokens the data flow will produce.

        Raises:
            ValueError: If no dataset is known, if it cannot count its tokens, or if it
                turns out to hold none.
        '''
        dataset = self._dataset
        if dataset is None:
            raise ValueError(
                'full_data=True needs the training dataset before setup(): pass it to '
                'train(...), or set total_tokens / total_steps instead.'
            )
        counter = getattr(dataset, 'count_tokens', None)
        if not callable(counter):
            raise ValueError(
                f'full_data=True needs a dataset that can count its tokens '
                f'(`TextFileDataset.count_tokens()`), got {type(dataset).__name__}: set '
                f'total_tokens / total_steps instead.'
            )
        started = time.time()
        tokens = int(counter(progress=self.is_main and self.config.use_progress))
        if tokens <= 0:
            raise ValueError(
                f'full_data=True found no tokens in {dataset!r}: is the corpus empty, or '
                f'is the preprocessing (drop_lang / dataset_type) dropping every row?'
            )
        if self.is_main:
            elapsed = time.time() - started
            root = getattr(dataset, 'root', None)
            try:
                rows = f'{len(dataset):,} rows of '
            except TypeError:
                rows = ''
            print(f'[*] full-data budget: {tokens:,} tokens over {rows}'
                  f'{root if root is not None else type(dataset).__name__} '
                  f'({elapsed:,.1f}s of counting); the corpus is walked once.')

        return tokens

    def build_optimizer(self) -> torch.optim.Optimizer:
        '''
        Build AdamW with a decay / no-decay split and the warmup-cosine schedule.

        Parameters with one dimension or less, and anything whose name contains
        'norm', go into the no-decay group.

        Returns:
            torch.optim.Optimizer: The optimizer that was created.
        '''
        cfg = self.config
        warmup = cfg.warmup_steps if cfg.warmup_steps is not None else cal_warmup(self._total_steps)

        decay, no_decay = [], []
        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or 'norm' in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)

        self._optimizer = torch.optim.AdamW(
            [
                {'params': decay, 'weight_decay': cfg.weight_decay},
                {'params': no_decay, 'weight_decay': 0.0},
            ],
            lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
        )
        self._scheduler = WarmupCosineSchedule(
            self._optimizer,
            warmup_steps=warmup,
            total_steps=self._total_steps,
            min_lr_ratio=cfg.min_lr_ratio,
        )
        return self._optimizer

    def iterate_epochs(self, dataset) -> Iterator[StatefulPlanRunner]:
        '''
        Yield one executor per epoch; once the whole plan has been walked, the next
        epoch starts over from the beginning.

        The executor is rebuilt every round (the plan itself is finite, so it has to
        be reset after a full pass), which is why the data-flow state from the
        checkpoint has to be pushed back in right after the new one is created.

        Every rank walks the same stream and takes its own contiguous row slice of each
        chunk (`shard_config()`), so the world size cancels out of the state: the token
        buffer and the upstream cursor are identical on all ranks, and a checkpoint can
        be resumed with a different number of GPUs.

        Args:
            dataset: Upstream corpus fed to the plan runner.

        Yields:
            StatefulPlanRunner: The executor for the current epoch.
        '''
        self._dataset = dataset
        shard_rank, shard_world = self.shard_config()
        while True:
            runner = StatefulPlanRunner(
                self._plan, dataset, self._eos_id,
                boundary_token_id=self._boundary_id,
                mask_boundary=self.config.mask_boundary_token,
                shard_rank=shard_rank,
                shard_world=shard_world,
            )
            self.register_stateful('plan', runner)
            self._restore_data_state()
            yield runner

    def _on_resume_ready(self) -> None:
        '''
        Drop the executor left over from the previous round; the restore itself is
        deferred until `iterate_epochs` has built the new one.
        '''
        self._stateful.pop('plan', None)

    def compute_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        '''
        Compute the loss of one micro-batch and its auxiliary metrics.

        Args:
            batch: `(stage, inputs, labels)` produced by the plan runner.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: The loss and the metrics
            `seq_len`, `tokens` and `tokens/masked`.
        '''
        stage, inputs, labels = batch
        inputs = inputs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        output = self.model(input_ids=inputs)

        loss = getattr(output, 'loss', None)
        if loss is None:
            logits = getattr(output, 'logits', output)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )

        aux = getattr(output, 'aux_loss', None)
        if aux is not None:
            loss = loss + aux

        trained = int((labels != self._ignore_index).sum())
        return loss, {
            'seq_len': float(stage.seq_len),
            'tokens': float(trained),
            'tokens/masked': float(labels.numel() - trained),
        }

    def state_payload(self) -> Dict[str, Any]:
        '''
        Returns:
            Dict[str, Any]: The model, optimizer and scheduler states that go into
            the checkpoint.
        '''
        return {
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict() if self._optimizer is not None else None,
            'scheduler': self._scheduler.state_dict() if self._scheduler is not None else None,
            'budget': {
                'total_tokens': self._plan.total_tokens if self._plan is not None else None,
                'step_mode': self._plan.step_mode if self._plan is not None else None,
            },
        }

    def _load_payload(self, payload: Dict[str, Any]) -> None:
        '''
        Restore the model, optimizer and scheduler from a checkpoint payload.

        Args:
            payload (Dict[str, Any]): Payload produced by `state_payload()`; missing
                entries are skipped.
        '''
        if payload.get('model') is not None:
            self._model.load_state_dict(payload['model'])
        if self._optimizer is not None and payload.get('optimizer') is not None:
            self._optimizer.load_state_dict(payload['optimizer'])
        if self._scheduler is not None and payload.get('scheduler') is not None:
            self._scheduler.load_state_dict(payload['scheduler'])

    def evaluate(self, dataset=None) -> Dict[str, float]:
        '''
        Average the loss over a dataset, ignoring the masked positions.

        Args:
            dataset: Dataset yielding `(stage, inputs, labels)` items; None returns
                an empty result.

        Returns:
            Dict[str, float]: `{'loss/eval': ...}`, or an empty dict when no dataset
            was given.
        '''
        if dataset is None:
            return {}
        self._model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for item in dataset:
                _, inputs, labels = item
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                output = self.model(input_ids=inputs)
                logits = getattr(output, 'logits', output)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction='sum'
                )
                total += float(loss)
                count += int((labels != self._ignore_index).sum())
        self._model.train()
        return {'loss/eval': total / max(count, 1)}

    def train(self, dataset=None, num_epochs: int = 1, **kwargs) -> Dict[str, float]:
        '''
        Run training, defaulting the gradient clipping to `grad_clip_norm`.

        The dataset is stashed before handing over to `BasicPipeline.train`: that call is
        where `setup()` runs, and a `full_data=True` budget has to see the corpus by then.

        Args:
            dataset: Training dataset.
            num_epochs (int): Number of epochs.
            **kwargs: Forwarded to `BasicPipeline.train`.

        Returns:
            Dict[str, float]: The final metric snapshot.
        '''
        kwargs.setdefault('max_grad_norm', self.config.grad_clip_norm)
        self._dataset = dataset
        return super().train(dataset, num_epochs=num_epochs, **kwargs)
