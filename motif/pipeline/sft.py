import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from motif.data.sft import MotifSFT
from motif.pipeline.base import BasicPipeline, move_to_device, select_model_kwargs
from motif.pipeline.callback import Callback
from motif.pipeline.probe import Probe
from motif.pipeline.sanity import SanityProbe, build_probe_samples
from motif.utils.lora import (
    LoRAConfig,
    apply_lora,
    load_lora,
    lora_state_dict,
    lora_summary,
    save_lora,
)


@dataclass
class SFTConfig:
    '''
    SFT configuration: optimizer, stage declarations and checkpointing.

    Attributes:
        compiled (bool): Compile the model with `torch.compile` before training.
        lora (Optional[Any]): Parameter-efficient mode. None trains every weight; a
            `LoRAConfig` (or plain `True`, which means `LoRAConfig()`) freezes the base
            model and trains only low-rank adapters, and the checkpoint payload then
            holds an `adapter` plus the `lora_spec` instead of a full `model`. A ready
            `SFTStage`/dataset pipeline does not change otherwise: the optimizer simply
            only sees the adapter tensors because everything else is frozen.
        learning_rate (float): Peak learning rate.
        weight_decay (float): Weight decay applied to the decayed parameter group.
        warmup_steps (int): Number of linear warmup steps.
        lr_min_ratio (float): Final learning rate as a fraction of the peak.
        grad_clip_norm (float): Gradient-norm clipping threshold.
        ckpt_dir (str): Checkpoint directory.
        save_every_steps (int): Write a periodic checkpoint every this many steps.
        keep_last (int): Number of periodic checkpoints kept on disk.
        auto_resume (bool): Load `<ckpt_dir>/last.pt` automatically in `train()`.
        amp (bool): fp32 weights with bf16 autocast, as in `PretrainConfig`.
        use_progress (bool): Show the progress bar.
        stage_specs (List[Dict[str, Any]]): Declarative stage list; used to build the
            `MotifSFT` datasets automatically when `stages` is not passed explicitly.
        pad_length (int): Length every SFT sample is padded / truncated to.
        batch_size (int): Number of conversation groups per batch.
        dataset_kwargs (Dict[str, Any]): Extra keyword arguments forwarded to the
            dataset constructor.
        probe_prompts (List[str]): Prompts used by the generation probe.
        probe_every_steps (int): Run the probe every this many steps; 0 disables it.
        probe_max_new_tokens (int): Generation budget of a probe.
        probe_temperature (float): Sampling temperature of a probe.
        sanity_samples (int): Number of fixed training-corpus conversations scored (and
            answered) at every probe, in the chat format training uses; 0 disables it.
            Positive values need `probe_every_steps > 0`, since the sanity probe rides
            the same schedule.
        tensorboard (Optional[Any]): Monitoring plug-in (TensorBoard); same semantics
            as the same-named field of `PretrainConfig`.
        log_file (Optional[Any]): Run log; same semantics as the same-named field of
            `PretrainConfig`. A path string / `TensorBoardConfig` / `FileLogConfig`,
            or a ready callback instance, are all accepted.
        distributed (Optional[Any]): Automatic multi-GPU mode; None uses the defaults
            of `DistributedConfig` and a `DistributedConfig(...)` tunes it. Every rank
            takes its own contiguous slice of each batch, so the global batch of a step
            is still `batch_size`.
    '''

    compiled: bool = False

    lora: Optional[Any] = None

    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    lr_min_ratio: float = 0.1
    grad_clip_norm: float = 1.0

    ckpt_dir: str = './sft_ckpt'
    save_every_steps: int = 2000
    keep_last: int = 3
    auto_resume: bool = True

    amp: bool = True
    use_progress: bool = True

    stage_specs: List[Dict[str, Any]] = field(default_factory=list)
    pad_length: int = 2048
    batch_size: int = 8
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)

    probe_prompts: List[str] = field(default_factory=list)
    probe_every_steps: int = 0
    probe_max_new_tokens: int = 64
    probe_temperature: float = 0.8
    sanity_samples: int = 0

    tensorboard: Optional[Any] = None
    log_file: Optional[Any] = None

    distributed: Optional[Any] = None


@dataclass
class SFTStage:
    '''
    One SFT stage: a dataset, how many epochs to run, and where to export at the end.

    Attributes:
        name (str): Stage name, used in logs and in the stateful registry key.
        dataset (Any): The dataset iterated by this stage.
        epochs (int): Number of epochs this stage runs for.
        ckpt (Optional[str]): Path the weights are exported to after the last epoch
            of the stage; None skips the export.
    '''

    name: str
    dataset: Any
    epochs: int = 1
    ckpt: Optional[str] = None


def build_sft_stages(
    stage_specs: List[Dict[str, Any]],
    tokenizer,
    pad_length: int,
    batch_size: int,
    dataset_cls: type = MotifSFT,
    **dataset_kwargs,
) -> List[SFTStage]:
    '''
    Turn `[{'name','folder','epochs','ckpt'}, ...]` into `List[SFTStage]`.

    Only the `dataset_kwargs` that the dataset constructor actually accepts are
    forwarded, so extra configuration keys never break the build.

    Args:
        stage_specs (List[Dict[str, Any]]): One dict per stage, with at least `name`
            and `folder`.
        tokenizer: Tokenizer handed to every dataset.
        pad_length (int): Length every sample is padded / truncated to.
        batch_size (int): Number of conversation groups per batch.
        dataset_cls (type): Dataset class to instantiate.
        **dataset_kwargs: Extra keyword arguments for that class.

    Returns:
        List[SFTStage]: The stages, in the order they were declared.
    '''
    import inspect

    try:
        accepted = set(inspect.signature(dataset_cls.__init__).parameters)
    except (TypeError, ValueError):
        accepted = None

    stages: List[SFTStage] = []
    for spec in stage_specs:
        kwargs = dict(dataset_kwargs)
        if accepted is not None:
            kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        dataset = dataset_cls(
            folder=spec['folder'],
            tokenizer=tokenizer,
            pad_length=pad_length,
            batch_size=batch_size,
            **kwargs,
        )
        stages.append(SFTStage(
            name=spec['name'],
            dataset=dataset,
            epochs=int(spec.get('epochs', 1)),
            ckpt=spec.get('ckpt'),
        ))
    return stages


class _StageExporter(Callback):
    '''Export the weights to `stage.ckpt` when the last epoch of a stage finishes.'''

    def __init__(self, pipeline: 'SFTPipeline'):
        '''
        Args:
            pipeline (SFTPipeline): The pipeline whose stages are watched.
        '''
        self.pipeline = pipeline

    def on_epoch_end(self, pipeline) -> None:
        '''
        Export the stage weights when its last epoch just ended.

        Args:
            pipeline: The training pipeline.
        '''
        finished = pipeline.state.epoch - 1
        if finished < 0:
            return
        si, ei = pipeline.epoch_pos(finished)
        stage = pipeline.stages[si]
        if ei == stage.epochs - 1 and stage.ckpt:
            pipeline.save_stage(stage)


class SFTPipeline(BasicPipeline):
    '''
    Full-parameter SFT stage by stage; every stage supports several epochs and a
    weight export.

        from motif.data import MotifSFT
        from motif.pipeline import SFTConfig, SFTPipeline, SFTStage

        pipe = SFTPipeline(model, tokenizer, config=SFTConfig(
            pad_length=2048, batch_size=8, ckpt_dir='./sft_ckpt',
            stage_specs=[
                {'name': 'general', 'folder': 'data/general', 'epochs': 2},
                {'name': 'code',    'folder': 'data/code',    'epochs': 1},
            ],
        ))
        pipe.train()

    Parameter-efficient variant: `lora=LoRAConfig(...)` (or plain `lora=True`) freezes
    the base model, trains only the adapters, and exports those adapters - a few MB -
    instead of the whole model, both in the periodic checkpoints and in `stage.ckpt`:

        from motif.utils.lora import LoRAConfig

        pipe = SFTPipeline(model, tokenizer, config=SFTConfig(
            lora=LoRAConfig(r=8, alpha=16.0, dropout=0.05),
            learning_rate=3e-4, pad_length=1024, batch_size=2, ckpt_dir='./sft_ckpt',
            stage_specs=[{'name': 'general', 'folder': 'data/general', 'epochs': 1,
                          'ckpt': 'adapter.safetensors'}],
        ))
        pipe.train()

    Resume semantics: the pipeline's `state.epoch` is a **global** epoch index, while
    every `MotifSFT` records "which global epoch, and how many batches of that epoch
    have been consumed". Both go into the checkpoint, so after restoring an interrupted
    run it continues at batch M of epoch N instead of re-running a whole epoch.

    Sanity probe: `probe_prompts` + `probe_every_steps` generate from a **raw text**
    prompt every N steps, which shows what the model writes but is not the shape it is
    being trained on. `sanity_samples=N` adds a second, chat-formatted probe
    (`SanityProbe`): N fixed conversations of the training corpus are re-rendered
    through the very path the datasets use (`MotifSFT.build_messages()` /
    `render_group()`), so every probe reports a teacher-forced loss / accuracy over the
    supervised turns **and** a reply written in the training format. Its scalars
    (`sanity/loss`, `sanity/ppl`, `sanity/acc`, `sanity/supervised`, `sanity/warnings`)
    join the step metrics, reach TensorBoard and the run log like any other curve, and
    the generated text goes out through the text sinks as `Sanity/sample_<i>`.
    '''

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: SFTConfig,
        stages: Optional[List[SFTStage]] = None,
        device: Union[str, torch.device, None] = None,
        callbacks: Optional[List[Callback]] = None,
        seed: Optional[int] = None,
    ):
        '''
        Assemble the pipeline: resolve the stages, attach every plug-in and build the
        global-epoch mapping.

        Args:
            model (nn.Module): Model to fine-tune.
            tokenizer: Tokenizer handed to the datasets and to the probe.
            config (SFTConfig): SFT configuration; required.
            stages (Optional[List[SFTStage]]): Explicit stages; None builds them from
                `config.stage_specs`.
            device (Union[str, torch.device, None]): Device to train on; None lets the
                base class pick one.
            callbacks (Optional[List[Callback]]): Extra callbacks.
            seed (Optional[int]): Random seed.

        Raises:
            ValueError: If `config` is None, if no stage is available, or if a stage
                declares fewer than one epoch.
        '''
        if config is None:
            raise ValueError('SFTPipeline needs config=SFTConfig(...).')

        self.config = config
        cfg = self.config

        self.tokenizer = tokenizer
        self.lora_config: Optional[LoRAConfig] = None
        self._stages: List[SFTStage] = (
            list(stages) if stages is not None else self._build_stages_from_config()
        )
        if not self._stages:
            raise ValueError('SFTPipeline needs at least one stage.')
        for stage in self._stages:
            if stage.epochs < 1:
                raise ValueError(f'stage {stage.name!r} must have epochs >= 1.')

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
        sink = self.attach_text_sinks()
        if cfg.probe_every_steps and cfg.probe_prompts:
            self.callbacks.append(Probe(
                prompts=cfg.probe_prompts,
                every_steps=cfg.probe_every_steps,
                max_new_tokens=cfg.probe_max_new_tokens,
                temperature=cfg.probe_temperature,
                sink=sink,
            ))
        if cfg.sanity_samples < 0:
            raise ValueError(f'sanity_samples must be >= 0, got {cfg.sanity_samples!r}.')
        if cfg.sanity_samples and not cfg.probe_every_steps:
            raise ValueError(
                'sanity_samples needs probe_every_steps > 0: the sanity probe rides the '
                'same schedule as the generation probe.'
            )
        if cfg.sanity_samples:
            self.callbacks.insert(0, SanityProbe(
                samples=build_probe_samples(
                    [stage.dataset for stage in self._stages], cfg.sanity_samples,
                ),
                tokenizer=self.tokenizer,
                every_steps=cfg.probe_every_steps,
                max_new_tokens=cfg.probe_max_new_tokens,
                temperature=cfg.probe_temperature,
                sink=sink,
            ))
        self.callbacks.append(_StageExporter(self))

        self._current_stage: Optional[SFTStage] = None

        self._stage_offsets: List[int] = []
        offset = 0
        for stage in self._stages:
            self._stage_offsets.append(offset)
            offset += stage.epochs
        self._total_epochs = offset

    def describe(self) -> List[str]:
        '''
        Run information followed by the stage table (used by the sanity check written
        to the log / TensorBoard).

        Returns:
            List[str]: The description lines.
        '''
        lines = super().describe()
        summary = lora_summary(self._model, self.lora_config)
        if summary:
            lines.append(f'lora        : {summary}')
        lines.append(f'epochs      : {self._total_epochs:,}')
        for i, stage in enumerate(self._stages):
            lines.append(
                f'-> [{i + 1}] {stage.name:<18} | epochs={stage.epochs:<4} | '
                f'ckpt={stage.ckpt}'
            )
        return lines

    def _build_stages_from_config(self) -> List[SFTStage]:
        '''
        Build the stages from `SFTConfig.stage_specs`.

        Returns:
            List[SFTStage]: The stages declared in the configuration.

        Raises:
            ValueError: If `stage_specs` is empty.
        '''
        cfg = self.config
        if not cfg.stage_specs:
            raise ValueError(
                'SFTPipeline needs either explicit stages=... or stage_specs declared '
                'in SFTConfig (of the form [{"name","folder","epochs"}]).'
            )
        return build_sft_stages(
            cfg.stage_specs, self.tokenizer,
            pad_length=cfg.pad_length, batch_size=cfg.batch_size,
            **dict(cfg.dataset_kwargs or {}),
        )

    @property
    def stages(self) -> List[SFTStage]:
        '''
        Returns:
            List[SFTStage]: The stages of this pipeline, in run order.
        '''
        return self._stages

    @property
    def current_stage(self) -> Optional[SFTStage]:
        '''
        Returns:
            Optional[SFTStage]: The stage being trained right now; None before setup.
        '''
        return self._current_stage

    def epoch_pos(self, epoch: int) -> Tuple[int, int]:
        '''
        Map a global epoch index to `(stage index, epoch within that stage)`.
        Out-of-range values wrap around.

        Args:
            epoch (int): Global epoch index.

        Returns:
            Tuple[int, int]: The stage index and the epoch index inside it.
        '''
        local = epoch % self._total_epochs
        for si, stage in enumerate(self._stages):
            if local < self._stage_offsets[si] + stage.epochs:
                return si, local - self._stage_offsets[si]
        return len(self._stages) - 1, self._stages[-1].epochs - 1

    def setup(self) -> None:
        '''
        Move the model to the device, count the total steps, register every stage
        dataset as a stateful data flow and restore a pending checkpoint.

        Under a distributed launch the model is wrapped in DDP here, before the
        optimizer captures the parameters.
        '''
        self._model = self._model.to(self.device)
        self._setup_lora()
        self._start_ddp()
        accum = max(1, int(self._grad_accum_steps))
        self._total_steps = sum((len(s.dataset) + accum - 1) // accum * s.epochs
                                for s in self._stages)

        self.build_optimizer()

        for si, stage in enumerate(self._stages):
            self.register_stateful(f'stage/{si}:{stage.name}', stage.dataset)

        self._apply_pending()
        self._restore_data_state()
        self._sync_current_stage()

    def _setup_lora(self) -> None:
        '''
        Freeze the base model and attach the low-rank adapters when `config.lora` asks
        for them.

        This runs before the DDP wrap and before the optimizer is built, so the wrapped
        modules are part of the final model and `build_optimizer` only ever sees the
        adapter tensors (everything else has `requires_grad=False`).

        Raises:
            ValueError: If `config.lora` is neither None/False, True nor a `LoRAConfig`.
        '''
        spec = self.config.lora
        if spec is None or spec is False:
            self.lora_config = None
            return
        if spec is True:
            config = LoRAConfig()
        elif isinstance(spec, LoRAConfig):
            config = spec
        else:
            raise ValueError(f'lora must be None, True or a LoRAConfig, got {type(spec).__name__}')

        self.lora_config = apply_lora(self._model, config)
        if self.is_main:
            print(f'[*] {lora_summary(self._model, self.lora_config)}')

    def _sync_current_stage(self) -> None:
        '''
        Make `current_stage` reflect the epoch position that was restored.
        '''
        si, _ = self.epoch_pos(self.state.epoch)
        self._current_stage = self._stages[si]

    def build_optimizer(self) -> torch.optim.Optimizer:
        '''
        Build AdamW with a decay / no-decay split and a linear-warmup plus
        cosine-decay schedule.

        Parameters with one dimension or less, and anything whose name contains
        'norm', go into the no-decay group.

        Returns:
            torch.optim.Optimizer: The optimizer that was created.

        Raises:
            RuntimeError: If the model has no trainable parameter at all.
        '''
        cfg = self.config
        total = max(1, self._total_steps)
        warmup = max(1, min(cfg.warmup_steps, total - 1)) if total > 1 else 1
        eta_min = cfg.lr_min_ratio * cfg.learning_rate

        decay, no_decay = [], []
        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or 'norm' in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)

        param_groups = []
        if decay:
            param_groups.append({'params': decay, 'weight_decay': cfg.weight_decay})
        if no_decay:
            param_groups.append({'params': no_decay, 'weight_decay': 0.0})
        if not param_groups:
            raise RuntimeError('there is no trainable parameter; check whether the model was frozen as a whole.')

        self._optimizer = torch.optim.AdamW(param_groups, lr=cfg.learning_rate)

        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            self._optimizer,
            start_factor=1e-8 / cfg.learning_rate,
            end_factor=1.0,
            total_iters=warmup,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer,
            T_max=max(1, total - warmup),
            eta_min=eta_min,
        )
        self._scheduler = torch.optim.lr_scheduler.SequentialLR(
            self._optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup],
        )
        return self._optimizer

    def _prepare_stage_epoch(self, stage: SFTStage, global_epoch: int) -> None:
        '''
        Switch the stage dataset to the order of `global_epoch`.

        `set_epoch` (which also clears the cursor) is only called when the dataset's
        current epoch differs from the target - when resuming they are the same, so
        batches that were already trained on are never replayed.

        Args:
            stage (SFTStage): Stage whose dataset is prepared.
            global_epoch (int): Global epoch index that is about to run.
        '''
        dataset = stage.dataset
        if not (hasattr(dataset, 'set_epoch') and hasattr(dataset, 'epoch')):
            return
        if dataset.epoch != global_epoch:
            dataset.set_epoch(global_epoch)

    def iterate_epochs(self, dataset=None) -> Iterator[Any]:
        '''
        Endlessly yield the dataset of the next stage to run; every yield is one
        epoch.

        Args:
            dataset: Unused; the stages own their datasets.

        Yields:
            Any: The dataset of the stage that the current epoch belongs to.
        '''
        epoch = self.state.epoch
        while True:
            si, _ = self.epoch_pos(epoch)
            stage = self._stages[si]
            self._current_stage = stage
            self._prepare_stage_epoch(stage, epoch)
            yield stage.dataset
            epoch += 1

    def compute_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        '''
        Compute the loss of one SFT batch.

        The batch's fields are filtered down to the model's accepted keyword
        arguments and moved to the device; the labels, which are position-aligned with
        `input_ids`, are shifted right by one so that every position predicts the
        next one.

        Every rank walks the same dataset and takes its own row slice of each batch
        (`_shard_batch`), which keeps the resumable epoch cursor identical everywhere
        while dividing the work of a step over the ranks.

        Args:
            batch (Dict[str, Any]): One batch produced by `MotifSFT`.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: The loss and an empty metric dict.
        '''
        batch = self._shard_batch(batch)
        labels = batch['labels'].to(self.device, non_blocking=True)

        model_kwargs = {
            key: move_to_device(value, self.device)
            for key, value in select_model_kwargs(self.model, batch).items()
        }
        output = self.model(**model_kwargs)

        loss = getattr(output, 'loss', None)
        if loss is None:
            logits = getattr(output, 'logits', output)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        aux = getattr(output, 'aux_loss', None)
        if aux is not None:
            loss = loss + aux

        return loss, {}

    def state_payload(self) -> Dict[str, Any]:
        '''
        Returns:
            Dict[str, Any]: The model (or, in LoRA mode, the adapter plus the
            `lora_spec` that produced it) together with the optimizer and scheduler
            states that go into the checkpoint.
        '''
        payload: Dict[str, Any] = {
            'optimizer': self._optimizer.state_dict() if self._optimizer is not None else None,
            'scheduler': self._scheduler.state_dict() if self._scheduler is not None else None,
        }
        if self.lora_config is not None:
            payload['adapter'] = lora_state_dict(self._model)
            payload['lora_spec'] = self.lora_config.state()
        else:
            payload['model'] = self._model.state_dict()
        return payload

    def _load_payload(self, payload: Dict[str, Any]) -> None:
        '''
        Restore the model, optimizer and scheduler from a checkpoint payload.

        Args:
            payload (Dict[str, Any]): Payload produced by `state_payload()`; missing
                entries are skipped.

        Raises:
            RuntimeError: If the payload carries an adapter while `config.lora` is off.
        '''
        if payload.get('adapter') is not None:
            if self.lora_config is None:
                raise RuntimeError(
                    'this checkpoint holds a LoRA adapter but config.lora is off; '
                    'resume it with the matching LoRAConfig'
                )
            load_lora(self._model, payload['adapter'], strict=True)
        elif payload.get('model') is not None:
            self._model.load_state_dict(payload['model'])
        if self._optimizer is not None and payload.get('optimizer') is not None:
            self._optimizer.load_state_dict(payload['optimizer'])
        if self._scheduler is not None and payload.get('scheduler') is not None:
            self._scheduler.load_state_dict(payload['scheduler'])

    def save_stage(self, stage: SFTStage) -> None:
        '''
        Export the current weights to `stage.ckpt` (written atomically).

        Args:
            stage (SFTStage): Stage whose `ckpt` path receives the weights; a stage
                without a path is a no-op.
        '''
        if not stage.ckpt:
            return
        if not self.is_main:
            return
        directory = os.path.dirname(os.path.abspath(stage.ckpt))
        os.makedirs(directory, exist_ok=True)
        if self.lora_config is not None:
            save_lora(
                self._model,
                stage.ckpt,
                config=self.lora_config,
                metadata={'stage': stage.name, 'pipeline': self.name},
            )
        else:
            tmp = stage.ckpt + '.tmp'
            torch.save(self._model.state_dict(), tmp)
            os.replace(tmp, stage.ckpt)
        tqdm.write(f'[*] stage {stage.name!r} weights exported -> {stage.ckpt}')

    def evaluate(self, dataset=None) -> Dict[str, float]:
        '''
        Average the next-token loss over a dataset.

        Args:
            dataset: Dataset yielding the same batch dicts as training; None returns
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
            for batch in dataset:
                labels = batch['labels'].to(self.device)
                model_kwargs = {
                    key: move_to_device(value, self.device)
                    for key, value in select_model_kwargs(self.model, batch).items()
                }
                output = self.model(**model_kwargs)
                logits = getattr(output, 'logits', output)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='sum',
                )
                total += float(loss)
                count += shift_labels.numel()
        self._model.train()
        return {'loss/eval': total / max(count, 1)}

    def train(self, dataset=None, num_epochs: Optional[int] = None, **kwargs) -> Dict[str, float]:
        '''
        Run the stages in order, defaulting `num_epochs` to the sum of every stage's
        epoch count and the gradient clipping to `grad_clip_norm`.

        Args:
            dataset: Unused; the stages own their datasets.
            num_epochs (Optional[int]): Total number of epochs; None uses the sum of
                all stage epochs.
            **kwargs: Forwarded to `BasicPipeline.train`.

        Returns:
            Dict[str, float]: The final metric snapshot.
        '''
        if num_epochs is None:
            num_epochs = sum(s.epochs for s in self._stages)
        kwargs.setdefault('max_grad_norm', self.config.grad_clip_norm)
        return super().train(dataset, num_epochs=num_epochs, **kwargs)
