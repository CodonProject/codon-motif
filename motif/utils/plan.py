import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from motif.data.base import MotifDataset
from motif.data.chunked import ChunkedTokenStream


@dataclass
class TrainingStepsConfig:
    '''
    Token budget and step counts derived from the parameter count (the three presets
    min / recommended / overfit).

    Attributes:
        tokens_per_step (int): Tokens consumed by one optimiser step.
        model_size (int): Parameter count of the model.
        min_tokens (int): Token budget of the 'min' preset.
        recommended_tokens (int): Token budget of the 'recommended' preset.
        overtrain_tokens (int): Token budget of the 'overfit' preset.
        min_steps (int): Step count of the 'min' preset.
        recommended_steps (int): Step count of the 'recommended' preset.
        overfit_steps (int): Step count of the 'overfit' preset.
    '''
    tokens_per_step: int
    model_size: int

    min_tokens: int
    recommended_tokens: int
    overtrain_tokens: int

    min_steps: int
    recommended_steps: int
    overfit_steps: int


def calculate_training_steps(
    model_size: int,
    tokens_per_sample: int,
    batch_size: int,
    min_tpp: float = 20.0,
    rec_tpp: float = 80.0,
    overfit_tpp: float = 200.0,
) -> TrainingStepsConfig:
    '''
    Convert the three training budgets using Chinchilla-style tokens-per-parameter
    ratios.

    Args:
        model_size (int): Parameter count of the model.
        tokens_per_sample (int): Sequence length of one sample.
        batch_size (int): Batch size.
        min_tpp (float): Tokens per parameter of the 'min' preset.
        rec_tpp (float): Tokens per parameter of the 'recommended' preset.
        overfit_tpp (float): Tokens per parameter of the 'overfit' preset.

    Returns:
        TrainingStepsConfig: Token budgets and step counts of the three presets.
    '''
    tokens_per_step = tokens_per_sample * batch_size

    min_tokens = int(model_size * min_tpp)
    recommended_tokens = int(model_size * rec_tpp)
    overtrain_tokens = int(model_size * overfit_tpp)

    return TrainingStepsConfig(
        tokens_per_step=tokens_per_step,
        model_size=model_size,
        min_tokens=min_tokens,
        recommended_tokens=recommended_tokens,
        overtrain_tokens=overtrain_tokens,
        min_steps=math.ceil(min_tokens / tokens_per_step),
        recommended_steps=math.ceil(recommended_tokens / tokens_per_step),
        overfit_steps=math.ceil(overtrain_tokens / tokens_per_step),
    )


@dataclass
class ContextTrainingParams:
    '''
    Inputs of the curriculum weight calculation.

    Attributes:
        base_context (int): Context length of the foundation stage.
        total_token_budget (float): Total budget in billions of tokens.
        data_distribution_factor (float): How uniform the training data is, in [0, 1].
        performance_tradeoff_preference (float): Weight given to long-context
            performance, in [0, 1].
    '''

    base_context: int = 1024
    total_token_budget: float = 100.0
    data_distribution_factor: float = 0.5
    performance_tradeoff_preference: float = 0.5


def calculate_training_schedule(params: ContextTrainingParams, target_len: int) -> Dict[str, Any]:
    '''
    Split the total budget into foundation / expansion / stabilization weights.

    The foundation weight is fixed at 0.4; the expansion weight scales with the
    logarithmic growth of the context length and with the budget sensitivity; the
    stabilization weight is a blend of the long-context preference, the binary log of
    the target length, the data-distribution factor and the budget. All three are then
    normalised to sum to one.

    Args:
        params (ContextTrainingParams): Curriculum inputs.
        target_len (int): Context length the curriculum expands to.

    Returns:
        Dict[str, Any]: The input parameters, the target context length, the three
        stage descriptions and the normalised weights.
    '''
    foundation_weight = 0.4

    length_ratio = target_len / params.base_context
    log_ratio = math.log(length_ratio) if length_ratio > 1 else 0.0
    budget_sensitivity = 1.0 - (1.0 / (1.0 + params.total_token_budget / 50.0))

    expansion_weight = (log_ratio * 0.2) * budget_sensitivity

    stabilization_weight = (
        (params.performance_tradeoff_preference * 0.3) +
        (math.log2(max(target_len, 2)) / 14.0) * 0.3 +
        (params.data_distribution_factor * 0.2) +
        ((params.total_token_budget - 20.0) / 100.0) * 0.2
    )

    total_weight = foundation_weight + expansion_weight + stabilization_weight
    normalized = [
        foundation_weight / total_weight,
        expansion_weight / total_weight,
        max(0.0, stabilization_weight / total_weight),
    ]

    stages = [
        {
            'name': 'foundation',
            'context_len': str(params.base_context),
            'type': 'fixed',
            'allocated_tokens_B': params.total_token_budget * normalized[0],
        },
        {
            'name': 'expansion',
            'context_len': f'{params.base_context}-{target_len}',
            'type': 'dynamic_increase',
            'allocated_tokens_B': params.total_token_budget * normalized[1],
        },
        {
            'name': 'stabilization',
            'context_len': str(target_len),
            'type': 'fixed',
            'allocated_tokens_B': params.total_token_budget * normalized[2],
        },
    ]

    return {
        'input_parameters': params.__dict__,
        'target_context_length': target_len,
        'stages': stages,
        'weights_normalized': normalized,
    }


@dataclass
class Stage:
    '''
    One stage with a fixed context length.

    Attributes:
        name (str): Stage name, used in reports.
        seq_len (int): Sequence length fed to the model.
        chunk_len (int): Tokens per chunk, `batch_size * (seq_len + 1)`.
        batch_size (int): Batch size of this stage.
        tokens (int): Token budget allocated to this stage.
        steps (int): Number of optimiser steps this stage runs for.
    '''
    name: str
    seq_len: int
    chunk_len: int
    batch_size: int
    tokens: int
    steps: int

    def build_stream(
        self,
        data: Union[Iterable[Any], MotifDataset],
        eos_token_id: int,
        boundary_token_id: Optional[int] = None,
        mask_boundary: bool = True,
        shard_rank: int = 0,
        shard_world: int = 1,
    ) -> ChunkedTokenStream:
        '''
        Build the data flow of this stage (a Stateful object, so it can be resumed).

        Args:
            data (Union[Iterable[Any], MotifDataset]): Upstream corpus.
            eos_token_id (int): Fallback boundary token.
            boundary_token_id (Optional[int]): Document boundary token.
            mask_boundary (bool): Mask the boundary positions in the labels.
            shard_rank (int): Row slice this rank consumes (data parallelism).
            shard_world (int): Number of ranks the chunk is split over; 1 keeps it
                whole.

        Returns:
            ChunkedTokenStream: The packing stream of this stage.
        '''
        return ChunkedTokenStream(
            data=data,
            chunk_len=self.chunk_len,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            eos_token_id=eos_token_id,
            boundary_token_id=boundary_token_id,
            mask_boundary=mask_boundary,
            shard_rank=shard_rank,
            shard_world=shard_world,
        )


@dataclass
class TrainingPlan:
    '''
    The whole curriculum: total budget, total step count and the stage list.

    Attributes:
        total_tokens (int): Total token budget; the plan's stages add up to it, and it is
            what those stages really consume when only `total_steps` was given.
        total_steps (int): Total optimiser steps, i.e. the sum over the stages.
        stages (List[Stage]): The stages, in run order.
        step_mode (str): The budget the plan was built from: the `step_mode` preset, or
            `'custom'` (explicit token budget), `'steps'` (explicit step count) /
            `'full_data'` (the corpus itself) when those were given.
    '''
    total_tokens: int
    total_steps: int
    stages: List[Stage]
    step_mode: str

    def report_lines(self) -> List[str]:
        '''
        The curriculum report.

        Printing and log writing share this single text, so the two formats cannot
        drift apart.

        Returns:
            List[str]: The report lines.
        '''
        lines = [
            '=' * 70,
            f'Context Training Plan | Strategy: [{self.step_mode.upper()}]',
            '=' * 70,
            f'Total Budget: {self.total_tokens / 1e9:,.3f} B Tokens | '
            f'Total Steps: {self.total_steps:,}',
            '-' * 70,
        ]
        for i, s in enumerate(self.stages):
            lines.append(
                f'-> [{i + 1}] {s.name:<18} | Seq_{s.seq_len:<5} | '
                f'BS={s.batch_size:<4} | Tokens: {s.tokens / 1e9:>5.3f}B | '
                f'Steps: {s.steps:>8,}'
            )
        lines.append('=' * 70)
        return lines

    def print_report(self) -> 'TrainingPlan':
        '''
        Print the curriculum report.

        Returns:
            TrainingPlan: This plan, for chaining.
        '''
        for line in self.report_lines():
            print(line)
        return self


def _split_by_weight(weights: List[float], total: int) -> List[int]:
    '''
    Split `total` over `weights` so the parts sum to exactly `total`.

    Largest-remainder apportionment: every part takes the floor of its exact share and
    the steps left over go to the parts with the largest fractions. Rounding therefore
    never loses or invents a step, and no stage is silently handed the whole error.

    Args:
        weights (List[float]): Relative weight of every part (non-negative).
        total (int): Number to split.

    Returns:
        List[int]: One part per weight, summing to `total`.

    Raises:
        ValueError: If `total` is smaller than the number of parts, so at least one
            part would be empty.
    '''
    if total < len(weights):
        raise ValueError(
            f'{total} steps cannot be split over {len(weights)} stages: every stage needs '
            f'at least one step.'
        )
    total_weight = sum(max(0.0, w) for w in weights)
    if total_weight <= 0:
        exact = [total / len(weights)] * len(weights)
    else:
        exact = [total * max(0.0, w) / total_weight for w in weights]
    parts = [int(math.floor(x)) for x in exact]
    leftover = total - sum(parts)
    order = sorted(
        range(len(weights)), key=lambda i: (exact[i] - parts[i], -i), reverse=True
    )
    for i in order[:leftover]:
        parts[i] += 1
    return parts


class ContextTrainingPlanner:
    '''
    Build a `TrainingPlan` from the model size and the context target.

    The token budget comes from the parameter count and `step_mode`, or from an
    explicit `total_tokens` - and, when `total_steps` is given instead, the horizon is
    a step count and the budget only shapes the stage weights. Whichever way it is
    derived, the split across Foundation / Expansion_* / Stabilization stays the same.

    Attributes:
        model: Model whose parameter count sizes the budget.
        step_mode (str): Budget preset, one of 'min' / 'recommended' / 'overfit'.
        base_context (int): Context length of the foundation stage.
        target_context (int): Context length the curriculum expands to.
        global_batch_tokens (int): Tokens consumed per optimiser step.
        total_tokens (Optional[int]): Explicit token budget; None derives it from
            `step_mode` and the parameter count.
        total_steps (Optional[int]): Explicit step count. It wins over the token budget:
            the weights shape how the steps are split, and the step count is then exact.
        budget_source (Optional[str]): Name the budget carries in the report; None takes
            it from `total_steps` / `total_tokens` / `step_mode`.
    '''

    def __init__(
        self,
        model,
        step_mode: str = 'recommended',
        base_context: int = 512,
        target_context: int = 8192,
        global_batch_tokens: int = 8192 * 2,
        total_tokens: Optional[int] = None,
        total_steps: Optional[int] = None,
        budget_source: Optional[str] = None,
    ):
        '''
        Configure the planner.

        Args:
            model: Model whose parameter count sizes the budget.
            step_mode (str): Budget preset.
            base_context (int): Context length of the foundation stage.
            target_context (int): Context length the curriculum expands to.
            global_batch_tokens (int): Tokens consumed per optimiser step.
            total_tokens (Optional[int]): Explicit token budget; None uses `step_mode`.
            total_steps (Optional[int]): Explicit step count; None derives it from the
                token budget.
            budget_source (Optional[str]): Label for the report; None derives it.

        Raises:
            ValueError: If `step_mode` is unknown, if `base_context` is not positive,
                if `target_context` is smaller than `base_context`, or if a given
                `total_tokens` / `total_steps` is not positive.
        '''
        if step_mode not in ('min', 'recommended', 'overfit'):
            raise ValueError(f"step_mode must be 'min' / 'recommended' / 'overfit', got {step_mode!r}")
        if base_context <= 0 or target_context < base_context:
            raise ValueError(f'invalid context range: base={base_context}, target={target_context}')
        if total_tokens is not None and total_tokens <= 0:
            raise ValueError(f'total_tokens must be positive, got {total_tokens!r}')
        if total_steps is not None and total_steps <= 0:
            raise ValueError(f'total_steps must be positive, got {total_steps!r}')
        self.model = model
        self.step_mode = step_mode
        self.base_context = base_context
        self.target_context = target_context
        self.global_batch_tokens = global_batch_tokens
        self.total_tokens = None if total_tokens is None else int(total_tokens)
        self.total_steps = None if total_steps is None else int(total_steps)
        self.budget_source = budget_source

    def _expansion_lengths(self) -> List[int]:
        '''
        Context lengths of the doubling expansion stages, i.e. every power-of-two
        multiple of `base_context` that is still below `target_context`.

        Returns:
            List[int]: The expansion context lengths.
        '''
        lengths = []
        cur = self.base_context * 2
        while cur < self.target_context:
            lengths.append(cur)
            cur *= 2
        return lengths

    def _create_stage(self, name: str, seq_len: int, target_tokens: int,
                      steps: Optional[int] = None) -> Stage:
        '''
        Create one stage, deriving the batch size from `global_batch_tokens`.

        One extra token per sample is reserved for the shifted label, so the effective
        length is `seq_len + 1`.

        Args:
            name (str): Stage name.
            seq_len (int): Sequence length of the stage.
            target_tokens (int): Token budget of the stage.
            steps (Optional[int]): Step count of the stage; None derives it from
                `target_tokens`. A given count wins, and the token figure reported for
                the stage becomes what those steps actually consume.

        Returns:
            Stage: The stage, with its step count already computed.
        '''
        effective_len = seq_len + 1
        batch_size = max(1, self.global_batch_tokens // effective_len)
        chunk_len = batch_size * effective_len
        if steps is None:
            tokens = target_tokens
            steps = math.ceil(target_tokens / chunk_len)
        else:
            steps = int(steps)
            tokens = steps * chunk_len
        return Stage(
            name=name,
            seq_len=seq_len,
            batch_size=batch_size,
            chunk_len=chunk_len,
            tokens=tokens,
            steps=steps,
        )

    def generate_plan(self) -> TrainingPlan:
        '''
        Build the plan: budget -> stage weights -> stages.

        The budget is `total_tokens` when it was given, otherwise the `step_mode` preset
        of the parameter count. Only the weights of the three stage kinds depend on it.
        With `total_steps` the split happens on the step counts instead: `_split_by_weight`
        hands the requested number of steps out along the same weights, and the token
        budget is only reported.

        Returns:
            TrainingPlan: The finished curriculum, whose `step_mode` names the budget it
            was built from (`'custom'` / `'steps'` / the preset).

        Raises:
            ValueError: If `total_steps` is smaller than the number of stages.
        '''
        from motif.utils.units import count_params

        param_count = count_params(self.model)

        base_bs = max(1, self.global_batch_tokens // self.base_context)
        data = calculate_training_steps(param_count, self.base_context, base_bs)

        if self.total_tokens is not None:
            total_tokens = self.total_tokens
            mode = 'custom'
        else:
            mode = self.step_mode
            if self.step_mode == 'min':
                total_tokens = data.min_tokens
            elif self.step_mode == 'overfit':
                total_tokens = data.overtrain_tokens
            else:
                total_tokens = data.recommended_tokens
        if self.total_steps is not None:
            mode = 'steps'
        if self.budget_source is not None:
            mode = self.budget_source

        schedule = calculate_training_schedule(
            ContextTrainingParams(
                base_context=self.base_context,
                total_token_budget=total_tokens / 1e9,
            ),
            target_len=self.target_context,
        )
        info = schedule['stages']

        expansion_lengths = self._expansion_lengths()
        names = ['Foundation'] + [f'Expansion_{c}' for c in expansion_lengths] + ['Stabilization']
        lengths = [self.base_context] + expansion_lengths + [self.target_context]

        targets = [int(info[0]['allocated_tokens_B'] * 1e9)]
        weights = [info[0]['allocated_tokens_B']]
        if expansion_lengths:
            per_stage = int(info[1]['allocated_tokens_B'] * 1e9) // len(expansion_lengths)
            targets += [per_stage] * len(expansion_lengths)
            weights += [info[1]['allocated_tokens_B'] / len(expansion_lengths)] * len(expansion_lengths)
        targets.append(int(info[2]['allocated_tokens_B'] * 1e9))
        weights.append(info[2]['allocated_tokens_B'])
        targets[-1] += total_tokens - sum(targets)

        if self.total_steps is None:
            stages = [
                self._create_stage(name, seq_len, target)
                for name, seq_len, target in zip(names, lengths, targets)
            ]
        else:
            step_counts = _split_by_weight(weights, self.total_steps)
            stages = [
                self._create_stage(name, seq_len, 0, steps=steps)
                for name, seq_len, steps in zip(names, lengths, step_counts)
            ]
            if self.total_tokens is None:
                total_tokens = sum(s.tokens for s in stages)

        return TrainingPlan(
            total_tokens=total_tokens,
            total_steps=sum(s.steps for s in stages),
            stages=stages,
            step_mode=mode,
        )


class StatefulPlanRunner:
    '''
    Yield `(stage, input_ids, labels)` stage by stage, following the plan.

    `state_dict()` holds `stage_idx` / `step_in_stage` plus the **data-flow state of
    the current stage** (token buffer and upstream cursor), so after a restore:

      * it does not start over from Foundation;
      * it does not throw away a token buffer that was half full inside a stage;
      * it does not tokenize the corpus before the checkpoint a second time.

    Data parallelism narrows every chunk to one contiguous row slice per rank
    (`shard_rank` / `shard_world`). Because that split is a pure function of the chunk
    while the buffer and cursor advance identically on every rank, this state stays
    **world-size agnostic**: a run started on one rank resumes on four and vice versa.

    Attributes:
        plan (TrainingPlan): The plan being executed.
        data: Upstream corpus (passed to `Stage.build_stream`).
        eos_token_id (int): EOS used for sampling / stopping.
        boundary_token_id (Optional[int]): Document boundary token; None falls back to
            `eos_token_id`.
        mask_boundary (bool): True = write -100 at the boundary-token positions of the
            labels (no loss on them).
        shard_rank (int): Row slice this rank consumes.
        shard_world (int): Number of ranks the chunk is split over; 1 keeps it whole.
        stage_idx (int): Index of the stage being executed.
        step_in_stage (int): Step already produced inside the current stage.
    '''

    def __init__(
        self,
        plan: TrainingPlan,
        data: Any,
        eos_token_id: int,
        boundary_token_id: Optional[int] = None,
        mask_boundary: bool = True,
        shard_rank: int = 0,
        shard_world: int = 1,
    ) -> None:
        '''
        Build the executor at the beginning of the plan.

        Args:
            plan (TrainingPlan): The plan to execute.
            data (Any): Upstream corpus.
            eos_token_id (int): Fallback boundary token.
            boundary_token_id (Optional[int]): Document boundary token.
            mask_boundary (bool): Mask the boundary positions in the labels.
            shard_rank (int): Row slice this rank consumes (data parallelism).
            shard_world (int): Number of ranks the chunk is split over.
        '''
        self.plan = plan
        self.data = data
        self.eos_token_id = eos_token_id
        self.boundary_token_id = boundary_token_id
        self.mask_boundary = bool(mask_boundary)
        self.shard_rank = int(shard_rank)
        self.shard_world = int(shard_world)

        self.stage_idx = 0
        self.step_in_stage = 0

        self._stream: Optional[ChunkedTokenStream] = None
        self._iterator: Optional[Iterator[Any]] = None

    def state_dict(self) -> Dict[str, Any]:
        '''
        Returns:
            Dict[str, Any]: The stage position, the step inside it and, when a stage
            is open, the data-flow state of that stage.
        '''
        state: Dict[str, Any] = {
            'stage_idx': int(self.stage_idx),
            'step_in_stage': int(self.step_in_stage),
        }
        if self._stream is not None:
            state['stream'] = self._stream.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        '''
        Restore the stage position and the stage-local data flow.

        Args:
            state (Dict[str, Any]): State produced by `state_dict()`.
        '''
        self.stage_idx = int(state.get('stage_idx', 0))
        self.step_in_stage = int(state.get('step_in_stage', 0))
        self._stream = None
        self._iterator = None

        if self.stage_idx >= len(self.plan.stages):
            return

        self._open_stage()
        if state.get('stream') is not None:
            self._stream.load_state_dict(state['stream'])
            self._iterator = iter(self._stream)

    def _open_stage(self) -> None:
        '''
        Build the packing stream of the current stage and start iterating it.
        '''
        stage = self.plan.stages[self.stage_idx]
        self._stream = stage.build_stream(
            self.data, self.eos_token_id, self.boundary_token_id, self.mask_boundary,
            shard_rank=self.shard_rank, shard_world=self.shard_world,
        )
        self._iterator = iter(self._stream)

    def __iter__(self) -> Iterator[Any]:
        '''
        Yield `(stage, input_ids, labels)` until the plan is exhausted.

        Yields:
            tuple: The current `Stage`, the input ids and the labels of one chunk.

        Raises:
            RuntimeError: If the corpus runs out twice in a row before the stage has
                produced all of its steps, which means the corpus is too small or the
                `seq_len` / `batch_size` configuration is wrong.
        '''
        while self.stage_idx < len(self.plan.stages):
            if self._stream is None:
                self._open_stage()
            stage = self.plan.stages[self.stage_idx]

            drained = 0
            while self.step_in_stage < stage.steps:
                try:
                    inputs, labels = next(self._iterator)
                except StopIteration:
                    drained += 1
                    if drained > 1:
                        raise RuntimeError(
                            f'stage {stage.name} needs {stage.steps} steps, but the corpus '
                            f'has been walked twice and still yields no data: the corpus '
                            f'is too small, or seq_len / batch_size is misconfigured.'
                        )
                    self._open_stage()
                    continue

                drained = 0
                self.step_in_stage += 1
                yield stage, inputs, labels

            self.stage_idx += 1
            self.step_in_stage = 0
            self._stream = None
            self._iterator = None
