import os
import socket
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from motif.utils.sharding import row_slice, shard_batch, slice_rows

__all__ = [
    'DistributedConfig',
    'DistributedContext',
    'distributed_env',
    'find_free_port',
    'get_context',
    'worker_count',
    'setup_distributed',
    'wrap_model',
    'unwrap_model',
    'launch',
    'widen_global_batch_tokens',
    'row_slice',
    'slice_rows',
    'shard_batch',
]

#: Launch-environment variables, the same names `torchrun` uses.
ENV_RANK = 'RANK'
ENV_WORLD_SIZE = 'WORLD_SIZE'
ENV_LOCAL_RANK = 'LOCAL_RANK'
ENV_MASTER_ADDR = 'MASTER_ADDR'
ENV_MASTER_PORT = 'MASTER_PORT'


@dataclass
class DistributedConfig:
    '''
    Knobs of the automatic multi-GPU mode.

    `None` (the `distributed` field of a pipeline config) is the same as building this
    object with every default: the pipeline then joins whatever launch environment it
    finds, and `launch()` may spawn workers on a multi-GPU machine.

    Attributes:
        auto_spawn (bool): Let `launch()` start one worker process per visible GPU when
            no distributed environment exists. False means "run inline", which is the
            way to keep a multi-GPU box single-process on purpose.
        follow_env (bool): Join a process group that the launch environment already
            describes (`RANK` / `WORLD_SIZE`). False ignores those variables, which is
            only useful for debugging a script that is normally launched with
            `torchrun`.
        num_processes (Optional[int]): Number of workers `launch()` spawns; None uses
            one per visible GPU.
        backend (Optional[str]): Process-group backend; None picks 'nccl' on CUDA and
            'gloo' on the CPU. It also decides the kind of device: 'gloo' keeps the run
            on the CPU, 'nccl' implies CUDA.
        master_addr (str): Rendezvous address used by the auto-spawned workers.
        master_port (Optional[int]): Rendezvous port; None picks a free one, so two
            jobs on the same machine do not collide.
        timeout_minutes (float): Collective timeout; a rank that dies is then reported
            as a timeout instead of hanging the job forever.
        find_unused_parameters (bool): Let DDP search for parameters that did not
            receive a gradient. Needed only by a model whose forward skips branches;
            it costs a graph traversal per step, so a model that always uses every
            parameter should turn it off. It is **on** by default because
            `MotifChordModel` is such a model: `AttnResCache` starts empty, so the
            first decoder of a block has nothing to attend to, `apply_block_attn_res`
            returns its single source unchanged and that decoder's `pseudo_queries` /
            `attn_res_norms` never receive a gradient - DDP would then refuse the next
            iteration with "Expected to have finished reduction in the prior
            iteration".
        static_graph (bool): Promise that the forward graph never changes; lets DDP
            plan the buckets once. Only set it for a model whose forward really is
            static - the context curriculum changes the sequence length between
            stages, so it is off by default.
        broadcast_buffers (bool): Broadcast the model's registered buffers from rank 0
            before every forward - what DDP does by default, so the argument is only
            handed to DDP when it is switched off. Recent torch versions deprecate that
            argument in favour of `forward_sync_buffers`.
        bucket_cap_mb (int): Gradient bucket size of DDP.
        gradient_as_bucket_view (bool): Point `param.grad` straight at the DDP bucket.
            It saves a copy but can surprise code that keeps a gradient alive across
            steps, so it is off by default.
        shard_data (bool): Give every rank its own slice of the batch (True). False
            replicates the same batch on all ranks: correct, but it computes the same
            gradient N times and is only useful for debugging.
    '''

    auto_spawn: bool = True
    follow_env: bool = True
    num_processes: Optional[int] = None
    backend: Optional[str] = None
    master_addr: str = '127.0.0.1'
    master_port: Optional[int] = None
    timeout_minutes: float = 30.0
    find_unused_parameters: bool = True
    static_graph: bool = False
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    gradient_as_bucket_view: bool = False
    shard_data: bool = True


def _env_int(name: str, default: int) -> int:
    '''
    Read an integer environment variable.

    Args:
        name (str): Variable name.
        default (int): Value used when it is unset or not an integer.

    Returns:
        int: The parsed value.
    '''
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def distributed_env() -> Dict[str, int]:
    '''
    The rank layout the launch environment describes.

    Absent variables mean "a plain single-process run": rank 0, local rank 0, world
    size 1. `LOCAL_RANK` falls back to `RANK` because a plain `torchrun` sets both but
    a hand-written launcher often only sets `RANK`.

    Returns:
        Dict[str, int]: `{'rank': ..., 'world_size': ..., 'local_rank': ...}`.
    '''
    rank = _env_int(ENV_RANK, 0)
    return {
        'rank': rank,
        'world_size': max(1, _env_int(ENV_WORLD_SIZE, 1)),
        'local_rank': _env_int(ENV_LOCAL_RANK, rank),
    }


def find_free_port() -> int:
    '''
    Ask the OS for a free TCP port, for the rendezvous of auto-spawned workers.

    Returns:
        int: A port number that was free a moment ago.
    '''
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def unwrap_model(model: nn.Module) -> nn.Module:
    '''
    Peel off the `DistributedDataParallel` and `torch.compile` wrappers.

    Both wrappers add a prefix to every key of `state_dict()` and both rename the
    module they hold, so anything that saves weights or inspects a `forward` signature
    has to look through them: unwrapping centrally is what keeps a checkpoint written
    on 4 GPUs loadable on 1.

    Args:
        model (nn.Module): Possibly wrapped model.

    Returns:
        nn.Module: The innermost module that is neither a DDP nor a compiled wrapper.
    '''
    seen = set()
    while True:
        inner = getattr(model, '_orig_mod', None) or getattr(model, 'module', None)
        if inner is None or inner is model or id(inner) in seen:
            return model
        seen.add(id(inner))
        model = inner


def wrap_model(model: nn.Module, context: Optional['DistributedContext'] = None) -> nn.Module:
    '''
    Wrap a model in `DistributedDataParallel` when this rank is part of a group.

    The model must already be on its final device: DDP captures the parameter
    locations when it is built. On a single process the model is returned unchanged, so
    the single-GPU path never pays for the wrapper.

    Args:
        model (nn.Module): The model to replicate.
        context (Optional[DistributedContext]): Context to use; None takes the
            process-level one.

    Returns:
        nn.Module: The wrapped model, or `model` itself when nothing is distributed.
    '''
    context = context if context is not None else get_context()
    if not context.active:
        return model
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model

    on_cuda = context.device is not None and context.device.type == 'cuda'
    device_ids = [context.local_rank] if on_cuda else None
    kwargs = dict(
        device_ids=device_ids,
        output_device=context.local_rank if on_cuda else None,
        find_unused_parameters=context.config.find_unused_parameters,
        static_graph=context.config.static_graph,
        bucket_cap_mb=context.config.bucket_cap_mb,
        gradient_as_bucket_view=context.config.gradient_as_bucket_view,
    )
    if not context.config.broadcast_buffers:
        kwargs['broadcast_buffers'] = False
    return nn.parallel.DistributedDataParallel(model, **kwargs)


def widen_global_batch_tokens(
    global_batch_tokens: int,
    world_size: int,
    target_context: int,
) -> int:
    '''
    Smallest global batch that still gives every rank at least one row per step.

    A step splits the chunk of its stage into one contiguous slice per rank, so a stage
    whose batch holds fewer rows than there are ranks cannot be split. The binding
    stage is the longest one (its rows are the biggest), so requiring
    `world_size * (target_context + 1)` tokens per step is enough for every stage of
    the curriculum.

    Args:
        global_batch_tokens (int): Configured tokens per optimiser step.
        world_size (int): Number of ranks.
        target_context (int): Longest sequence length of the curriculum.

    Returns:
        int: `world_size * (target_context + 1)` when that exceeds the configured
        value, the configured value otherwise.
    '''
    if world_size <= 1:
        return int(global_batch_tokens)
    return max(int(global_batch_tokens), int(world_size) * (int(target_context) + 1))


class DistributedContext:
    '''
    The data-parallel view of this process: which rank it is, how many peers it has,
    which device it owns and how to talk to the others.

    Every method is a no-op while the context is inactive (a plain single-process
    run), so callers never have to branch on "am I distributed": a `world_size` of 1
    just means every collective is already done.

    The group is owned by whoever started it: `shutdown()` only destroys a group this
    object created itself, so a `torchrun`-managed group survives a pipeline that
    happens to be torn down first.

    One script, N GPUs: `launch(build)` runs `build(rank, world_size, local_rank)` once
    per worker process and spawns those processes itself when the machine has more than
    one GPU - so the caller never writes a `.sh` file, a `torchrun` command line or an
    `if rank == 0` branch. When only one GPU is visible (or `auto_spawn=False`) the very
    same `build` runs inline as rank 0, which keeps a laptop run and an 8-card run on
    the same code path:

        from motif.pipeline import DistributedConfig, launch

        def build(rank, world_size, local_rank):
            model = MotifChordModel(MotifChordConfig())
            pipe = PretrainPipeline(model, tokenizer, PretrainConfig(...),
                                    distributed=DistributedConfig())
            pipe.train(dataset)

        if __name__ == '__main__':          # required: spawn re-imports this module
            launch(build)

    `torchrun --nproc_per_node=8` works as well: `launch` notices the launch
    environment (`RANK` / `WORLD_SIZE` / `LOCAL_RANK`) and simply runs `build` in place
    on every rank it was given, so the process count can also be decided outside the
    script.

    What stays invariant: `global_batch_tokens` keeps meaning **the batch of one
    optimiser step over the whole world**, not per rank. Splitting it across ranks
    therefore changes neither the step count, nor the token budget, nor the
    learning-rate schedule: a 4-GPU run follows the same curriculum as a 1-GPU run and
    the two produce the same loss curve and (up to floating-point addition order) the
    same weights. What changes is only *who computes which rows*.

    How the data is split: every rank walks the **same** packed stream and takes a
    contiguous row slice of each chunk (`ChunkedTokenStream(shard_rank=...,
    shard_world=...)`), rather than every rank seeing a different part of the corpus.
    That keeps two things true at once:

    * the global batch of a step is exactly the chunk the plan asked for, and
    * each rank's resumable cursor (token buffer + upstream offset) is identical on all
      ranks, so the checkpoint is **world-size agnostic** - a run started on one GPU
      can be resumed on four and vice versa without skipping or replaying a single
      token.

    A chunk has to be divisible into at least one row per rank. `world_size *
    (target_context + 1)` is therefore the smallest usable global batch; when the
    configured `global_batch_tokens` is below that, `PretrainPipeline` widens it (and
    says so), which is the multi-GPU equivalent of "raise the global batch when you add
    cards".

    Why the loss is weighted by tokens: `F.cross_entropy(..., reduction='mean')`
    averages over the **unmasked** positions of one rank, and DDP averages the per-rank
    gradients with an equal weight, so a rank that happens to hold fewer unmasked tokens
    would silently count less. `BasicPipeline._scale_loss` multiplies each local loss by
    `world_size * local_tokens / global_tokens` before backward, which makes the DDP
    average *exactly* the single-process mean. When every rank holds the same number of
    tokens - the usual case, and always so with `world_size == 1` - the factor is 1 and
    nothing changes.

    Who does what: checkpoints, log files, TensorBoard, probes and weight
    snapshots are side effects, and rank 0 owns all of them (see
    `Callback.main_process_only`), otherwise N processes would race on the same path.
    The stop decision is the one thing that must not be taken independently - rank 0
    decides and the flag is broadcast, because a rank that leaves the loop while its
    peers are waiting inside an all-reduce hangs the job.

    Attributes:
        config (DistributedConfig): The knobs this context was started with.
        rank (int): Global rank of this process.
        local_rank (int): Index of the device this process owns on its node.
        world_size (int): Number of ranks in the group.
        device (torch.device): Device this rank trains on.
        backend (Optional[str]): Process-group backend, when one was started.
        active (bool): True when this process really is one of several ranks.
    '''

    def __init__(
        self,
        config: Optional[DistributedConfig] = None,
        device: Optional[Any] = None,
        env: Optional[Dict[str, int]] = None,
    ) -> None:
        '''
        Build a context, without starting a process group yet.

        Args:
            config (Optional[DistributedConfig]): Knobs; None uses the defaults.
            device (Optional[Any]): Device the pipeline asked for, if any.
            env (Optional[Dict[str, int]]): Launch layout; None reads the environment.
        '''
        self.config = config if isinstance(config, DistributedConfig) else DistributedConfig()
        layout = distributed_env() if env is None else env
        self.rank = int(layout['rank'])
        self.local_rank = int(layout['local_rank'])
        self.world_size = int(layout['world_size'])
        self.device: Optional[torch.device] = None if device is None else torch.device(device)
        self.backend: Optional[str] = None
        self.active = False
        self._owns_group = False

    def start(self, device: Optional[Any] = None) -> 'DistributedContext':
        '''
        Resolve the device and, when there is more than one rank, start the process
        group.

        A group that already exists (started by the user, or by an outer `launch`) is
        adopted instead of being started twice: it is the same rendezvous, and
        `torch.distributed` refuses a second `init_process_group` anyway.

        Args:
            device (Optional[Any]): Device requested by the pipeline; None lets this
                method choose `cuda:<local_rank>` or the CPU.

        Returns:
            DistributedContext: This context, for chaining.
        '''
        if device is not None:
            self.device = torch.device(device)

        if not self.config.follow_env:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self._resolve_device()
            return self

        if self.world_size <= 1:
            self._resolve_device()
            return self

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.backend = str(dist.get_backend())
            self.active = self.world_size > 1
            self._resolve_device()
            return self

        self._resolve_device()
        backend = self.config.backend or ('nccl' if self.device.type == 'cuda' else 'gloo')
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=float(self.config.timeout_minutes)),
        )
        self.backend = backend
        self.active = True
        self._owns_group = True
        dist.barrier()
        return self

    def _resolve_device(self) -> None:
        '''
        Pick the device of this rank.

        It is `cuda:<local_rank>` when the pipeline asked for the GPU (or left the
        choice to `resolve_device`, which answers CUDA on a CUDA machine) and this rank
        has a card, and the CPU otherwise. Two details matter:

          * an explicitly configured backend decides the kind of device as well:
            'gloo' is the CPU backend, so asking for it keeps the run on the CPU, and
            'nccl' implies CUDA. Every rank reads the same configuration, so that
            decision is symmetric;
          * a rank whose `local_rank` has no card behind it - more worker processes
            than GPUs, which an explicit `num_processes` or a `torchrun
            --nproc_per_node` larger than the device count produces - is a mistake
            rather than a fallback: silently moving that rank to the CPU would leave the
            others on `nccl` and hang the first collective. It raises instead.

        Raises:
            RuntimeError: If this rank's `local_rank` has no CUDA device while the run
                was not configured for the CPU.
        '''
        wants_cuda = self.device is None or self.device.type == 'cuda'
        available = torch.cuda.device_count() if torch.cuda.is_available() else 0
        cpu_backend = self.config.backend == 'gloo'

        if wants_cuda and not cpu_backend and available > 0:
            if self.local_rank >= available:
                raise RuntimeError(
                    f'rank {self.rank} has local_rank {self.local_rank}, but only '
                    f'{available} CUDA device(s) are visible. Running that rank on the '
                    f'CPU while the others use nccl would hang the first collective. '
                    f'Use at most {available} worker process(es) (that is what '
                    f'`launch()` does by default), or set '
                    f"DistributedConfig(backend='gloo') / device='cpu' to put every "
                    f'rank on the CPU.'
                )
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device('cuda', self.local_rank)
            return

        if self.device is None or self.device.type == 'cuda':
            self.device = torch.device('cpu')

    def shutdown(self) -> None:
        '''
        Destroy the process group, but only when this object started it.

        Called by `launch()` when the worker function is done; a pipeline does not call
        it, because several pipelines may share one context.
        '''
        if self._owns_group and dist.is_initialized():
            dist.destroy_process_group()
        self._owns_group = False
        self.active = False

    @property
    def is_main(self) -> bool:
        '''
        Returns:
            bool: True on rank 0, the rank that owns checkpoints and reporting.
        '''
        return self.rank == 0

    @property
    def shard_data(self) -> bool:
        '''
        Returns:
            bool: True when every rank should take its own slice of the batch.
        '''
        return bool(self.config.shard_data)

    def describe(self) -> str:
        '''
        Returns:
            str: One line describing this rank, for the sanity check.
        '''
        if not self.active:
            return 'off (single process)'
        return (f'rank {self.rank}/{self.world_size}, local {self.local_rank}, '
                f'backend {self.backend or dist.get_backend()}')

    def reduce_sum_(self, tensor: torch.Tensor) -> torch.Tensor:
        '''
        Sum a tensor over every rank, in place.

        Args:
            tensor (torch.Tensor): Tensor to reduce; unchanged when there is one rank.

        Returns:
            torch.Tensor: The same tensor, holding the sum.
        '''
        if not self.active or self.world_size <= 1:
            return tensor
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def reduce_mean_(self, tensor: torch.Tensor) -> torch.Tensor:
        '''
        Average a tensor over every rank, in place.

        `ReduceOp.AVG` is not implemented by every backend, so the sum is divided by
        the world size instead.

        Args:
            tensor (torch.Tensor): Tensor to reduce; unchanged when there is one rank.

        Returns:
            torch.Tensor: The same tensor, holding the mean.
        '''
        if not self.active or self.world_size <= 1:
            return tensor
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.div_(self.world_size)

    def broadcast_flag(self, flag: bool) -> bool:
        '''
        Agree on a boolean with every rank, with rank 0 as the source of truth.

        Stopping is the one decision that must not be local: a rank that leaves the
        training loop while its peers are inside a gradient all-reduce deadlocks the
        job, so rank 0 decides and everybody follows.

        Args:
            flag (bool): This rank's opinion; only rank 0's is kept.

        Returns:
            bool: The agreed value.
        '''
        if not self.active or self.world_size <= 1:
            return bool(flag)
        tensor = torch.tensor([1 if flag else 0], dtype=torch.int32, device=self.device)
        dist.broadcast(tensor, src=0)
        return bool(int(tensor.item()))

    def barrier(self) -> None:
        '''
        Wait until every rank has reached this point; a no-op when not distributed.
        '''
        if self.active and dist.is_initialized():
            dist.barrier()


#: The one context of this process; `get_context()` builds it on first use.
_CONTEXT: Optional[DistributedContext] = None


def get_context() -> DistributedContext:
    '''
    The process-level distributed context.

    There is one per process on purpose: the process group, the rank and the device
    are properties of the *process*, so two pipelines in the same process must share
    them, and a checkpoint written by one must be readable by the other.

    Returns:
        DistributedContext: The context, created (but not started) on first use.
    '''
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = DistributedContext()
    return _CONTEXT


def setup_distributed(
    config: Optional[DistributedConfig] = None,
    device: Optional[Any] = None,
) -> DistributedContext:
    '''
    Start (once per process) and return the distributed context.

    Idempotent: the first caller decides the configuration and the device, later
    callers get the same context back. That is what makes `PretrainPipeline(model, ...)`
    safe to construct under a launch even when the caller did not pass a
    `DistributedConfig`.

    Args:
        config (Optional[DistributedConfig]): Knobs; None uses the defaults.
        device (Optional[Any]): Device the pipeline resolved for itself.

    Returns:
        DistributedContext: The started context.
    '''
    global _CONTEXT
    if _CONTEXT is not None and _CONTEXT.active:
        return _CONTEXT
    context = DistributedContext(config=config, device=device)
    context.start()
    _CONTEXT = context
    return context


def worker_count(config: Optional[DistributedConfig] = None) -> int:
    '''
    How many worker processes an automatic launch should use: one per visible GPU.

    Args:
        config (Optional[DistributedConfig]): Knobs; `num_processes` overrides the
            device count.

    Returns:
        int: The process count, at least 1.
    '''
    config = config if isinstance(config, DistributedConfig) else DistributedConfig()
    if config.num_processes:
        return max(1, int(config.num_processes))
    if torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    return 1


def _spawn_entry(
    rank: int,
    build: Callable[..., Any],
    world_size: int,
    config: DistributedConfig,
    port: int,
    args: Tuple[Any, ...],
) -> None:
    '''
    Body of an auto-spawned worker: publish the launch layout, then run `build`.

    The child inherits the parent's environment, so the variables are written here
    rather than being passed as arguments - that way the worker looks exactly like a
    `torchrun` worker to everything downstream.

    Args:
        rank (int): Rank of this worker.
        build (Callable[..., Any]): The caller's worker function.
        world_size (int): Total number of workers.
        config (DistributedConfig): Launch configuration.
        port (int): Rendezvous port.
        args (Tuple[Any, ...]): Extra positional arguments for `build`.
    '''
    os.environ[ENV_RANK] = str(rank)
    os.environ[ENV_LOCAL_RANK] = str(rank)
    os.environ[ENV_WORLD_SIZE] = str(world_size)
    os.environ[ENV_MASTER_ADDR] = config.master_addr
    os.environ[ENV_MASTER_PORT] = str(port)
    try:
        build(rank, world_size, rank, *args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def launch(
    build: Callable[..., Any],
    config: Optional[DistributedConfig] = None,
    nprocs: Optional[int] = None,
    args: Tuple[Any, ...] = (),
) -> Any:
    '''
    Run `build(rank, world_size, local_rank, *args)` once per worker process.

    Three cases, decided automatically:

      1. a launch environment already exists (`RANK` / `WORLD_SIZE`, i.e. `torchrun`
         or an outer `launch`) -> `build` runs inline on this rank;
      2. several GPUs are visible and `config.auto_spawn` is on -> that many worker
         processes are spawned here, each with its own rank in the environment;
      3. otherwise -> `build(0, 1, 0, *args)` runs inline, which is the ordinary
         single-GPU / CPU path.

    On Windows (and wherever the `spawn` start method is used) the children re-import
    the file that called `launch`, so the call belongs under an
    `if __name__ == '__main__':` guard and `build` has to be a module-level function -
    a nested closure cannot be pickled.

    Args:
        build (Callable[..., Any]): Worker function taking
            `(rank, world_size, local_rank, *args)`; it must build the model, the
            tokenizer and the dataset itself, because each worker is a fresh process.
        config (Optional[DistributedConfig]): Launch configuration; None uses the
            defaults.
        nprocs (Optional[int]): Process count, overriding both the configuration and
            the visible device count (the way to exercise the multi-process path on a
            CPU-only machine).
        args (Tuple[Any, ...]): Extra positional arguments forwarded to `build`.

    Returns:
        Any: Whatever `build` returned when it ran inline; None when workers were
        spawned (the parent has no result to return).
    '''
    config = config if isinstance(config, DistributedConfig) else DistributedConfig()
    layout = distributed_env()

    if layout['world_size'] > 1 and config.follow_env:
        try:
            return build(layout['rank'], layout['world_size'], layout['local_rank'], *args)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    workers = worker_count(config) if nprocs is None else max(1, int(nprocs))
    if not config.auto_spawn or workers <= 1:
        return build(0, 1, 0, *args)

    port = int(config.master_port or find_free_port())
    os.environ.setdefault(ENV_MASTER_ADDR, config.master_addr)
    os.environ[ENV_MASTER_PORT] = str(port)
    print(f'[*] multi-process: spawning {workers} workers '
          f'({config.master_addr}:{port}); set DistributedConfig(auto_spawn=False) '
          f'or nprocs=1 to stay single-process.')
    torch.multiprocessing.spawn(
        _spawn_entry,
        args=(build, workers, config, port, args),
        nprocs=workers,
        join=True,
    )
    return None
