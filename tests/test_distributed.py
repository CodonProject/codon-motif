
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from motif.pipeline import (
    Callback,
    DistributedConfig,
    DistributedContext,
    PretrainConfig,
    PretrainPipeline,
    launch,
    row_slice,
    shard_batch,
    slice_rows,
    unwrap_model,
    widen_global_batch_tokens,
    worker_count,
)
from motif.pipeline.base import forward_accepted_kwargs
from motif.utils.plan import Stage, StatefulPlanRunner, TrainingPlan

from support import ListDataset

DIST_AVAILABLE = dist.is_available()
requires_dist = pytest.mark.skipif(not DIST_AVAILABLE, reason='torch.distributed is unavailable')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(ROOT, 'tests', '_dist_worker.py')
LAUNCHER = os.path.join(ROOT, 'tests', '_dist_launch.py')
TIMEOUT = 900

#: Directories created when a test runs without pytest's `tmp_path` fixture.
_SCRATCH = []


def scratch(tmp_path=None) -> Path:
    '''
    A writable directory, from pytest when there is one.

    Args:
        tmp_path (Optional[Path]): The pytest fixture, when the test was called by
            pytest; None creates a temporary directory instead (so the file can also
            be run directly, like the other tests here).

    Returns:
        Path: The directory to use.
    '''
    if tmp_path is not None:
        return Path(tmp_path)
    directory = Path(tempfile.mkdtemp(prefix='motif_dist_'))
    _SCRATCH.append(directory)
    return directory


class Recorder(Callback):
    '''Records how often each hook ran (used to check the rank-0-only rule).'''

    def __init__(self):
        self.calls = 0

    def on_step_end(self, pipeline, step_metrics):
        self.calls += 1


class FakeParallel(nn.Module):
    '''Stand-in for `DistributedDataParallel`: a module holding another module.'''

    def __init__(self, module):
        super().__init__()
        self.module = module


def test_row_slices_cover_every_row_exactly_once():
    for total in range(1, 9):
        for world in range(1, 5):
            seen = []
            for rank in range(world):
                start, end = row_slice(total, rank, world)
                assert 0 <= start <= end <= total
                seen.extend(range(start, end))
            assert seen == list(range(total)), f'{total} rows over {world} ranks'


def test_row_slices_are_contiguous_and_even_by_one():
    assert row_slice(5, 0, 2) == (0, 3)
    assert row_slice(5, 1, 2) == (3, 5)
    assert row_slice(4, 0, 2) == (0, 2)
    assert row_slice(4, 1, 2) == (2, 4)
    assert row_slice(7, 3, 1) == (0, 7)


def test_widen_global_batch_tokens():
    assert widen_global_batch_tokens(8192, 1, 4096) == 8192
    assert widen_global_batch_tokens(8192, 4, 4096) == 4 * 4097
    assert widen_global_batch_tokens(4 * 4097, 4, 4096) == 4 * 4097
    assert widen_global_batch_tokens(100000, 8, 512) == 100000


def test_widened_batch_gives_every_rank_a_row_on_every_stage():
    world, target = 8, 4096
    tokens = widen_global_batch_tokens(8192, world, target)
    for seq_len in (512, 1024, 2048, 4096):
        batch_size = tokens // (seq_len + 1)
        assert batch_size >= world, f'seq {seq_len} would leave a rank empty'


def test_shard_batch_slices_rows_and_leaves_auxiliary_fields_alone():
    batch = {
        'input_ids': torch.arange(20).view(5, 4),
        'labels': torch.arange(20).view(5, 4),
        'meta': 'not-a-tensor',
        'images': [torch.ones(3, 2) for _ in range(5)],
        'flat': [torch.ones(2) for _ in range(3)],
    }
    first, second = shard_batch(batch, 0, 2), shard_batch(batch, 1, 2)

    assert first['input_ids'].shape == (3, 4)
    assert second['input_ids'].shape == (2, 4)
    merged = torch.cat([first['input_ids'], second['input_ids']])
    assert torch.equal(merged, batch['input_ids'])

    assert len(first['images']) == 3 and len(second['images']) == 2
    assert len(first['flat']) == 3 and len(second['flat']) == 3
    assert first['meta'] == second['meta'] == 'not-a-tensor'


def test_slice_rows_leaves_tensors_that_are_not_per_row_alone():
    assert slice_rows(torch.ones(7, 2), 0, 1, 5).shape == (7, 2)
    assert slice_rows(torch.ones(5, 2), 0, 1, 5).shape == (1, 2)
    assert slice_rows(torch.tensor(3.0), 0, 1, 5).shape == ()


def _records():
    return [[(index * 7 + offset) % 40 + 4 for offset in range(1 + index % 6)]
            for index in range(40)]


def _stream(rank, world, batch_size=3, seq_len=3):
    from motif.data import ChunkedTokenStream

    return ChunkedTokenStream(
        ListDataset(_records()),
        chunk_len=batch_size * (seq_len + 1),
        batch_size=batch_size,
        seq_len=seq_len,
        eos_token_id=2,
        boundary_token_id=3,
        shard_rank=rank,
        shard_world=world,
    )


def test_sharded_streams_reassemble_into_the_unsharded_stream():
    full = list(_stream(0, 1, batch_size=5))
    assert len(full) > 3, 'corpus too small to test the split'

    shards = [list(_stream(rank, 2, batch_size=5)) for rank in range(2)]
    assert [len(shard) for shard in shards] == [len(full), len(full)]

    for index, (inputs, labels) in enumerate(full):
        got_inputs = torch.cat([shards[0][index][0], shards[1][index][0]])
        got_labels = torch.cat([shards[0][index][1], shards[1][index][1]])
        assert torch.equal(got_inputs, inputs), 'rank slices do not cover the chunk'
        assert torch.equal(got_labels, labels), 'rank slices misalign the labels'
        assert shards[0][index][0].shape[0] == 3
        assert shards[1][index][0].shape[0] == 2


def test_sharded_stream_state_is_identical_on_every_rank():
    streams = [_stream(rank, 2) for rank in range(2)]
    iterators = [iter(stream) for stream in streams]
    for _ in range(4):
        for iterator in iterators:
            next(iterator)
    assert streams[0].state_dict() == streams[1].state_dict()


def test_chunk_sharding_refuses_to_leave_a_rank_empty():
    from motif.data import ChunkedTokenStream

    try:
        ChunkedTokenStream(ListDataset(_records()), 2 * 4, 2, 3, eos_token_id=2,
                           shard_rank=0, shard_world=3)
    except ValueError as exc:
        assert 'empty chunk' in str(exc)
    else:
        raise AssertionError('an unsplittable chunk was accepted')


def test_plan_runner_step_count_does_not_depend_on_the_world_size():
    plan = TrainingPlan(total_tokens=0, total_steps=6, step_mode='test', stages=[
        Stage(name='Foundation', seq_len=3, batch_size=6, chunk_len=6 * 4, tokens=0, steps=4),
        Stage(name='Stabilization', seq_len=4, batch_size=4, chunk_len=4 * 5, tokens=0, steps=2),
    ])
    reference = list(StatefulPlanRunner(plan, ListDataset(_records()), 2, boundary_token_id=3))

    for world in (2, 3):
        per_rank = [list(StatefulPlanRunner(
            plan, ListDataset(_records()), 2, boundary_token_id=3,
            shard_rank=rank, shard_world=world,
        )) for rank in range(world)]
        assert all(len(rank_steps) == len(reference) for rank_steps in per_rank)
        for step_index, expected in enumerate(reference):
            inputs = torch.cat([rank_steps[step_index][1] for rank_steps in per_rank])
            labels = torch.cat([rank_steps[step_index][2] for rank_steps in per_rank])
            assert torch.equal(inputs, expected[1]), f'world {world}: rows lost'
            assert torch.equal(labels, expected[2]), f'world {world}: labels misaligned'
            rows = [rank_steps[step_index][1].shape[0] for rank_steps in per_rank]
            assert sum(rows) == expected[1].shape[0]


def test_unwrap_model_peels_ddp_and_compile_wrappers():
    model = nn.Linear(4, 2)
    assert unwrap_model(FakeParallel(model)) is model

    compiled = nn.Module()
    compiled._orig_mod = model
    assert unwrap_model(compiled) is model
    assert unwrap_model(FakeParallel(compiled)) is model


def test_forward_signature_is_read_through_the_ddp_wrapper():
    class Model(nn.Module):
        def forward(self, input_ids, mask=None):
            return input_ids

    plain = forward_accepted_kwargs(Model())
    wrapped = forward_accepted_kwargs(FakeParallel(Model()))
    assert plain == wrapped == {'input_ids', 'mask'}


def test_single_process_context_is_inert():
    config = DistributedConfig()
    context = DistributedContext(config, device='cpu', env={'rank': 0, 'world_size': 1, 'local_rank': 0})
    context.start()

    assert not context.active
    assert context.is_main and context.world_size == 1
    assert context.describe() == 'off (single process)'

    tensor = torch.tensor([2.0])
    assert context.reduce_sum_(tensor) is tensor and float(tensor) == 2.0
    assert float(context.reduce_mean_(tensor)) == 2.0
    assert context.broadcast_flag(True) is True
    context.barrier()
    context.shutdown()


def test_follow_env_false_ignores_a_launch_environment():
    context = DistributedContext(
        DistributedConfig(follow_env=False), device='cpu',
        env={'rank': 3, 'world_size': 4, 'local_rank': 3},
    )
    context.start()
    assert not context.active and context.rank == 0 and context.world_size == 1


def _pipeline(ckpt_dir, callbacks=None):
    return PretrainPipeline(
        nn.Linear(1, 1), None,
        PretrainConfig(
            base_context=4, target_context=8, global_batch_tokens=16, step_mode='min',
            ckpt_dir=ckpt_dir, use_progress=False, amp=False, probe_every_steps=0,
            eos_token_id=2, boundary_token_id=3,
        ),
        device='cpu', callbacks=callbacks,
    )


def test_pipeline_is_single_process_by_default(tmp_path=None):
    base = scratch(tmp_path)
    pipeline = _pipeline(str(base / 'ckpt'), callbacks=[Recorder()])
    assert not pipeline.distributed
    assert pipeline.rank == 0 and pipeline.local_rank == 0 and pipeline.world_size == 1
    assert pipeline.is_main
    assert pipeline.shard_config() == (0, 1)
    assert any('distributed : off (single process)' in line for line in pipeline.describe())

    loss = torch.tensor(1.5, requires_grad=True)
    assert pipeline._scale_loss(loss, {'tokens': 8.0}) is loss


def test_rank_zero_owns_checkpoints_and_callbacks(tmp_path=None):
    base = scratch(tmp_path)
    recorder = Recorder()
    pipeline = _pipeline(str(base / 'ckpt'), callbacks=[recorder])
    fake = DistributedContext(
        DistributedConfig(), device='cpu', env={'rank': 1, 'world_size': 2, 'local_rank': 1}
    )
    fake.active = True
    pipeline._dist = fake

    assert not pipeline.is_main
    assert pipeline.shard_config() == (1, 2)

    path = str(base / 'ckpt' / 'last.pt')
    pipeline.save_checkpoint(path)
    assert not os.path.exists(path), 'a non-main rank wrote a checkpoint'

    pipeline._callback('on_step_end', {})
    assert recorder.calls == 0, 'a side-effect callback ran on a non-main rank'


def test_main_process_only_can_be_switched_off_for_a_participating_callback(tmp_path=None):
    base = scratch(tmp_path)

    class Participating(Recorder):
        main_process_only = False

    recorder = Participating()
    pipeline = _pipeline(str(base / 'ckpt'), callbacks=[recorder])
    fake = DistributedContext(
        DistributedConfig(), device='cpu', env={'rank': 1, 'world_size': 2, 'local_rank': 1}
    )
    fake.active = True
    pipeline._dist = fake
    pipeline._callback('on_step_end', {})
    assert recorder.calls == 1


class FakeGroup(DistributedContext):
    '''A context that stands in for one peer, to test the loss weighting maths.'''

    peer_tokens = 0.0

    def reduce_sum_(self, tensor):
        tensor += self.peer_tokens
        return tensor


def test_token_weighted_loss_reproduces_the_global_mean(tmp_path=None):
    '''
    The DDP average of the two ranks' weighted losses has to be the loss the same
    three positions would have produced in one batch - and the *plain* average of the
    two partial means must not be, otherwise the test would not say anything.
    '''
    import torch.nn.functional as F

    base = scratch(tmp_path)
    pipeline = _pipeline(str(base / 'ckpt'))
    group = FakeGroup(DistributedConfig(), device='cpu',
                      env={'rank': 0, 'world_size': 2, 'local_rank': 0})
    group.active = True
    pipeline._dist = group

    torch.manual_seed(0)
    logits = torch.randn(3, 7, requires_grad=True)
    labels = torch.randint(0, 7, (3,))

    full = F.cross_entropy(logits, labels)
    first = F.cross_entropy(logits[:2], labels[:2])
    second = F.cross_entropy(logits[2:], labels[2:])

    group.peer_tokens = 1.0
    scaled_first = pipeline._scale_loss(first, {'tokens': 2.0})
    group.peer_tokens = 2.0
    scaled_second = pipeline._scale_loss(second, {'tokens': 1.0})

    averaged = (scaled_first + scaled_second) / 2
    assert torch.allclose(averaged, full, atol=1e-6), 'the weighted average is not the global mean'
    assert not torch.allclose((first + second) / 2, full, atol=1e-4), \
        'the plain average already matches, so the test data proves nothing'


def test_a_single_rank_leaves_the_loss_alone(tmp_path=None):
    base = scratch(tmp_path)
    pipeline = _pipeline(str(base / 'ckpt'))
    loss = torch.tensor(2.0, requires_grad=True)
    assert pipeline._scale_loss(loss, {'tokens': 5.0}) is loss


def test_backend_gloo_keeps_the_run_on_the_cpu():
    context = DistributedContext(
        DistributedConfig(backend='gloo'), env={'rank': 0, 'world_size': 2, 'local_rank': 0}
    )
    context._resolve_device()
    assert context.device.type == 'cpu'


@pytest.mark.skipif(torch.cuda.device_count() == 0,
                    reason='needs one visible GPU that this local rank can be pushed past')
def test_local_rank_without_a_card_is_refused():
    '''
    More worker processes than cards is a mistake, not a fallback: putting that one rank
    on the CPU while its peers use nccl would hang the first collective.
    '''
    context = DistributedContext(
        DistributedConfig(), env={'rank': 1, 'world_size': 2, 'local_rank': 1}
    )
    with pytest.raises(RuntimeError, match='local_rank'):
        context._resolve_device()

    cpu_context = DistributedContext(
        DistributedConfig(backend='gloo'), env={'rank': 1, 'world_size': 2, 'local_rank': 1}
    )
    cpu_context._resolve_device()
    assert cpu_context.device.type == 'cpu'


def test_launch_runs_inline_when_the_environment_already_describes_the_world(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '3')
    monkeypatch.setenv('RANK', '2')
    monkeypatch.setenv('LOCAL_RANK', '2')
    seen = []

    result = launch(lambda rank, world, local: seen.append((rank, world, local)) or 'done')

    assert seen == [(2, 3, 2)]
    assert result == 'done'


def test_launch_runs_inline_when_auto_spawn_is_off(monkeypatch):
    for name in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK'):
        monkeypatch.delenv(name, raising=False)
    seen = []
    launch(lambda rank, world, local: seen.append((rank, world, local)),
           DistributedConfig(auto_spawn=False))
    assert seen == [(0, 1, 0)]


def test_launch_runs_inline_for_a_single_worker(monkeypatch):
    for name in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK'):
        monkeypatch.delenv(name, raising=False)
    seen = []
    launch(lambda rank, world, local: seen.append((rank, world, local)),
           DistributedConfig(), nprocs=1)
    assert seen == [(0, 1, 0)]


def test_worker_count_prefers_the_configured_process_count():
    assert worker_count(DistributedConfig(num_processes=3)) == 3
    assert worker_count(DistributedConfig()) >= 1


def _run_world(world, worker_args):
    '''
    Start `world` copies of the comparison worker with a `torchrun`-style environment.

    Args:
        world (int): Number of ranks.
        worker_args (list): Arguments for the worker script.

    Returns:
        list: `(returncode, stdout, stderr)` per rank.
    '''
    from motif.pipeline import find_free_port

    base_env = dict(os.environ)
    base_env['MASTER_ADDR'] = '127.0.0.1'
    base_env['MASTER_PORT'] = str(find_free_port())
    base_env['WORLD_SIZE'] = str(world)

    processes = []
    for rank in range(world):
        env = dict(base_env, RANK=str(rank), LOCAL_RANK=str(rank))
        processes.append(subprocess.Popen(
            [sys.executable, WORKER, *worker_args],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ))

    results = []
    for rank, process in enumerate(processes):
        try:
            out, err = process.communicate(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            out, err = process.communicate()
            raise AssertionError(f'rank {rank} timed out (a collective is probably '
                                 f'waiting for a peer)\n{out}\n{err}')
        results.append((process.returncode, out, err))

    for rank, (code, out, err) in enumerate(results):
        assert code == 0, f'rank {rank} exited with {code}\nstdout:\n{out}\nstderr:\n{err}'
    return results


def _assert_same_gradients(got, want, label):
    '''
    Two gradients of the same optimiser step, taken at the same weights, must be
    identical up to the order of the floating-point additions.

    This is the assertion that actually pins the data parallelism down: at a shared
    starting point a wrong split, a missing token weight or a dropped rank would move a
    gradient by a fraction of its own size, while reordering the reductions moves it by
    ~1e-7 of it.
    '''
    assert set(want) == set(got), f'{label}: different parameter names'
    unused = [key for key in want if (want[key] is None) != (got[key] is None)]
    assert not unused, f'{label}: parameters used by one run only: {unused}'

    worst, worst_key = 0.0, None
    for key, expected in want.items():
        if expected is None:
            continue
        difference = float((got[key].float() - expected.float()).abs().max())
        if difference > worst:
            worst, worst_key = difference, key
        assert torch.allclose(got[key].float(), expected.float(), atol=1e-6, rtol=1e-4), (
            f'{label}: gradient of {key} differs by {difference:.3e}'
        )
    return worst, worst_key


def _assert_same_weights(got, want, label, max_drift):
    '''
    Weights are only compared loosely: AdamW's `eps` turns a near-zero gradient
    component into a full-size update, so a 1e-7 relative gradient difference can move
    a weight by a fraction of one learning-rate-sized step. The bound therefore has to
    be read as "the two runs took the same step", not as bitwise equality.
    '''
    assert set(got) == set(want), f'{label}: different parameter names'
    worst, worst_key = 0.0, None
    for key in want:
        difference = float((got[key].float() - want[key].float()).abs().max())
        if difference > worst:
            worst, worst_key = difference, key
    assert worst <= max_drift, f'{label}: {worst_key} differs by {worst:.3e} (> {max_drift:.3e})'
    return worst


def _assert_same_losses(got, want, label, tolerance):
    assert len(got) == len(want), f'{label}: {len(got)} steps vs {len(want)}'
    worst = max(abs(a - b) for a, b in zip(got, want))
    assert worst <= tolerance, f'{label}: loss curves differ by up to {worst:.3e}\n{got}\n{want}'
    return worst


@requires_dist
def test_one_rank_and_two_ranks_compute_the_same_step(tmp_path=None):
    '''One optimiser step: the split has to be a real split and the gradient the same.'''
    base = scratch(tmp_path)
    single_out = base / 'single_out'
    double_out = base / 'double_out'

    _run_world(1, ['--steps', '1', '--ckpt', str(base / 'single_ckpt'), '--out', str(single_out)])
    _run_world(2, ['--steps', '1', '--ckpt', str(base / 'double_ckpt'), '--out', str(double_out)])

    single = torch.load(single_out / 'rank_0.pt', map_location='cpu', weights_only=False)
    rank0 = torch.load(double_out / 'rank_0.pt', map_location='cpu', weights_only=False)
    rank1 = torch.load(double_out / 'rank_1.pt', map_location='cpu', weights_only=False)

    assert single['world_size'] == 1 and not single['distributed']
    assert rank0['world_size'] == 2 and rank0['distributed']
    assert rank1['rank'] == 1 and rank1['local_rank'] == 1
    assert single['stage_batches'] == rank0['stage_batches'] == rank1['stage_batches']
    assert single['steps_in_plan'] == rank0['steps_in_plan']
    assert single['global_step'] == rank0['global_step'] == rank1['global_step'] == 1
    assert rank0['shard'] == [0, 2] and rank1['shard'] == [1, 2]

    single_step, first_step, second_step = single['batches'][0], rank0['batches'][0], rank1['batches'][0]
    assert single_step[0] == first_step[0] == second_step[0], 'stage mismatch'
    assert torch.equal(torch.cat([first_step[1], second_step[1]]), single_step[1]), \
        'the ranks did not cover the global batch'
    assert torch.equal(torch.cat([first_step[2], second_step[2]]), single_step[2]), \
        'the labels of the ranks do not reassemble'
    assert first_step[1].shape[0] < single_step[1].shape[0], 'the batch was not split'

    _assert_same_losses(rank0['losses'], single['losses'], 'step loss', tolerance=1e-5)
    assert rank0['tokens'] == single['tokens'], 'the global token count drifted'

    _assert_same_gradients(rank0['grads'][0], single['grads'][0], 'rank 0 gradient')

    _assert_same_weights(rank1['weights'], rank0['weights'], 'replica weights', max_drift=0.0)
    _assert_same_weights(rank0['weights'], single['weights'], 'weights', max_drift=0.5 * 1e-2)


@requires_dist
def test_distributed_ranks_stay_in_step(tmp_path=None):
    '''Both ranks have to end up with the same weights, or DDP is not syncing.'''
    base = scratch(tmp_path)
    out = base / 'out'
    _run_world(2, ['--steps', '3', '--ckpt', str(base / 'ckpt'), '--out', str(out)])

    rank0 = torch.load(out / 'rank_0.pt', map_location='cpu', weights_only=False)
    rank1 = torch.load(out / 'rank_1.pt', map_location='cpu', weights_only=False)
    _assert_same_weights(rank1['weights'], rank0['weights'], 'replica weights', max_drift=0.0)
    _assert_same_losses(rank1['losses'], rank0['losses'], 'replica loss curve', tolerance=0.0)


@requires_dist
def test_checkpoint_of_a_distributed_run_is_a_plain_single_model_checkpoint(tmp_path=None):
    base = scratch(tmp_path)
    ckpt_dir = base / 'ckpt'
    _run_world(2, ['--steps', '2', '--ckpt', str(ckpt_dir), '--out', str(base / 'out')])

    assert os.path.exists(ckpt_dir / 'last.pt'), 'rank 0 did not write the checkpoint'
    payload = torch.load(ckpt_dir / 'last.pt', map_location='cpu', weights_only=False)
    keys = list(payload['payload']['model'].keys())
    assert not any(key.startswith('module.') for key in keys), 'DDP prefix leaked into the checkpoint'
    assert payload['state']['global_step'] == 2


@requires_dist
def test_resuming_a_single_gpu_run_on_two_gpus_matches_an_uninterrupted_run(tmp_path=None):
    '''The data-flow cursor is world-size agnostic: 3 steps on 1 rank plus 3 steps on
    2 ranks has to compute what 6 uninterrupted single-rank steps compute.'''
    base = scratch(tmp_path)
    reference_out, first_out, resumed_out = (base / name for name in
                                             ('ref_out', 'first_out', 'resumed_out'))

    _run_world(1, ['--steps', '6', '--ckpt', str(base / 'ref_ckpt'), '--out', str(reference_out)])
    _run_world(1, ['--steps', '3', '--ckpt', str(base / 'split_ckpt'), '--out', str(first_out)])
    _run_world(2, ['--steps', '6', '--ckpt', str(base / 'split_ckpt'), '--out', str(resumed_out)])

    reference = torch.load(reference_out / 'rank_0.pt', map_location='cpu', weights_only=False)
    first = torch.load(first_out / 'rank_0.pt', map_location='cpu', weights_only=False)
    resumed = torch.load(resumed_out / 'rank_0.pt', map_location='cpu', weights_only=False)

    assert first['global_step'] == 3 and len(first['losses']) == 3
    assert resumed['global_step'] == 6, 'the resumed world did not continue the run'
    assert len(resumed['losses']) == 3, 'the resumed run replayed steps it had already done'
    assert len(resumed['batches']) == 3, 'the resumed run replayed data it had already seen'

    _assert_same_gradients(resumed['grads'][0], reference['grads'][3], 'gradient after resume')
    _assert_same_losses([resumed['losses'][0]], [reference['losses'][3]],
                        'first step after resume', tolerance=1e-5)

    _assert_same_losses(resumed['losses'], reference['losses'][3:],
                        'loss curve after resume', tolerance=1e-3)
    _assert_same_weights(resumed['weights'], reference['weights'], 'weights after resume',
                         max_drift=0.5 * 1e-2)


@requires_dist
def test_launch_spawns_workers_and_publishes_the_rank_layout(tmp_path=None):
    out_dir = scratch(tmp_path) / 'ranks'
    env = dict(os.environ)
    for name in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(name, None)
    env['DIST_LAUNCH_OUT'] = str(out_dir)

    process = subprocess.run(
        [sys.executable, LAUNCHER, '2'], cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=TIMEOUT,
    )
    assert process.returncode == 0, f'{process.stdout}\n{process.stderr}'

    reports = []
    for rank in range(2):
        path = out_dir / f'rank_{rank}.json'
        assert path.exists(), f'rank {rank} never reported'
        reports.append(json.loads(path.read_text(encoding='utf-8')))

    for rank, report in enumerate(reports):
        assert report['rank'] == rank
        assert report['context_rank'] == rank
        assert report['world_size'] == report['context_world'] == 2
        assert report['local_rank'] == rank
        assert report['sum'] == 1.0, 'the all-reduce over the spawned group is wrong'


if __name__ == '__main__':
    try:
        for name, function in sorted(globals().items()):
            if name.startswith('test_') and callable(function):
                print(f'--- {name}')
                function()
    finally:
        for directory in _SCRATCH:
            shutil.rmtree(directory, ignore_errors=True)
    print('all distributed tests passed')
