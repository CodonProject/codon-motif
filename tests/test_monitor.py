import os
import shutil
import tempfile
import time

import torch

from motif.model.config import MotifChordConfig
from motif.model.motif import MotifChordModel
from motif.pipeline import (
    FileLogConfig,
    FileLogger,
    PretrainConfig,
    PretrainPipeline,
    SFTConfig,
    SFTPipeline,
    TensorBoardCallback,
    TensorBoardConfig,
    build_monitor,
    combine_sinks,
)
from motif.pipeline.callback import Callback

from support import ListDataset, make_tokenizer

def make_workdir(name: str = 'monitor') -> str:
    path = os.path.join(
        tempfile.gettempdir(), f'{name}_{os.getpid()}_{int(time.time() * 1000)}'
    )
    os.makedirs(path, exist_ok=True)
    return path

class StopAfter(Callback):

    def __init__(self, n):
        self.n = n

    def on_step_end(self, pipeline, step_metrics):
        if pipeline.state.global_step >= self.n:
            self.stop_requested = True

class FakeTBWriter:

    def __init__(self, fail_on=None):
        self.scalars = []
        self.texts = []
        self.flushed = 0
        self.closed = 0
        self.fail_on = fail_on

    def _maybe_fail(self, kind):
        if self.fail_on == kind:
            raise RuntimeError(f'boom:{kind}')

    def add_scalar(self, tag, value, step):
        self._maybe_fail('scalar')
        self.scalars.append((tag, float(value), int(step)))

    def add_text(self, tag, text, step):
        self._maybe_fail('text')
        self.texts.append((tag, text, int(step)))

    def flush(self):
        self.flushed += 1

    def close(self):
        self.closed += 1

    def tags(self):
        return [tag for tag, _, _ in self.scalars]

    def steps_for(self, tag):
        return [step for name, _, step in self.scalars if name == tag]

def make_pipeline(workdir, records=48, **config_kwargs):
    tokenizer = make_tokenizer()
    torch.manual_seed(0)
    model = MotifChordModel(MotifChordConfig(
        vocab_size=tokenizer.vocab_size, model_dim=32, num_heads=4, num_kv_heads=2,
        head_k_dim=8, block=('gdn', 'chord'), block_n=2))
    data = [[(i * 5 + j) % 97 + 20 for j in range(1 + (i % 5))] for i in range(records)]
    kwargs = dict(
        base_context=4, target_context=8, global_batch_tokens=16, step_mode='min',
        learning_rate=1e-3, warmup_steps=1, ckpt_dir=os.path.join(workdir, 'ckpt'),
        save_every_steps=0, probe_every_steps=0, use_progress=False, amp=False)
    kwargs.update(config_kwargs)
    pipe = PretrainPipeline(model, tokenizer, PretrainConfig(**kwargs),
                            device='cpu', seed=0)
    return pipe, ListDataset(data)

def test_file_logger_splits_sanity_and_progress_logs():
    workdir = make_workdir('log_split')
    log_path = os.path.join(workdir, 'progress.log')
    sanity_path = os.path.join(workdir, 'sanity.log')
    try:
        pipe, data = make_pipeline(workdir, log_file=log_path)
        pipe.train(data, num_epochs=1, steps_per_epoch=3)

        assert os.path.isfile(sanity_path), 'sanity.log was not created'
        with open(sanity_path, encoding='utf-8') as f:
            sanity = f.read()
        with open(log_path, encoding='utf-8') as f:
            progress = f.read()

        assert 'sanity check | PretrainPipeline' in sanity
        assert 'pipeline    : PretrainPipeline' in sanity, 'describe() was not written to sanity.log'
        assert 'Context Training Plan' in sanity, 'the curriculum table was not written to sanity.log'
        assert 'Foundation' in sanity and 'Stabilization' in sanity
        assert 'loss/train=' not in sanity, 'sanity.log must not contain per-step progress'

        lines = progress.splitlines()
        assert 'sanity check ->' in progress and 'sanity.log' in progress, 'there is no pointer to sanity.log'
        assert 'pipeline    : PretrainPipeline' not in progress, 'the sanity block must not be duplicated into the progress log'
        steps = [ln for ln in lines if 'step=' in ln and 'loss/train=' in ln]
        assert len(steps) == 3, f'expected 3 progress lines, got {len(steps)}: {steps[:3]}'
        assert 'lr=' in steps[0] and 'grad_norm=' in steps[0]
        assert 'run end | FINISHED' in progress, 'the final summary was not written'
        assert 'loss/train =' in progress, 'the final metric snapshot was not written'
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def test_file_logger_can_keep_everything_in_one_file():
    workdir = make_workdir('log_single')
    log_path = os.path.join(workdir, 'all.log')
    try:
        pipe, data = make_pipeline(
            workdir, log_file=FileLogConfig(path=log_path, split_sanity=False))
        pipe.train(data, num_epochs=1, steps_per_epoch=2)

        with open(log_path, encoding='utf-8') as f:
            text = f.read()
        assert 'sanity check | PretrainPipeline' in text
        assert 'Context Training Plan' in text
        assert any('loss/train=' in ln for ln in text.splitlines())
        assert not os.path.exists(os.path.join(workdir, 'sanity.log')), \
            'sanity.log must not be created when split_sanity=False'
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def test_file_logger_custom_sanity_path():
    workdir = make_workdir('log_custom')
    log_path = os.path.join(workdir, 'progress.log')
    sanity_path = os.path.join(workdir, 'meta', 'sanity.log')
    try:
        pipe, data = make_pipeline(
            workdir, log_file=FileLogConfig(path=log_path, sanity_path=sanity_path))
        pipe.train(data, num_epochs=1, steps_per_epoch=1)
        assert os.path.isfile(sanity_path), 'the custom sanity_path did not take effect (its directory should be created automatically too)'
        assert 'sanity check | PretrainPipeline' in open(
            sanity_path, encoding='utf-8').read()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def test_file_logger_flushes_each_line_before_training_ends():
    workdir = make_workdir('log_flush')
    log_path = os.path.join(workdir, 'progress.log')
    try:
        pipe, data = make_pipeline(workdir, log_file=log_path)
        seen = []
        pipe.setup()
        runner = next(pipe.iterate_epochs(data))
        iterator = iter(runner)
        for _ in range(2):
            pipe._optimizer_step([next(iterator)], 1.0, False)
            with open(log_path, encoding='utf-8') as f:
                seen.append([ln for ln in f.read().splitlines() if 'step=' in ln])
        assert len(seen[0]) == 1, 'the file should hold 1 progress line after the first step'
        assert len(seen[1]) == 2, 'the file should hold 2 progress lines after the second step'
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def test_file_logger_every_steps_and_append():
    workdir = make_workdir('log_every')
    log_path = os.path.join(workdir, 'progress.log')
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('上一次训练的遗留内容\n')

        pipe, data = make_pipeline(
            workdir, log_file=FileLogConfig(path=log_path, every_steps=2, append=True))
        pipe.train(data, num_epochs=1, steps_per_epoch=4)

        with open(log_path, encoding='utf-8') as f:
            text = f.read()
        assert '上一次训练的遗留内容' in text, 'append=True must not wipe the old log'
        progress = [ln for ln in text.splitlines() if 'step=' in ln and 'loss/train=' in ln]
        assert len(progress) == 2, f'every_steps=2 over 4 steps should write 2 lines, got {len(progress)}'
        assert 'step=2 ' in progress[0] and 'step=4 ' in progress[1]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def test_file_logger_overwrite_when_append_false():
    workdir = make_workdir('log_trunc')
    log_path = os.path.join(workdir, 'train.log')
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('旧内容\n')
        logger = FileLogger(FileLogConfig(path=log_path, append=False))
        logger.log('新内容')
        logger.close()
        with open(log_path, encoding='utf-8') as f:
            text = f.read()
        assert '旧内容' not in text and '新内容' in text
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def test_tensorboard_scalars_texts_and_close():
    workdir = make_workdir('tb_fake')
    writer = FakeTBWriter()
    callback = TensorBoardCallback(
        TensorBoardConfig(log_dir=os.path.join(workdir, 'tb'), every_steps=1),
        writer=writer,
    )
    try:
        pipe, data = make_pipeline(workdir, tensorboard=callback)
        pipe.train(data, num_epochs=1, steps_per_epoch=3)

        assert 'Loss/train_avg' in writer.tags(), f'tags were not mapped as in a2.py: {set(writer.tags())}'
        assert 'LR' in writer.tags() and 'Grad_norm' in writer.tags()
        assert writer.steps_for('Loss/train_avg') == [1, 2, 3], 'step must follow global_step'
        assert any(tag == 'Run/config' for tag, _, _ in writer.texts), 'the sanity check was not reported'
        assert any('Context Training Plan' in text for _, text, _ in writer.texts), \
            'the sanity check should carry the curriculum table'
        assert writer.flushed >= 1 and writer.closed == 1, 'teardown must flush + close exactly once'
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def test_tensorboard_every_steps_gates_scalars():
    workdir = make_workdir('tb_every')
    writer = FakeTBWriter()
    callback = TensorBoardCallback(TensorBoardConfig(every_steps=2), writer=writer)
    try:
        pipe, data = make_pipeline(workdir, tensorboard=callback)
        pipe.train(data, num_epochs=1, steps_per_epoch=4)
        assert writer.steps_for('Loss/train_avg') == [2, 4]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def test_tensorboard_writer_failure_does_not_kill_training():
    workdir = make_workdir('tb_fail')
    writer = FakeTBWriter(fail_on='scalar')
    try:
        pipe, data = make_pipeline(
            workdir, tensorboard=TensorBoardCallback(TensorBoardConfig(), writer=writer))
        metrics = pipe.train(data, num_epochs=1, steps_per_epoch=2)
        assert pipe.state.global_step == 2, 'a reporting failure must not affect the training step count'
        assert metrics, 'training metrics are still returned as usual'
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def test_tensorboard_real_writer_creates_event_file():
    workdir = make_workdir('tb_real')
    tb_dir = os.path.join(workdir, 'tb')
    try:
        try:
            import torch.utils.tensorboard
            available = True
        except Exception:
            available = False

        pipe, data = make_pipeline(workdir, tensorboard=TensorBoardConfig(log_dir=tb_dir))
        pipe.train(data, num_epochs=1, steps_per_epoch=2)

        files = []
        for root, _, names in os.walk(tb_dir):
            files += [os.path.join(root, n) for n in names]
        if available:
            assert any(os.path.basename(p).startswith('events.out.tfevents')
                       for p in files), f'no event file found: {files}'
            assert any(os.path.getsize(p) > 0 for p in files), 'the event file is empty'
        else:
            assert not files, 'no event file should be produced when tensorboard is not installed'
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def test_probe_text_reaches_both_tensorboard_and_log_file():
    workdir = make_workdir('fan_out')
    log_path = os.path.join(workdir, 'train.log')
    writer = FakeTBWriter()
    try:
        pipe, data = make_pipeline(
            workdir,
            tensorboard=TensorBoardCallback(TensorBoardConfig(), writer=writer),
            log_file=log_path,
            probe_every_steps=1,
            probe_prompts=['the quick'],
            probe_max_new_tokens=4,
        )
        pipe.train(data, num_epochs=1, steps_per_epoch=1)

        tb_texts = [tag for tag, _, _ in writer.texts]
        assert any(tag.startswith('Sanity/probe') for tag in tb_texts), \
            f'TensorBoard did not receive the probe text: {tb_texts}'

        with open(log_path, encoding='utf-8') as f:
            log_text = f.read()
        assert 'Sanity/probe' in log_text, 'the log file did not receive the probe text'
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def test_combine_sinks_fanout_and_isolation():
    calls = []

    def good(tag, text, step):
        calls.append((tag, text, step))

    def bad(tag, text, step):
        raise RuntimeError('boom')

    assert combine_sinks([]) is None
    assert combine_sinks([None, None]) is None
    assert combine_sinks([good]) is good, 'a single sink must not be wrapped again'

    fan = combine_sinks([bad, good])
    fan('t', 'x', 1)
    fan('t', 'y', 2)
    assert calls == [('t', 'x', 1), ('t', 'y', 2)], 'a broken sink must not block another one'

def test_build_monitor_accepts_path_dataclass_and_callback():
    workdir = make_workdir('build_monitor')
    try:
        assert build_monitor(None) is None
        assert build_monitor(False) is None

        from_path = build_monitor(os.path.join(workdir, 'a.log'))
        assert isinstance(from_path, FileLogger) and from_path.path.endswith('a.log')

        from_config = build_monitor(FileLogConfig(path=os.path.join(workdir, 'b.log')))
        assert isinstance(from_config, FileLogger)

        tb = build_monitor(TensorBoardConfig(log_dir=os.path.join(workdir, 'tb')))
        assert isinstance(tb, TensorBoardCallback)

        existing = FileLogger(FileLogConfig(path=os.path.join(workdir, 'c.log')))
        assert build_monitor(existing) is existing

        try:
            build_monitor(123)
        except TypeError as exc:
            assert 'FileLogConfig' in str(exc)
        else:
            raise AssertionError('an invalid type must raise')
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def test_sft_pipeline_logs_stage_table():
    workdir = make_workdir('sft_log')
    log_path = os.path.join(workdir, 'sft.log')
    try:
        folder = os.path.join(workdir, 'data')
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, 'part.jsonl'), 'w', encoding='utf-8') as f:
            for i in range(16):
                f.write(f'{{"input": "q{i}", "content": "a{i}"}}\n')

        tokenizer = make_tokenizer()
        torch.manual_seed(0)
        model = MotifChordModel(MotifChordConfig(
            vocab_size=tokenizer.vocab_size, model_dim=32, num_heads=4, num_kv_heads=2,
            head_k_dim=8, block=('gdn', 'chord'), block_n=2))
        pipe = SFTPipeline(
            model, tokenizer,
            SFTConfig(learning_rate=1e-3, warmup_steps=1,
                      ckpt_dir=os.path.join(workdir, 'ck'), save_every_steps=0,
                      use_progress=False, amp=False, pad_length=48, batch_size=2,
                      stage_specs=[{'name': 'main', 'folder': folder, 'epochs': 1}],
                      log_file=log_path),
            callbacks=[StopAfter(2)], seed=0, device='cpu')
        pipe.train()

        with open(os.path.join(workdir, 'sanity.log'), encoding='utf-8') as f:
            sanity = f.read()
        with open(log_path, encoding='utf-8') as f:
            text = f.read()
        assert 'sanity check | SFTPipeline' in sanity
        assert 'epochs      :' in sanity, 'the SFT sanity check did not write the total epoch count'
        assert '-> [1] main' in sanity, 'the SFT sanity check did not write the stage table'
        assert any('loss/train=' in ln for ln in text.splitlines()), 'there is no per-step progress'
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def test_unwritable_log_file_degrades_without_required():
    workdir = make_workdir('log_bad')
    try:
        blocker = os.path.join(workdir, 'blocker')
        with open(blocker, 'w', encoding='utf-8') as f:
            f.write('x')
        bad_path = os.path.join(blocker, 'nested', 'train.log')

        pipe, data = make_pipeline(workdir, log_file=bad_path)
        metrics = pipe.train(data, num_epochs=1, steps_per_epoch=1)
        assert metrics and pipe.state.global_step == 1, 'an unavailable log must not affect training'

        logger = FileLogger(FileLogConfig(path=bad_path, required=True))
        try:
            logger.on_train_start(pipe)
        except RuntimeError as exc:
            assert 'train.log' in str(exc)
        else:
            raise AssertionError('an unopenable log must raise when required=True')
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# --- snapshot callbacks ---
import shutil
import tempfile
import time

import torch

from motif.model.config import MotifChordConfig
from motif.model.motif import MotifChordModel
from motif.pipeline import (
    PretrainConfig,
    PretrainPipeline,
    SnapshotCallback,
    SnapshotConfig,
    build_snapshot,
    snapshot_targets,
)
from motif.pipeline.callback import Callback

from support import ListDataset, make_tokenizer

class ForceTotalSteps(Callback):

    def __init__(self, total):
        self.total = total

    def on_train_start(self, pipeline):
        pipeline._total_steps = self.total

def _budgetmake_workdir(name: str = 'snapshot') -> str:
    path = os.path.join(
        tempfile.gettempdir(), f'{name}_{os.getpid()}_{int(time.time() * 1000)}')
    os.makedirs(path, exist_ok=True)
    return path

def _budgetmake_pipeline(workdir, records=64, callbacks=None, **config_kwargs):
    tokenizer = make_tokenizer()
    torch.manual_seed(0)
    model = MotifChordModel(MotifChordConfig(
        vocab_size=tokenizer.vocab_size, model_dim=32, num_heads=4, num_kv_heads=2,
        head_k_dim=8, block=('gdn', 'chord'), block_n=2))
    data = [[(i * 5 + j) % 97 + 20 for j in range(1 + (i % 5))] for i in range(records)]
    kwargs = dict(
        base_context=4, target_context=8, global_batch_tokens=16, step_mode='min',
        learning_rate=1e-3, warmup_steps=1, ckpt_dir=os.path.join(workdir, 'ckpt'),
        save_every_steps=0, probe_every_steps=0, use_progress=False, amp=False)
    kwargs.update(config_kwargs)
    pipe = PretrainPipeline(model, tokenizer, PretrainConfig(**kwargs),
                            device='cpu', seed=0, callbacks=list(callbacks or []))
    return pipe, ListDataset(data), model

def test_snapshot_targets_are_deterministic():
    targets = snapshot_targets(176672, 5)
    assert len(targets) == 20, f'5% should give 20 targets, got {len(targets)}'
    assert targets[0] == 8834, f'the first target should be ceil(176672*5%) = 8834, got {targets[0]}'
    assert targets[-1] == 176672, 'the last target should land exactly on the total step count'
    assert targets == sorted(set(targets)), 'targets must be strictly increasing and unique'

    assert snapshot_targets(100, 33.3) == [34, 67, 100]
    assert snapshot_targets(10, 100) == [10]
    assert snapshot_targets(0, 5) == []
    assert snapshot_targets(100, 0) == []

    assert snapshot_targets(176672, 5) == targets

def test_build_snapshot_switch():
    assert build_snapshot(None) is None
    assert build_snapshot(0) is None
    assert build_snapshot(-3) is None

    cb = build_snapshot(5, dir='somewhere')
    assert isinstance(cb, SnapshotCallback) and cb.config.every_percent == 5

    try:
        build_snapshot(150)
    except ValueError as exc:
        assert '100' in str(exc)
    else:
        raise AssertionError('a percentage above 100 must raise')

def test_pipeline_saves_pure_weights_every_percent():
    workdir = _budgetmake_workdir('snap_e2e')
    snap_dir = os.path.join(workdir, 'weights')
    try:
        pipe, data, model = _budgetmake_pipeline(
            workdir, snapshot_every_percent=25, snapshot_dir=snap_dir,
            callbacks=[ForceTotalSteps(8)])
        pipe.train(data, num_epochs=1, steps_per_epoch=8)

        files = sorted(os.listdir(snap_dir))
        assert files == [f'step_{s:09d}.pt' for s in (2, 4, 6, 8)], \
            f'25% x total_steps=8 should save step 2/4/6/8, got {files}'

        reference = model.state_dict()
        for name in files:
            state = torch.load(os.path.join(snap_dir, name), map_location='cpu',
                               weights_only=False)
            assert isinstance(state, dict), f'{name} is not a pure state_dict'
            assert set(state) == set(reference), f'{name} keys do not match the model'
            assert 'optimizer' not in state and 'model' not in state and 'data' not in state, \
                f'{name} carries other parts of a checkpoint'
            torch.manual_seed(1)
            fresh = MotifChordModel(MotifChordConfig(
                vocab_size=make_tokenizer().vocab_size, model_dim=32, num_heads=4,
                num_kv_heads=2, head_k_dim=8, block=('gdn', 'chord'), block_n=2))
            fresh.load_state_dict(state)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def test_pipeline_snapshot_default_dir_and_at_end():
    workdir = _budgetmake_workdir('snap_end')
    try:
        pipe, data, _ = _budgetmake_pipeline(
            workdir, snapshot_every_percent=25, callbacks=[ForceTotalSteps(8)])
        pipe.train(data, num_epochs=1, steps_per_epoch=3)

        snap_dir = os.path.join(workdir, 'ckpt', 'snapshots')
        files = sorted(os.listdir(snap_dir))
        assert files == ['step_000000002.pt', 'step_000000003.pt'], \
            f'expected one 25% target (2) plus the closing one (3), got {files}'
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def test_callback_standalone_and_sink():
    workdir = _budgetmake_workdir('snap_sink')
    snap_dir = os.path.join(workdir, 'w')
    seen = []

    config = SnapshotConfig(every_percent=50, dir=snap_dir, also_at_end=False)
    callback = SnapshotCallback(config, sink=lambda tag, text, step: seen.append((tag, step)))
    try:
        pipe, data, _ = _budgetmake_pipeline(workdir, callbacks=[ForceTotalSteps(4)])
        pipe.callbacks.append(callback)
        pipe.train(data, num_epochs=1, steps_per_epoch=4)

        assert sorted(os.listdir(snap_dir)) == ['step_000000002.pt', 'step_000000004.pt']
        assert [tag for tag, _ in seen] == ['Snapshots/save', 'Snapshots/save'], \
            f'the sink should receive two records, got {seen}'
        assert [step for _, step in seen] == [2, 4]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def test_existing_snapshot_is_not_overwritten():
    workdir = _budgetmake_workdir('snap_skip')
    snap_dir = os.path.join(workdir, 'w')
    os.makedirs(snap_dir, exist_ok=True)
    try:
        sentinel = os.path.join(snap_dir, 'step_000000002.pt')
        with open(sentinel, 'wb') as f:
            f.write(b'already-there')

        pipe, data, _ = _budgetmake_pipeline(
            workdir, snapshot_every_percent=25, snapshot_dir=snap_dir,
            callbacks=[ForceTotalSteps(4)])
        pipe.train(data, num_epochs=1, steps_per_epoch=4)

        with open(sentinel, 'rb') as f:
            assert f.read() == b'already-there', 'an existing snapshot was overwritten'
        assert os.path.isfile(os.path.join(snap_dir, 'step_000000004.pt')), \
            'the other targets must still be saved'
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def test_snapshot_recorded_in_log_file():
    workdir = _budgetmake_workdir('snap_log')
    log_path = os.path.join(workdir, 'progress.log')
    try:
        pipe, data, _ = _budgetmake_pipeline(
            workdir, snapshot_every_percent=50, snapshot_dir=os.path.join(workdir, 'w'),
            log_file=log_path, callbacks=[ForceTotalSteps(4)])
        pipe.train(data, num_epochs=1, steps_per_epoch=4)

        with open(log_path, encoding='utf-8') as f:
            text = f.read()
        assert 'Snapshots/save' in text, 'the snapshot was not recorded in the progress log'
        assert text.count('snapshot @') >= 2, 'both snapshots should leave a record'
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def test_snapshot_failure_degrades_but_required_raises():
    workdir = _budgetmake_workdir('snap_fail')
    try:
        blocker = os.path.join(workdir, 'blocker')
        with open(blocker, 'w', encoding='utf-8') as f:
            f.write('x')
        bad_dir = os.path.join(blocker, 'nested', 'weights')

        pipe, data, _ = _budgetmake_pipeline(
            workdir, snapshot_every_percent=50, snapshot_dir=bad_dir,
            callbacks=[ForceTotalSteps(4)])
        pipe.train(data, num_epochs=1, steps_per_epoch=4)
        assert pipe.state.global_step == 4, 'a failed snapshot must not affect training'

        required = SnapshotCallback(
            SnapshotConfig(every_percent=50, dir=bad_dir, required=True, also_at_end=False))
        pipe2, data2, _ = _budgetmake_pipeline(workdir, callbacks=[ForceTotalSteps(4)])
        pipe2.setup()
        pipe2._total_steps = 4
        required.on_train_start(pipe2)
        pipe2.state.global_step = 2
        try:
            required.on_step_end(pipe2, {})
        except Exception as exc:
            assert 'snapshot' in str(exc).lower()
        else:
            raise AssertionError('with required=True a failed snapshot must raise')
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def test_snapshot_disabled_by_default():
    workdir = _budgetmake_workdir('snap_off')
    try:
        pipe, data, _ = _budgetmake_pipeline(workdir, callbacks=[ForceTotalSteps(4)])
        assert not any(isinstance(cb, SnapshotCallback) for cb in pipe.callbacks)
        pipe.train(data, num_epochs=1, steps_per_epoch=2)
        assert not os.path.exists(os.path.join(workdir, 'ckpt', 'snapshots'))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
