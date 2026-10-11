import os
import shutil
import tempfile

import torch

from motif.data import TextFileDataset
from motif.model.motif import MotifChordModel
from motif.pipeline import Callback, PretrainConfig, PretrainPipeline

from support import make_model, make_tokenizer

class StopAfter(Callback):

    def __init__(self, n):
        self.n = n

    def on_step_end(self, pipeline, step_metrics):
        if pipeline.state.global_step >= self.n:
            self.stop_requested = True

class RecordingPretrain(PretrainPipeline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = []

    def compute_loss(self, batch):
        stage, inputs, labels = batch
        self.seen.append((stage.name, inputs.clone(), labels.clone()))
        return super().compute_loss(batch)

def _make_text_corpus(root: str, files: int = 6) -> str:
    os.makedirs(root, exist_ok=True)
    for i in range(files):
        with open(os.path.join(root, f'{i}.txt'), 'w', encoding='utf-8') as f:
            f.write(' '.join(f'word{j}' for j in range(60)) + f' file{i}')
    return root

def _pretrain_config(ckpt_dir: str, **overrides) -> PretrainConfig:
    config = PretrainConfig(
        compiled=False,
        base_context=16,
        target_context=32,
        global_batch_tokens=32,
        step_mode='min',
        learning_rate=1e-3,
        warmup_steps=1,
        ckpt_dir=ckpt_dir,
        save_every_steps=0,
        probe_every_steps=0,
        use_progress=False,
        amp=False,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config

def _dataset(tokenizer, root, seq_len=16, drop_last=True):
    return TextFileDataset(root, recursive=False, tokenizer=tokenizer, seq_len=seq_len,
                           drop_last=drop_last)

def _train_source(tokenizer, corpus, ckpt_dir, steps=3):
    pipe = RecordingPretrain(
        make_model(vocab_size=tokenizer.vocab_size), tokenizer, _pretrain_config(ckpt_dir),
        callbacks=[StopAfter(steps)], seed=0, device='cpu',
    )
    pipe.train(_dataset(tokenizer, corpus), num_epochs=1, grad_accum_steps=1)
    assert pipe.state.global_step == steps, 'the source run did not finish, so later assertions would be meaningless'
    assert os.path.exists(os.path.join(ckpt_dir, 'last.pt'))
    return pipe

def _constant_state(model: MotifChordModel, base: float) -> None:
    with torch.no_grad():
        for index, param in enumerate(model.parameters()):
            param.fill_(base + 0.001 * index)

def test_warm_start_takes_weights_only():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_continue_')
    try:
        corpus = _make_text_corpus(os.path.join(base, 'corpus'))
        source = _train_source(tokenizer, corpus, os.path.join(base, 'ckpt_src'), steps=3)
        assert source._optimizer.state, 'the source run should leave AdamW momentum behind, so we can verify it was not carried over'
        assert source.state.global_step == 3

        warm = PretrainPipeline(
            make_model(vocab_size=tokenizer.vocab_size), tokenizer,
            _pretrain_config(os.path.join(base, 'ckpt_new'),
                             init_from=os.path.join(base, 'ckpt_src', 'last.pt')),
            seed=1, device='cpu',
        )
        warm.setup()

        source_state = source.model.state_dict()
        for key, value in warm.model.state_dict().items():
            assert torch.equal(value, source_state[key]), f'{key} was not loaded from init_from'

        assert warm.state.global_step == 0, 'a warm start must not inherit the old step count'
        assert warm._scheduler.last_epoch == 0, 'a warm start must not inherit the old schedule position'
        assert warm._scheduler.total_steps == warm._total_steps
        assert len(warm._optimizer.state) == 0, 'AdamW momentum must not be carried over from init_from'
        assert not warm._pending_data, 'the data cursor must not be carried over from init_from'
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_warm_start_accepts_bare_snapshot():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_continue_')
    try:
        source = make_model(vocab_size=tokenizer.vocab_size)
        _constant_state(source, 0.01)
        snapshot = os.path.join(base, 'snapshot.pt')
        torch.save(source.state_dict(), snapshot)

        reader = PretrainPipeline(
            make_model(vocab_size=tokenizer.vocab_size), tokenizer,
            _pretrain_config(os.path.join(base, 'ckpt_a'), init_from=snapshot),
            device='cpu',
        )
        report = reader.load_weights(snapshot, verbose=False)
        assert report == {'missing': [], 'unexpected': []}, report

        warm = PretrainPipeline(
            make_model(vocab_size=tokenizer.vocab_size), tokenizer,
            _pretrain_config(os.path.join(base, 'ckpt_b'), init_from=snapshot),
            device='cpu',
        )
        warm.setup()
        source_state = source.state_dict()
        for key, value in warm.model.state_dict().items():
            assert torch.equal(value, source_state[key]), f'{key} does not match the snapshot'
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_load_weights_reports_mismatches_and_strict_raises():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_continue_')
    try:
        state = dict(make_model(vocab_size=tokenizer.vocab_size).state_dict())
        dropped = sorted(state)[0]
        state.pop(dropped)
        state['not.a.real.tensor'] = torch.zeros(1)
        path = os.path.join(base, 'odd.pt')
        torch.save(state, path)

        pipe = PretrainPipeline(
            make_model(vocab_size=tokenizer.vocab_size), tokenizer,
            _pretrain_config(os.path.join(base, 'ckpt')), device='cpu',
        )
        report = pipe.load_weights(path, verbose=False)
        assert dropped in report['missing'], report
        assert 'not.a.real.tensor' in report['unexpected'], report

        pipe.load_weights(path, verbose=True)

        try:
            pipe.load_weights(path, strict=True, verbose=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError('strict=True should raise on mismatched keys instead of silently half-loading')
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_resume_of_this_run_beats_init_from():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_continue_')
    try:
        corpus = _make_text_corpus(os.path.join(base, 'corpus'))
        ckpt_dir = os.path.join(base, 'ckpt')
        _train_source(tokenizer, corpus, ckpt_dir, steps=2)
        frozen = os.path.join(base, 'frozen.pt')
        shutil.copyfile(os.path.join(ckpt_dir, 'last.pt'), frozen)

        decoy = make_model(vocab_size=tokenizer.vocab_size)
        _constant_state(decoy, 0.5)
        decoy_path = os.path.join(base, 'decoy.pt')
        torch.save(decoy.state_dict(), decoy_path)

        resumed = RecordingPretrain(
            make_model(vocab_size=tokenizer.vocab_size), tokenizer,
            _pretrain_config(os.path.join(base, 'ckpt_resumed'), init_from=decoy_path),
            callbacks=[StopAfter(4)], seed=0, device='cpu',
        )
        resumed.train(_dataset(tokenizer, corpus), num_epochs=2, resume_from=frozen)
        assert resumed.state.global_step == 4, 'the resume did not continue from step 2'

        control = RecordingPretrain(
            make_model(vocab_size=tokenizer.vocab_size), tokenizer,
            _pretrain_config(os.path.join(base, 'ckpt_control')),
            callbacks=[StopAfter(4)], seed=0, device='cpu',
        )
        control.train(_dataset(tokenizer, corpus), num_epochs=2, resume_from=frozen)

        for key, value in control.model.state_dict().items():
            assert torch.equal(value, resumed.model.state_dict()[key]), \
                f'{key} was affected by init_from: it must be ignored on resume'
        assert [item[0] for item in control.seen] == [item[0] for item in resumed.seen], \
            'the resume read init_from, which shifted the curriculum stage'
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_warm_start_starts_the_new_corpus_from_the_beginning():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_continue_')
    try:
        corpus = _make_text_corpus(os.path.join(base, 'corpus'))
        source = _train_source(tokenizer, corpus, os.path.join(base, 'ckpt_src'), steps=3)
        source_ckpt = os.path.join(base, 'ckpt_src', 'last.pt')

        fresh = RecordingPretrain(
            make_model(vocab_size=tokenizer.vocab_size), tokenizer,
            _pretrain_config(os.path.join(base, 'ckpt_fresh')),
            callbacks=[StopAfter(2)], seed=0, device='cpu',
        )
        fresh.train(_dataset(tokenizer, corpus), num_epochs=1)

        warm = RecordingPretrain(
            make_model(vocab_size=tokenizer.vocab_size), tokenizer,
            _pretrain_config(os.path.join(base, 'ckpt_warm'), init_from=source_ckpt),
            callbacks=[StopAfter(2)], seed=0, device='cpu',
        )
        warm.train(_dataset(tokenizer, corpus), num_epochs=1)

        assert len(warm.seen) == 2 and len(fresh.seen) == 2
        for (got_stage, got_x, got_y), (want_stage, want_x, want_y) in zip(warm.seen, fresh.seen):
            assert got_stage == want_stage, 'after the warm start it stopped at the wrong curriculum stage'
            assert torch.equal(got_x, want_x), 'the input_ids after the warm start do not start at the head of the corpus'
            assert torch.equal(got_y, want_y), 'the labels after the warm start do not start at the head of the corpus'

        fresh_state = fresh.model.state_dict()
        assert any(not torch.equal(value, fresh_state[key])
                   for key, value in warm.model.state_dict().items()), \
            'the warm-started weights are identical to a fresh run, so init_from never took effect'
        assert source.state.global_step == 3
    finally:
        shutil.rmtree(base, ignore_errors=True)


# --- token budget / full_data ---
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from motif.data import TextFileDataset
from motif.pipeline import Callback, PretrainConfig, PretrainPipeline
from motif.pipeline.pretrain import cal_warmup

from support import make_model, make_tokenizer

LANG_PATTERN = ('zh2en', 'en2zh', None, 'en')

class PlainRows:

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int):
        return [1, 2, 3]

def _lang_of(row: int):
    return LANG_PATTERN[row % len(LANG_PATTERN)]

def _write_lang_shards(directory: str, shards: int = 3, rows: int = 21,
                       row_group_size: int = 7) -> str:
    os.makedirs(directory, exist_ok=True)
    for shard in range(shards):
        table = pa.table({
            'content': [f'shard{shard:02d}-row{row:03d}' for row in range(rows)],
            'lang': [_lang_of(row) for row in range(rows)],
        })
        pq.write_table(
            table,
            os.path.join(directory, f'part-{shard:03d}.parquet'),
            row_group_size=row_group_size,
        )
    return directory

def _pipeline(tokenizer, ckpt_dir, callbacks=None, **overrides) -> PretrainPipeline:
    return PretrainPipeline(
        make_model(vocab_size=tokenizer.vocab_size), tokenizer, _pretrain_config(ckpt_dir, **overrides),
        callbacks=callbacks, device='cpu',
    )

def _tokens_of(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, apply_safe_rule=False, add_special_tokens=False))

def test_total_tokens_replaces_the_preset():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_budget_tokens_')
    try:
        preset = _pipeline(tokenizer, os.path.join(base, 'ckpt_preset'))
        preset.setup()
        assert preset._plan.step_mode == 'min', preset._plan.step_mode

        custom = _pipeline(tokenizer, os.path.join(base, 'ckpt_custom'), total_tokens=100_000)
        custom.setup()
        plan = custom._plan

        assert plan.step_mode == 'custom', plan.step_mode
        assert plan.total_tokens == 100_000, plan.total_tokens
        assert plan.total_tokens != preset._plan.total_tokens, 'the budget was not replaced by the custom value'

        assert sum(stage.tokens for stage in plan.stages) == 100_000
        consumed = sum(stage.steps * stage.chunk_len for stage in plan.stages)
        assert consumed >= 100_000, consumed
        assert consumed - 100_000 < sum(stage.chunk_len for stage in plan.stages)

        assert plan.total_steps == sum(stage.steps for stage in plan.stages)
        assert custom._total_steps == plan.total_steps
        assert custom._scheduler.total_steps == plan.total_steps
        assert '[CUSTOM]' in '\n'.join(plan.report_lines())
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_total_steps_is_exact():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_budget_steps_')
    try:
        pipe = _pipeline(tokenizer, os.path.join(base, 'ckpt'),
                         total_steps=100, warmup_steps=None)
        pipe.setup()
        plan = pipe._plan

        assert plan.step_mode == 'steps', plan.step_mode
        assert plan.total_steps == 100, plan.total_steps
        assert sum(stage.steps for stage in plan.stages) == 100
        assert all(stage.steps >= 1 for stage in plan.stages), [s.steps for s in plan.stages]

        for stage in plan.stages:
            assert stage.tokens == stage.steps * stage.chunk_len, stage.name

        assert pipe._total_steps == 100
        assert pipe._scheduler.total_steps == 100
        assert plan.total_tokens == sum(stage.tokens for stage in plan.stages)
        assert pipe._scheduler.warmup_steps == cal_warmup(100) == 8
        assert '[STEPS]' in '\n'.join(plan.report_lines())
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_total_steps_wins_over_total_tokens():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_budget_both_')
    try:
        pipe = _pipeline(tokenizer, os.path.join(base, 'ckpt'),
                         total_tokens=9_999_999, total_steps=100, warmup_steps=None)
        pipe.setup()
        plan = pipe._plan

        assert plan.step_mode == 'steps', plan.step_mode
        assert plan.total_steps == 100
        assert plan.total_tokens == 9_999_999
        assert sum(stage.steps for stage in plan.stages) == 100
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_total_steps_turns_full_data_into_a_no_op():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_budget_noop_')
    try:
        pipe = _pipeline(tokenizer, os.path.join(base, 'ckpt'),
                         full_data=True, total_steps=100, warmup_steps=None)
        pipe.setup()
        plan = pipe._plan

        assert plan.step_mode == 'steps', plan.step_mode
        assert sum(stage.steps for stage in plan.stages) == 100
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_too_few_steps_to_cover_every_stage():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_budget_few_')
    try:
        pipe = _pipeline(tokenizer, os.path.join(base, 'ckpt'), total_steps=1)
        try:
            pipe.setup()
        except ValueError as exc:
            assert 'cannot be split over' in str(exc), str(exc)
        else:
            raise AssertionError('fewer steps than stages must raise ValueError')
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_a_non_positive_budget_is_rejected():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_budget_bad_')
    try:
        for overrides in ({'total_tokens': 0}, {'total_steps': -3}):
            pipe = _pipeline(tokenizer, os.path.join(base, 'ckpt'), **overrides)
            try:
                pipe.setup()
            except ValueError as exc:
                assert 'must be positive' in str(exc), str(exc)
            else:
                raise AssertionError(f'{overrides} must raise ValueError')
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_full_data_budget_is_one_pass_over_the_corpus():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_budget_full_')
    try:
        corpus = _make_text_corpus(os.path.join(base, 'corpus'))
        counted = _dataset(tokenizer, corpus).count_tokens()
        assert counted > 0

        pipe = _pipeline(tokenizer, os.path.join(base, 'ckpt'), full_data=True,
                         callbacks=[StopAfter(1)])
        pipe.train(_dataset(tokenizer, corpus), num_epochs=1)
        plan = pipe._plan

        assert plan.step_mode == 'full_data', plan.step_mode
        assert plan.total_tokens == counted, (plan.total_tokens, counted)
        assert '[FULL_DATA]' in '\n'.join(plan.report_lines())

        assert sum(stage.tokens for stage in plan.stages) == counted
        consumed = sum(stage.steps * stage.chunk_len for stage in plan.stages)
        assert consumed >= counted, (consumed, counted)
        assert consumed - counted < sum(stage.chunk_len for stage in plan.stages)
        assert pipe.state.global_step == 1
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_a_full_data_resume_reuses_the_counted_budget():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_budget_resume_')
    try:
        corpus = _make_text_corpus(os.path.join(base, 'corpus'))
        counted = _dataset(tokenizer, corpus).count_tokens()
        ckpt = os.path.join(base, 'ckpt')

        first = _pipeline(tokenizer, ckpt, full_data=True, callbacks=[StopAfter(1)])
        first.train(_dataset(tokenizer, corpus), num_epochs=1)
        assert first._plan.total_tokens == counted
        assert os.path.exists(os.path.join(ckpt, 'last.pt'))

        second = _pipeline(tokenizer, ckpt, full_data=True, callbacks=[StopAfter(2)])

        def forbidden():
            raise AssertionError('a resume must not count the corpus again')

        second._corpus_token_budget = forbidden
        second.train(_dataset(tokenizer, corpus), num_epochs=1)

        assert second.state.global_step == 2, 'the second run must resume rather than start from scratch'
        assert second._plan.step_mode == 'full_data'
        assert second._plan.total_tokens == counted, second._plan.total_tokens
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_full_data_rejects_an_explicit_token_budget():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_budget_conflict_')
    try:
        corpus = _make_text_corpus(os.path.join(base, 'corpus'))
        pipe = _pipeline(tokenizer, os.path.join(base, 'ckpt'),
                         full_data=True, total_tokens=1000)
        try:
            pipe.train(_dataset(tokenizer, corpus), num_epochs=1)
        except ValueError as exc:
            assert 'contradictory' in str(exc), str(exc)
        else:
            raise AssertionError('supplying full_data together with total_tokens must raise ValueError')
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_full_data_needs_a_dataset_that_can_count():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_budget_nodata_')
    try:
        pipe = _pipeline(tokenizer, os.path.join(base, 'ckpt'), full_data=True)
        try:
            pipe.train(None, num_epochs=1)
        except ValueError as exc:
            assert 'full_data' in str(exc), str(exc)
        else:
            raise AssertionError('full_data without a dataset must raise ValueError')

        other = _pipeline(tokenizer, os.path.join(base, 'ckpt2'), full_data=True)
        try:
            other.train(PlainRows(), num_epochs=1)
        except ValueError as exc:
            assert 'count its tokens' in str(exc), str(exc)
        else:
            raise AssertionError('full_data with a dataset that cannot count tokens must raise ValueError')
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_count_tokens_matches_the_tokenizer():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_count_')
    try:
        corpus = _make_text_corpus(os.path.join(base, 'corpus'), files=3)
        dataset = _dataset(tokenizer, corpus, drop_last=False)

        expected = 0
        for name in sorted(os.listdir(corpus)):
            with open(os.path.join(corpus, name), encoding='utf-8') as f:
                expected += _tokens_of(tokenizer, f.read())
        assert expected > 0
        assert dataset.count_tokens() == expected, (dataset.count_tokens(), expected)
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_count_tokens_needs_a_tokenizer():
    base = tempfile.mkdtemp(prefix='motif_count_none_')
    try:
        corpus = _make_text_corpus(os.path.join(base, 'corpus'), files=2)
        try:
            TextFileDataset(corpus).count_tokens()
        except ValueError as exc:
            assert 'tokenizer' in str(exc), str(exc)
        else:
            raise AssertionError('a dataset without a tokenizer must not be able to count tokens')
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_count_tokens_ignores_dropped_language_rows():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_count_lang_')
    try:
        folder = _write_lang_shards(os.path.join(base, 'shards'), shards=3, rows=21)
        plain = TextFileDataset(folder, tokenizer=tokenizer, seq_len=16, drop_last=False)
        filtered = TextFileDataset(folder, tokenizer=tokenizer, seq_len=16, drop_last=False,
                                   drop_lang=('zh2en', 'en2zh'))

        assert len(filtered) == len(plain) == 63

        dropped = 0
        for shard in range(3):
            for row in range(21):
                if _lang_of(row) in ('zh2en', 'en2zh'):
                    dropped += _tokens_of(tokenizer, f'shard{shard:02d}-row{row:03d}')
        assert dropped > 0

        assert filtered.count_tokens() == plain.count_tokens() - dropped, (
            filtered.count_tokens(), plain.count_tokens(), dropped
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)
