import os
import shutil
import tempfile

import torch

from motif.data import ChunkedTokenStream, MotifSFT, TextFileDataset
from motif.pipeline import (
    Callback,
    PretrainConfig,
    PretrainPipeline,
    SFTConfig,
    SFTPipeline,
)
from motif.utils.plan import Stage, StatefulPlanRunner, TrainingPlan

from support import ListDataset, make_model, make_tokenizer


class StopAfter(Callback):
    '''Request a stop once n optimization steps have completed (used to simulate training being interrupted midway).'''

    def __init__(self, n):
        self.n = n

    def on_step_end(self, pipeline, step_metrics):
        if pipeline.state.global_step >= self.n:
            self.stop_requested = True


def test_chunked_stream_resume_is_exact():
    records = [[(i * 7 + j) % 101 + 5 for j in range(1 + (i % 6))] for i in range(40)]
    make = lambda: ChunkedTokenStream(
        ListDataset(records), chunk_len=8, batch_size=2, seq_len=3, eos_token_id=2
    )

    full = list(make())
    assert len(full) > 6, 'the corpus is too small to test a mid-run checkpoint'

    stream = make()
    iterator = iter(stream)
    partial = [next(iterator) for _ in range(5)]
    state = stream.state_dict()
    del iterator

    resumed = make()
    resumed.load_state_dict(state)
    rest = list(resumed)

    assert len(partial) + len(rest) == len(full)
    for got, want in zip(partial + rest, full):
        assert torch.equal(got[0], want[0]), 'input_ids are misaligned across the checkpoint'
        assert torch.equal(got[1], want[1]), 'labels are misaligned across the checkpoint'


def test_chunked_stream_state_roundtrip():
    records = [[(i * 3 + j) % 50 + 5 for j in range(6)] for i in range(6)]
    make = lambda: ChunkedTokenStream(ListDataset(records), 8, 2, 3, eos_token_id=2)

    stream = make()
    iterator = iter(stream)
    next(iterator)
    state = stream.state_dict()

    other = make()
    other.load_state_dict(state)
    assert other.state_dict() == state


def _tiny_plan() -> TrainingPlan:
    stages = [
        Stage(name='Foundation', seq_len=4, batch_size=2, chunk_len=2 * 5, tokens=0, steps=6),
        Stage(name='Stabilization', seq_len=8, batch_size=1, chunk_len=1 * 9, tokens=0, steps=4),
    ]
    return TrainingPlan(total_tokens=0, total_steps=10, stages=stages, step_mode='test')


def _plan_data() -> ListDataset:
    return ListDataset([[(i * 13 + j) % 97 + 5 for j in range(1 + (i % 7))] for i in range(30)])


def test_plan_runner_resume_is_exact():
    plan = _tiny_plan()

    def steps(runner):
        return [(stage.name, inputs.clone(), labels.clone())
                for stage, inputs, labels in runner]

    full = steps(StatefulPlanRunner(plan, _plan_data(), 2))
    assert len(full) == 10

    runner = StatefulPlanRunner(plan, _plan_data(), 2)
    iterator = iter(runner)
    partial = [next(iterator) for _ in range(4)]
    state = runner.state_dict()
    del iterator

    resumed = StatefulPlanRunner(plan, _plan_data(), 2)
    resumed.load_state_dict(state)
    rest = list(resumed)

    assert len(partial) + len(rest) == 10
    for got, want in zip(steps(partial + rest), full):
        assert got[0] == want[0], 'the stage transition point does not line up'
        assert torch.equal(got[1], want[1])
        assert torch.equal(got[2], want[2])


SFT_ROWS = [
    '{"input": "what is 2+2?", "cot": "two plus two", "content": "4"}',
    '{"input": "say hi", "cot": "", "content": "hi there"}',
    '{"input": "name a color", "content": "blue"}',
    '{"input": "capital of France?", "cot": "France", "content": "Paris"}',
    '{"input": "1+1", "content": "2"}',
    '{"input": "who wrote Hamlet?", "content": "Shakespeare"}',
    '{"input": "sky color", "content": "blue"}',
    '{"input": "3*3", "content": "9"}',
] + [
    f'{{"input": "question {i}?", "content": "answer {i}"}}' for i in range(24)
]


def _write_sft_folder(directory: str) -> str:
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, 'part.jsonl'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(SFT_ROWS) + '\n')
    return directory


def _make_sft(folder: str, tokenizer) -> MotifSFT:
    return MotifSFT(folder, tokenizer, pad_length=48, batch_size=2, seed=7)


def test_motif_sft_resume_is_exact(tmp_path=None):
    tokenizer = make_tokenizer()
    folder = _write_sft_folder(os.path.join(tempfile.mkdtemp(prefix='motif_sft_'), 'data'))

    reference = _make_sft(folder, tokenizer)
    full = list(reference.iter_epoch(0))
    assert len(full) >= 3, 'too few samples to test a mid-run checkpoint'

    dataset = _make_sft(folder, tokenizer)
    iterator = iter(dataset.set_epoch(0))
    partial = [next(iterator) for _ in range(2)]
    state = dataset.state_dict()
    del iterator
    assert state == {'epoch': 0, 'cursor': 2}

    resumed = _make_sft(folder, tokenizer)
    resumed.load_state_dict(state)
    rest = list(resumed)

    assert len(partial) + len(rest) == len(full)
    for got, want in zip(partial + rest, full):
        assert torch.equal(got['input_ids'], want['input_ids']), 'SFT batches are misaligned across the checkpoint'
        assert torch.equal(got['labels'], want['labels'])


def test_motif_sft_epoch_order_is_deterministic():
    tokenizer = make_tokenizer()
    folder = _write_sft_folder(os.path.join(tempfile.mkdtemp(prefix='motif_sft_'), 'data'))

    first = [b['input_ids'] for b in _make_sft(folder, tokenizer).iter_epoch(0)]
    second = [b['input_ids'] for b in _make_sft(folder, tokenizer).iter_epoch(0)]
    for a, b in zip(first, second):
        assert torch.equal(a, b), 'the order within an epoch must be reproducible (otherwise resumed training cannot align)'

    dataset = _make_sft(folder, tokenizer)
    list(dataset.iter_epoch(0))
    assert list(dataset) == []
    assert len(list(dataset.iter_epoch(1))) == len(first)


class RecordingPretrain(PretrainPipeline):
    '''Record the batch actually consumed at each step, so interrupted-then-resumed and uninterrupted runs can be compared.'''

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


def _pretrain_config(ckpt_dir: str) -> PretrainConfig:
    return PretrainConfig(
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


def _pretrain_dataset(tokenizer, root, seq_len=16):
    return TextFileDataset(root, recursive=False, tokenizer=tokenizer, seq_len=seq_len)


def test_pretrain_pipeline_resume_is_exact():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_pretrain_')
    try:
        corpus = _make_text_corpus(os.path.join(base, 'corpus'))
        dataset = _pretrain_dataset(tokenizer, corpus)

        reference = RecordingPretrain(
            make_model(vocab_size=tokenizer.vocab_size), tokenizer,
            _pretrain_config(os.path.join(base, 'ckpt_ref')),
            callbacks=[StopAfter(5)], seed=0, device='cpu',
        )
        reference.train(dataset, num_epochs=1, grad_accum_steps=1)
        assert len(reference.seen) == 5

        ckpt_dir = os.path.join(base, 'ckpt')
        first = RecordingPretrain(
            make_model(vocab_size=tokenizer.vocab_size), tokenizer,
            _pretrain_config(ckpt_dir), callbacks=[StopAfter(3)], seed=0, device='cpu',
        )
        first.train(dataset, num_epochs=1, grad_accum_steps=1)
        assert len(first.seen) == 3
        assert os.path.exists(os.path.join(ckpt_dir, 'last.pt'))

        second = RecordingPretrain(
            make_model(vocab_size=tokenizer.vocab_size), tokenizer,
            _pretrain_config(ckpt_dir), callbacks=[StopAfter(5)], seed=0, device='cpu',
        )
        second.train(dataset, num_epochs=2)
        assert second.state.global_step == 5

        seen = first.seen + second.seen
        assert len(seen) == 5
        for (got_stage, got_x, got_y), (want_stage, want_x, want_y) in zip(seen, reference.seen):
            assert got_stage == want_stage, 'after resuming it landed on the wrong curriculum stage'
            assert torch.equal(got_x, want_x), 'input_ids after resuming differ from uninterrupted training'
            assert torch.equal(got_y, want_y), 'labels after resuming differ from uninterrupted training'

        for key, value in reference.model.state_dict().items():
            assert torch.equal(value, second.model.state_dict()[key]), f'{key} was not restored correctly'
    finally:
        shutil.rmtree(base, ignore_errors=True)


class RecordingSFT(SFTPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = []

    def compute_loss(self, batch):
        self.seen.append(batch['input_ids'].clone())
        return super().compute_loss(batch)


def _sft_config(folder: str, ckpt_dir: str) -> SFTConfig:
    return SFTConfig(
        compiled=False,
        learning_rate=1e-3,
        warmup_steps=1,
        ckpt_dir=ckpt_dir,
        save_every_steps=0,
        use_progress=False,
        amp=False,
        stage_specs=[{'name': 'tiny', 'folder': folder, 'epochs': 2}],
        pad_length=48,
        batch_size=2,
    )


def test_sft_pipeline_resume_is_exact():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_sft_pipe_')
    try:
        folder = _write_sft_folder(os.path.join(base, 'data'))

        reference = RecordingSFT(
            make_model(vocab_size=tokenizer.vocab_size), tokenizer,
            _sft_config(folder, os.path.join(base, 'ckpt_ref')),
            callbacks=[StopAfter(3)], seed=0, device='cpu',
        )
        reference.train()
        assert len(reference.seen) == 3

        ckpt_dir = os.path.join(base, 'ckpt')
        first = RecordingSFT(
            make_model(vocab_size=tokenizer.vocab_size), tokenizer,
            _sft_config(folder, ckpt_dir), callbacks=[StopAfter(2)], seed=0, device='cpu',
        )
        first.train()
        assert len(first.seen) == 2

        second = RecordingSFT(
            make_model(vocab_size=tokenizer.vocab_size), tokenizer,
            _sft_config(folder, ckpt_dir), callbacks=[StopAfter(3)], seed=0, device='cpu',
        )
        second.train(num_epochs=2)
        assert len(second.seen) == 1

        for got, want in zip(first.seen + second.seen, reference.seen):
            assert torch.equal(got, want), 'SFT batches after resuming differ from uninterrupted training'
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_sft_pipeline_advances_to_next_epoch():
    '''Once an epoch finishes it must switch to a fresh order for the next epoch rather than replaying in place.'''
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_sft_pipe_')
    try:
        folder = _write_sft_folder(os.path.join(base, 'data'))
        pipe = RecordingSFT(
            make_model(vocab_size=tokenizer.vocab_size), tokenizer,
            _sft_config(folder, os.path.join(base, 'ckpt')),
            seed=0, device='cpu',
        )
        pipe.train(num_epochs=2)

        dataset = pipe.stages[0].dataset
        assert pipe.state.epoch == 2
        assert dataset.epoch == 1, f'the second epoch should switch the dataset to epoch 1, got {dataset.epoch}'
        assert dataset.cursor == len(dataset), 'the epoch was not consumed completely'
        assert len(pipe.seen) == 2 * len(dataset), 'the step count for two epochs is wrong'

        pipe.train(num_epochs=3)
        assert dataset.epoch == 2
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_save_checkpoint_refuses_unrestored_data_state():
    '''Saving without restoring the data-flow state must raise instead of silently writing a corrupt checkpoint.'''
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_guard_')
    try:
        corpus = _make_text_corpus(os.path.join(base, 'corpus'))
        pipe = PretrainPipeline(
            make_model(vocab_size=tokenizer.vocab_size), tokenizer,
            _pretrain_config(os.path.join(base, 'ckpt')), seed=0, device='cpu',
        )
        pipe._pending_data = {'plan': {'stage_idx': 0, 'step_in_stage': 0}}
        try:
            pipe.save_checkpoint(os.path.join(base, 'boom.pt'))
        except RuntimeError as exc:
            assert 'data-flow state' in str(exc)
        else:
            raise AssertionError('saving without the data-flow state did not raise')
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_probe_generate_reply_is_signature_adaptive():
    '''The probe must be able to drive MotifChordModel.generate directly (no temperature argument, returns a Generation).'''
    from motif.pipeline.probe import build_generate_kwargs, generate_reply

    tokenizer = make_tokenizer()
    model = make_model(vocab_size=tokenizer.vocab_size)

    kwargs = build_generate_kwargs(model, max_new_tokens=4, temperature=0.8, eos_token_id=2)
    assert kwargs['max_new_tokens'] == 4
    assert kwargs['eos_token_id'] == 2
    assert 'temperature' not in kwargs, 'MotifChordModel.generate has no temperature parameter'
    assert 'sampler' in kwargs, 'it should fall back to a Sampler to carry the temperature'

    text = generate_reply(
        model, tokenizer, 'hello world', torch.device('cpu'),
        max_new_tokens=4, eos_token_id=2,
    )
    assert isinstance(text, str) and text


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            print(f'--- {name}')
            fn()
    print('all stateful tests passed')
