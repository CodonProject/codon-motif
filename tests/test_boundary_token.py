import os
import shutil
import tempfile
import time

import torch

from motif.data import ChunkedTokenStream
from motif.model.config import MotifChordConfig
from motif.model.motif import MotifChordModel
from motif.pipeline import PretrainConfig, PretrainPipeline, resolve_boundary_token_id
from motif.utils.plan import Stage, StatefulPlanRunner, TrainingPlan

from support import ListDataset, make_tokenizer


EOS = 9
BOUNDARY = 6


def make_ckpt_dir() -> str:
    '''
    Create a temporary checkpoint directory.

    `os.makedirs` is used deliberately instead of `tempfile.mkdtemp`: in some
    sandboxed environments, creating files inside a directory produced by
    mkdtemp (mode 0o700) is blocked and hangs, while `save_checkpoint` itself
    only does `os.makedirs(dirname(path), exist_ok=True)`, so makedirs actually
    matches the real training path.
    '''
    path = os.path.join(
        tempfile.gettempdir(), f'motif_boundary_{os.getpid()}_{int(time.time() * 1000)}'
    )
    os.makedirs(path, exist_ok=True)
    return path


def test_boundary_token_appended_and_trainable_when_mask_off():
    '''With the switch off (`mask_boundary=False`) this is the old behavior: the
    boundary token is a prediction target.'''
    stream = ChunkedTokenStream(
        ListDataset([[1, 2], [3]]), chunk_len=3, batch_size=1, seq_len=2,
        eos_token_id=EOS, boundary_token_id=BOUNDARY, mask_boundary=False,
    )
    iterator = iter(stream)
    inputs, labels = next(iterator)

    assert inputs.tolist() == [[1, 2]]
    assert labels.tolist() == [[2, BOUNDARY]], 'with masking off the boundary token must reach labels'

    list(iterator)
    assert stream.state_dict()['token_buffer'] == [3, BOUNDARY]


def test_mask_boundary_writes_ignore_index_into_labels():
    '''With the switch on (the pipeline default) the boundary token position is written
    as -100 in labels but still stays in input_ids.'''
    records = [[10, 11], [12]]
    stream = ChunkedTokenStream(
        ListDataset(records), chunk_len=4, batch_size=1, seq_len=3,
        eos_token_id=EOS, boundary_token_id=BOUNDARY, mask_boundary=True,
    )
    inputs, labels = next(iter(stream))
    assert inputs.tolist() == [[10, 11, BOUNDARY]], 'the boundary token must stay in the input (it is the boundary signal)'
    assert labels.tolist() == [[11, -100, 12]], 'the boundary token label position must be masked to -100'


def test_mask_boundary_ignored_by_cross_entropy():
    '''-100 positions really are excluded from the loss (PyTorch ignore_index defaults
    to -100).'''
    import torch.nn.functional as F
    stream = ChunkedTokenStream(
        ListDataset([[10, 11]]), chunk_len=3, batch_size=1, seq_len=2,
        eos_token_id=EOS, boundary_token_id=BOUNDARY, mask_boundary=True,
    )
    inputs, labels = next(iter(stream))
    vocab = 16
    logits = torch.zeros(1, labels.size(1), vocab)
    loss = F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1))
    kept = labels.reshape(-1)
    manual = F.cross_entropy(logits.reshape(-1, vocab)[kept != -100],
                             kept[kept != -100])
    assert torch.allclose(loss, manual), 'masked positions must be ignored by cross_entropy'


def test_boundary_is_trainable_target_not_just_a_separator():
    '''With masking off: for many consecutive short records every boundary token must
    land in labels, none may be dropped.'''
    records = [[10], [11], [12]]
    stream = ChunkedTokenStream(
        ListDataset(records), chunk_len=4, batch_size=1, seq_len=3,
        eos_token_id=EOS, boundary_token_id=BOUNDARY, mask_boundary=False,
    )
    inputs, labels = next(iter(stream))
    assert inputs.tolist() == [[10, BOUNDARY, 11]]
    assert labels.tolist() == [[BOUNDARY, 11, BOUNDARY]]


def test_default_boundary_falls_back_to_eos():
    '''When boundary_token_id is not passed it falls back to eos; masking is on by
    default, so the label is -100.'''
    stream = ChunkedTokenStream(
        ListDataset([[1, 2], [3]]), chunk_len=3, batch_size=1, seq_len=2,
        eos_token_id=EOS,
    )
    assert stream.boundary_token_id == EOS
    inputs, labels = next(iter(stream))
    assert inputs.tolist() == [[1, 2]]
    assert labels.tolist() == [[2, -100]], 'by default (masking on) the eos position must not count toward the loss'


def test_boundary_token_survives_resume():
    records = [[(i * 7 + j) % 101 + 5 for j in range(1 + (i % 6))] for i in range(40)]
    make = lambda: ChunkedTokenStream(
        ListDataset(records), chunk_len=8, batch_size=2, seq_len=3,
        eos_token_id=EOS, boundary_token_id=BOUNDARY,
    )

    full = list(make())
    stream = make()
    iterator = iter(stream)
    partial = [next(iterator) for _ in range(4)]
    state = stream.state_dict()
    del iterator
    assert state['boundary_token_id'] == BOUNDARY

    resumed = make()
    resumed.load_state_dict(state)
    rest = list(resumed)

    assert len(partial) + len(rest) == len(full), 'resuming with a boundary token gives a mismatched length'
    for got, want in zip(partial + rest, full):
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def test_changing_boundary_token_on_resume_is_loud():
    '''A resume that switches delimiters fails loudly instead of silently letting old
    records use the old delimiter and new records the new one.'''
    stream = ChunkedTokenStream(
        ListDataset([[1, 2], [3]]), chunk_len=3, batch_size=1, seq_len=2,
        eos_token_id=EOS, boundary_token_id=BOUNDARY,
    )
    state = stream.state_dict()

    other = ChunkedTokenStream(
        ListDataset([[1, 2], [3]]), chunk_len=3, batch_size=1, seq_len=2,
        eos_token_id=EOS, boundary_token_id=EOS,
    )
    try:
        other.load_state_dict(state)
    except ValueError as exc:
        assert 'boundary_token_id' in str(exc) or '边界 token' in str(exc)
    else:
        raise AssertionError('resuming with a different boundary token must fail, not silently mix delimiters')

    same = ChunkedTokenStream(
        ListDataset([[1, 2], [3]]), chunk_len=3, batch_size=1, seq_len=2,
        eos_token_id=EOS, boundary_token_id=BOUNDARY,
    )
    same.load_state_dict(state)
    assert same.boundary_token_id == BOUNDARY


def test_legacy_state_without_boundary_still_loads():
    '''A legacy checkpoint's stream state has no boundary field and must not blow up
    because of that.'''
    legacy = {'token_buffer': [7, 8], 'upstream_offset': 1}
    stream = ChunkedTokenStream(
        ListDataset([[1, 2], [3]]), chunk_len=3, batch_size=1, seq_len=2,
        eos_token_id=EOS, boundary_token_id=BOUNDARY,
    )
    stream.load_state_dict(legacy)
    assert stream.state_dict()['token_buffer'] == [7, 8]
    assert stream.state_dict()['upstream_offset'] == 1


def test_resolve_boundary_prefers_end_of_text():
    tokenizer = make_tokenizer()
    assert resolve_boundary_token_id(tokenizer) == tokenizer.token_to_id('<|end_of_text|>')

    assert resolve_boundary_token_id(tokenizer, 123) == 123

    assert resolve_boundary_token_id(None) is None

    class Empty:
        def token_to_id(self, token):
            return None

    assert resolve_boundary_token_id(Empty()) is None


def _plan(seq_len: int, steps: int) -> TrainingPlan:
    stage = Stage(name='foundation', seq_len=seq_len, chunk_len=1 * (seq_len + 1),
                  batch_size=1, tokens=0, steps=steps)
    return TrainingPlan(total_tokens=0, total_steps=steps, stages=[stage], step_mode='min')


def test_plan_runner_passes_boundary_and_mask_to_stream():
    runner = StatefulPlanRunner(
        _plan(seq_len=2, steps=1), ListDataset([[1, 2]]), EOS, boundary_token_id=BOUNDARY,
    )
    stage, inputs, labels = next(iter(runner))
    assert stage.name == 'foundation'
    assert inputs.tolist() == [[1, 2]]
    assert labels.tolist() == [[2, -100]]
    assert runner.state_dict()['stream']['boundary_token_id'] == BOUNDARY

    plain = StatefulPlanRunner(
        _plan(seq_len=2, steps=1), ListDataset([[1, 2]]), EOS,
        boundary_token_id=BOUNDARY, mask_boundary=False,
    )
    _, inputs2, labels2 = next(iter(plain))
    assert labels2.tolist() == [[2, BOUNDARY]]


def test_pretrain_pipeline_defaults_to_masking_boundary():
    '''End to end: the pipeline masks the boundary token by default (label -100) while
    the separation still stays in input_ids, and the pipeline resolves
    `<|end_of_text|>` from the tokenizer itself instead of leaving it to
    `ChunkedTokenStream` alone.'''
    tokenizer = make_tokenizer()
    boundary = tokenizer.token_to_id('<|end_of_text|>')
    assert boundary is not None

    class Recording(PretrainPipeline):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.seen = []

        def compute_loss(self, batch):
            stage, inputs, labels = batch
            self.seen.append((inputs.clone(), labels.clone()))
            return super().compute_loss(batch)

    base = make_ckpt_dir()
    try:
        torch.manual_seed(0)
        model = MotifChordModel(MotifChordConfig(
            vocab_size=tokenizer.vocab_size, model_dim=32, num_heads=4, num_kv_heads=2,
            head_k_dim=8, block=('gdn', 'chord'), block_n=2))
        config = PretrainConfig(
            base_context=4, target_context=8, global_batch_tokens=16, step_mode='min',
            learning_rate=1e-3, warmup_steps=1, ckpt_dir=base, save_every_steps=0,
            probe_every_steps=0, use_progress=False, amp=False)
        pipe = Recording(model, tokenizer, config, device='cpu', seed=0)

        assert pipe._boundary_id == boundary, 'the pipeline did not resolve <|end_of_text|>'
        assert pipe._boundary_id != pipe._eos_id, 'the document boundary must not equal the eos used for generation'

        records = [[(i * 5 + j) % 97 + 20 for j in range(1 + (i % 5))] for i in range(60)]
        pipe.train(ListDataset(records), num_epochs=1, steps_per_epoch=2)
        assert len(pipe.seen) == 2

        masked = sum(int((labels == -100).sum()) for _, labels in pipe.seen)
        in_inputs = sum(int((inputs == boundary).sum()) for inputs, _ in pipe.seen)
        in_labels = sum(int((labels == boundary).sum()) for _, labels in pipe.seen)
        assert in_labels == 0, 'with default masking the boundary token must not appear in labels'
        assert masked >= 2, 'the boundary token position should be written as -100'
        assert in_inputs >= 2, 'the boundary token must stay in input_ids (the boundary signal must not be lost)'

        other = Recording(model, tokenizer, PretrainConfig(
            base_context=4, target_context=8, global_batch_tokens=16, step_mode='min',
            ckpt_dir=os.path.join(base, 'x'), save_every_steps=0, probe_every_steps=0,
            use_progress=False, amp=False, boundary_token_id=tokenizer.token_to_id('<|eos|>')),
            device='cpu', seed=0)
        assert other._boundary_id == tokenizer.token_to_id('<|eos|>')

        unmasked = Recording(model, tokenizer, PretrainConfig(
            base_context=4, target_context=8, global_batch_tokens=16, step_mode='min',
            ckpt_dir=os.path.join(base, 'y'), save_every_steps=0, probe_every_steps=0,
            use_progress=False, amp=False, mask_boundary_token=False),
            device='cpu', seed=0)
        unmasked.train(ListDataset(records), num_epochs=1, steps_per_epoch=2)
        hits = sum(int((labels == boundary).sum()) for _, labels in unmasked.seen)
        assert hits >= 2, 'with mask_boundary_token=False the boundary token should return to labels'
        assert all(int((labels == -100).sum()) == 0 for _, labels in unmasked.seen), \
            'with the switch off there must be no -100'
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_token_specs_accept_text_not_just_ids():
    '''eos / boundary may be written as an id, a logical name, or the literal
    vocabulary text (hand-writing ids is far too error-prone).'''
    from motif.pipeline import resolve_eos_token_id, resolve_token_spec

    tokenizer = make_tokenizer()
    im_end = tokenizer.token_to_id('<|im_end|>')
    eot = tokenizer.token_to_id('<|end_of_text|>')

    assert resolve_token_spec(tokenizer, im_end, 'x') == im_end
    assert resolve_token_spec(tokenizer, '<|im_end|>', 'x') == im_end
    assert resolve_token_spec(tokenizer, 'im_end', 'x') == im_end
    assert resolve_token_spec(tokenizer, 'end_of_text', 'x') == eot

    assert resolve_eos_token_id(tokenizer, '<|im_end|>') == im_end
    assert resolve_eos_token_id(tokenizer, 'im_end') == im_end
    assert resolve_boundary_token_id(tokenizer, '<|end_of_text|>') == eot
    assert resolve_boundary_token_id(tokenizer, 'end_of_text') == eot

    try:
        resolve_token_spec(tokenizer, '<|no_such_token|>', 'boundary_token_id')
    except ValueError as exc:
        assert 'vocabulary' in str(exc)
    else:
        raise AssertionError('a token missing from the vocabulary must raise')

    assert resolve_token_spec(tokenizer, '9', 'x') == tokenizer.token_to_id('9')

    try:
        resolve_token_spec(tokenizer, 1.5, 'eos_token_id')
    except TypeError as exc:
        assert 'int' in str(exc) and 'str' in str(exc)
    else:
        raise AssertionError('an invalid type must raise')


def test_pipeline_accepts_token_text_in_config():
    '''`PretrainConfig(eos_token_id='<|im_end|>', boundary_token_id='<|end_of_text|>')` works.'''
    tokenizer = make_tokenizer()
    eot = tokenizer.token_to_id('<|end_of_text|>')

    class Rec(PretrainPipeline):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.seen = []

        def compute_loss(self, batch):
            stage, inputs, labels = batch
            self.seen.append((inputs.clone(), labels.clone()))
            return super().compute_loss(batch)

    base = make_ckpt_dir()
    try:
        torch.manual_seed(0)
        model = MotifChordModel(MotifChordConfig(
            vocab_size=tokenizer.vocab_size, model_dim=32, num_heads=4, num_kv_heads=2,
            head_k_dim=8, block=('gdn', 'chord'), block_n=2))
        pipe = Rec(model, tokenizer, PretrainConfig(
            base_context=4, target_context=8, global_batch_tokens=16, step_mode='min',
            learning_rate=1e-3, warmup_steps=1, ckpt_dir=base, save_every_steps=0,
            probe_every_steps=0, use_progress=False, amp=False,
            eos_token_id='<|im_end|>', boundary_token_id='<|end_of_text|>'),
            device='cpu', seed=0)

        assert pipe._eos_id == tokenizer.token_to_id('<|im_end|>'), 'the text eos was not resolved correctly'
        assert pipe._boundary_id == eot, 'the text boundary was not resolved correctly'

        records = [[(i * 5 + j) % 97 + 20 for j in range(1 + (i % 5))] for i in range(48)]
        pipe.train(ListDataset(records), num_epochs=1, steps_per_epoch=2)
        masked = sum(int((labels == -100).sum()) for _, labels in pipe.seen)
        in_inputs = sum(int((inputs == eot).sum()) for inputs, _ in pipe.seen)
        assert masked >= 2, 'the text boundary was not packed into the data stream (no -100 in labels)'
        assert in_inputs >= 2, 'the text boundary did not appear in input_ids'
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            print(f'--- {name}')
            fn()
    print('all boundary token tests passed')
