import json
import math
import os
import shutil
import tempfile
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from motif.data import MotifSFT
from motif.pipeline import SFTConfig, SFTPipeline
from motif.pipeline.base import select_model_kwargs
from motif.pipeline.callback import Callback
from motif.pipeline.sanity import (
    IGNORE_INDEX,
    ProbeSample,
    SanityProbe,
    _perplexity,
    _pick_indices,
    _repeats,
    build_probe_samples,
)
from support import make_model, make_tokenizer


def _chat(index: int, turns: int = 1) -> dict:
    '''A ready-made conversation with `turns` user/model turns.'''
    messages = []
    for turn in range(turns):
        messages.append({'role': 'human', 'content': f'question {index} turn {turn}'})
        messages.append({'role': 'model', 'content': f'answer {index} turn {turn}'})
    return {'conversations': messages}


def _folder(rows, name: str = 'part.jsonl') -> str:
    folder = tempfile.mkdtemp(prefix='motif_sanity_')
    with open(os.path.join(folder, name), 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    return folder


def _dataset(rows, tokenizer=None, **kwargs) -> MotifSFT:
    tokenizer = tokenizer or make_tokenizer()
    kwargs.setdefault('pad_length', 64)
    kwargs.setdefault('batch_size', 2)
    kwargs.setdefault('seed', 7)
    return MotifSFT(_folder(rows), tokenizer, **kwargs)


class FakePipeline:
    '''The probe uses only these attributes, so a fake replaces the whole training pipeline.'''

    def __init__(self, model, tokenizer, step: int = 0):
        self.model = model
        self.raw_model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cpu')
        self.state = SimpleNamespace(global_step=step)


class Recorder(Callback):
    '''Records the step_metrics seen at each step (to verify the probe runs before monitors).'''

    def __init__(self):
        self.steps = []

    def on_step_end(self, pipeline, step_metrics):
        self.steps.append(dict(step_metrics))


class StopAfter(Callback):
    '''Requests a stop after n optimizer steps.'''

    def __init__(self, n):
        self.n = n

    def on_step_end(self, pipeline, step_metrics):
        if pipeline.state.global_step >= self.n:
            self.stop_requested = True


def _sft_pipeline(folder: str, ckpt_dir: str, tokenizer, callbacks=None, **overrides) -> SFTPipeline:
    kwargs = dict(
        compiled=False,
        learning_rate=1e-3,
        warmup_steps=1,
        ckpt_dir=ckpt_dir,
        save_every_steps=0,
        auto_resume=False,
        use_progress=False,
        amp=False,
        stage_specs=[{'name': 'tiny', 'folder': folder, 'epochs': 1}],
        pad_length=96,
        batch_size=2,
        probe_every_steps=1,
        probe_max_new_tokens=4,
        sanity_samples=1,
    )
    kwargs.update(overrides)
    return SFTPipeline(
        make_model(vocab_size=tokenizer.vocab_size), tokenizer, SFTConfig(**kwargs),
        callbacks=callbacks if callbacks is not None else [],
        seed=0, device='cpu',
    )


def test_samples_hold_out_the_last_model_turn():
    '''The last model turn becomes the answer and the prompt is exactly its prefix.'''
    dataset = _dataset([_chat(0), _chat(1), _chat(2)])
    samples = build_probe_samples([dataset], 2)

    assert [sample.name for sample in samples] == ['#0', '#2'], [s.name for s in samples]
    for sample in samples:
        group = dataset.groups[int(sample.name[1:])]
        messages = dataset.build_messages(group)
        assert sample.answer == messages[-1]['content'], sample.answer
        assert sample.prompt_ids.numel() > 0
        head = sample.input_ids[:sample.prompt_ids.numel()]
        assert torch.equal(sample.prompt_ids, head), 'prompt is not a rendering prefix'
        assert sample.prompt_ids[-1].item() == make_tokenizer().token_to_id('<|model|>'), (
            'prompt should stop at <|im_start|><|model|>'
        )


def test_samples_spread_evenly_over_the_corpus():
    '''Samples are taken evenly over the whole corpus, not just the first two rows.'''
    dataset = _dataset([_chat(index) for index in range(6)])
    assert [sample.name for sample in build_probe_samples([dataset], 3)] == ['#0', '#2', '#4']
    assert [sample.name for sample in build_probe_samples([dataset], 6)] == [
        f'#{index}' for index in range(6)
    ]
    assert len(build_probe_samples([dataset], 99)) == 6


def test_single_model_turn_row_is_probed_as_a_continuation():
    '''A conversation with one model turn has no answer to hold out, so it is continued.'''
    dataset = _dataset([{'conversations': [{'role': 'model', 'content': 'hello there'}]}])
    sample = build_probe_samples([dataset], 1)[0]
    assert sample.answer == ''
    assert sample.prompt_ids.numel() > 0


def test_long_prompt_is_truncated_to_pad_length():
    '''A prompt longer than pad_length is truncated to pad_length without overrunning.'''
    long_row = {'conversations': [
        {'role': 'human', 'content': ' '.join(f'word{index}' for index in range(200))},
        {'role': 'model', 'content': 'done'},
    ]}
    dataset = _dataset([long_row], pad_length=16)
    sample = build_probe_samples([dataset], 1)[0]
    assert sample.prompt_ids.numel() <= 16, sample.prompt_ids.numel()


def test_samples_over_several_stages_are_kept_apart():
    '''With several stages the samples come from all corpora and names carry a stage prefix.'''
    first = _dataset([_chat(0)])
    second = _dataset([_chat(1)])
    samples = build_probe_samples([first, second], 2)
    assert sorted(sample.name for sample in samples) == ['stage0#0', 'stage1#0']


def test_pick_indices_is_deterministic_and_clamped():
    assert _pick_indices(10, 3) == [0, 3, 7]
    assert _pick_indices(10, 1) == [0]
    assert _pick_indices(2, 5) == [0, 1]
    assert _pick_indices(1, 1) == [0]


def test_perplexity_and_repeat_detector():
    assert abs(_perplexity(0.0) - 1.0) < 1e-9
    assert abs(_perplexity(math.log(4.0)) - 4.0) < 1e-6
    assert _perplexity(float('nan')) == float('inf')
    assert _perplexity(1000.0) == float('inf')

    assert _repeats([1, 2, 3, 4] * 5), 'an obviously repeated 4-gram should be detected'
    assert not _repeats(list(range(30))), 'mutually different tokens do not count as repeats'
    assert not _repeats([1, 2, 3]), 'too short to count as a repeat'


def test_bad_build_inputs_raise():
    tokenizer = make_tokenizer()
    dataset = _dataset([_chat(0)])
    try:
        build_probe_samples([dataset], 0)
    except ValueError as exc:
        assert 'must be positive' in str(exc), str(exc)
    else:
        raise AssertionError('count=0 should raise ValueError')

    for bad in (SimpleNamespace(groups=[{'turns': []}]),):
        try:
            build_probe_samples([bad], 1)
        except ValueError as exc:
            assert 'MotifSFT' in str(exc), str(exc)
        else:
            raise AssertionError('a dataset without the rendering seam should raise ValueError')

    empty = SimpleNamespace(
        groups=[], build_messages=lambda group: [], render_group=lambda group: {},
        session_for=lambda messages: None, pad_length=8,
    )
    try:
        build_probe_samples([empty], 1)
    except ValueError as exc:
        assert 'no conversation group' in str(exc), str(exc)
    else:
        raise AssertionError('an empty corpus should raise ValueError')

    blank = SimpleNamespace(
        groups=[{}], build_messages=lambda group: [], render_group=lambda group: {},
        session_for=lambda messages: None, pad_length=8,
    )
    try:
        build_probe_samples([blank], 1)
    except ValueError as exc:
        assert 'no usable conversation' in str(exc), str(exc)
    else:
        raise AssertionError('a dataset that renders no messages should raise ValueError')
    assert tokenizer is not None


def test_report_scores_only_the_supervised_positions():
    tokenizer = make_tokenizer()
    dataset = _dataset([_chat(0), _chat(1)])
    samples = build_probe_samples([dataset], 2)
    model = make_model(vocab_size=tokenizer.vocab_size)
    probe = SanityProbe(samples, tokenizer, generate=False, every_steps=1)

    metrics = probe._report(FakePipeline(model, tokenizer, step=7))

    assert set(metrics) == {
        'sanity/loss', 'sanity/ppl', 'sanity/acc', 'sanity/supervised', 'sanity/warnings',
    }, metrics
    supervised = sum(int((sample.labels[1:] != IGNORE_INDEX).sum()) for sample in samples)
    assert metrics['sanity/supervised'] == float(supervised)
    assert 0.0 <= metrics['sanity/acc'] <= 1.0
    assert metrics['sanity/loss'] > 0

    model.eval()
    with torch.no_grad():
        losses = []
        for sample in samples:
            batch = {
                'input_ids': sample.input_ids.unsqueeze(0),
                'attention_mask': sample.attention_mask.unsqueeze(0),
            }
            kwargs = select_model_kwargs(model, batch)
            logits = model(**kwargs).logits.float()[:, :-1, :][0]
            labels = sample.labels[1:]
            mask = labels != IGNORE_INDEX
            losses.append(float(F.cross_entropy(logits[mask], labels[mask])))
    expected = sum(losses) / len(losses)
    assert abs(metrics['sanity/loss'] - expected) < 1e-5, (metrics['sanity/loss'], expected)
    assert abs(metrics['sanity/ppl'] - math.exp(expected)) < 1e-3


def test_supervision_matches_the_training_labels():
    '''The tensors the probe sees equal row 0 of a training batch, so the counts agree.'''
    tokenizer = make_tokenizer()
    dataset = _dataset([_chat(0), _chat(1)])
    batch = dataset[0]
    samples = build_probe_samples([dataset], 1)
    model = make_model(vocab_size=tokenizer.vocab_size)
    probe = SanityProbe(samples, tokenizer, generate=False)

    metrics = probe._report(FakePipeline(model, tokenizer))
    assert torch.equal(batch['labels'][0], samples[0].labels), 'the probe uses the training tensors'
    assert torch.equal(batch['input_ids'][0], samples[0].input_ids)
    assert metrics['sanity/supervised'] == float(samples[0].labels[1:].ne(IGNORE_INDEX).sum())


def test_baseline_is_the_first_probe():
    tokenizer = make_tokenizer()
    samples = build_probe_samples([_dataset([_chat(0), _chat(1)])], 2)
    probe = SanityProbe(samples, tokenizer, generate=False)
    pipeline = FakePipeline(make_model(vocab_size=tokenizer.vocab_size), tokenizer)

    assert probe.baseline is None
    first = probe._report(pipeline)
    assert probe.baseline == first['sanity/loss']
    probe._report(pipeline)
    assert probe.baseline == first['sanity/loss'], 'the baseline is recorded only on the first probe'


def test_loss_regression_only_warns():
    tokenizer = make_tokenizer()
    samples = build_probe_samples([_dataset([_chat(0), _chat(1)])], 1)
    probe = SanityProbe(samples, tokenizer, generate=False, warn_ratio=2.0)
    probe.baseline = 1e-6

    metrics = probe._report(FakePipeline(make_model(vocab_size=tokenizer.vocab_size), tokenizer))
    assert metrics['sanity/warnings'] >= 1.0, metrics


def test_a_sample_without_supervision_is_warned():
    '''A sample with no supervised token at all only warns and skips the loss average.'''
    tokenizer = make_tokenizer()
    sample = ProbeSample(
        name='#0',
        input_ids=torch.tensor([1, 2, 3, 4]),
        labels=torch.full((4,), IGNORE_INDEX),
        attention_mask=torch.ones(4, dtype=torch.long),
        prompt_ids=torch.tensor([1, 2, 3, 4]),
    )
    probe = SanityProbe([sample], tokenizer, generate=False)
    metrics = probe._report(FakePipeline(make_model(vocab_size=tokenizer.vocab_size), tokenizer))

    assert metrics['sanity/warnings'] == 2.0, metrics
    assert 'sanity/loss' not in metrics, metrics


def test_probe_without_tokenizer_does_nothing():
    tokenizer = make_tokenizer()
    samples = build_probe_samples([_dataset([_chat(0)])], 1)
    probe = SanityProbe(samples, tokenizer, generate=False)
    assert probe._report(FakePipeline(make_model(vocab_size=tokenizer.vocab_size), None)) == {}


def test_probe_needs_at_least_one_sample():
    try:
        SanityProbe([], make_tokenizer())
    except ValueError as exc:
        assert 'at least one probe sample' in str(exc), str(exc)
    else:
        raise AssertionError('an empty sample list should raise ValueError')


def test_generation_text_reaches_the_sink():
    '''Generation uses the training format and the text reaches `Sanity/sample_<i>`.'''
    tokenizer = make_tokenizer()
    samples = build_probe_samples([_dataset([_chat(0)])], 1)
    written = []
    probe = SanityProbe(
        samples, tokenizer, max_new_tokens=4,
        sink=lambda tag, text, step: written.append((tag, text, step)),
    )
    probe._report(FakePipeline(make_model(vocab_size=tokenizer.vocab_size), tokenizer, step=3))

    assert len(written) == 1, written
    tag, text, step = written[0]
    assert tag == 'Sanity/sample_0', tag
    assert isinstance(text, str)
    assert step == 3


def test_sftpipeline_attaches_the_probe_before_the_monitors():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_sanity_pipe_')
    try:
        folder = _folder([_chat(index) for index in range(4)])
        recorder = Recorder()
        pipe = _sft_pipeline(
            folder, os.path.join(base, 'ckpt'), tokenizer,
            callbacks=[recorder, StopAfter(2)], sanity_samples=2,
        )
        assert isinstance(pipe.callbacks[0], SanityProbe), 'the probe must come first'
        probe = pipe.callbacks[0]
        assert [sample.name for sample in probe.samples] == ['#0', '#2']

        metrics = pipe.train()
        assert pipe.state.global_step == 2
        assert 'loss/train' in metrics
        assert recorder.steps, 'the callback did not receive step_metrics'
        assert 'sanity/loss' in recorder.steps[0], (
            f'the same-step metrics are missed when the probe follows the monitors: '
            f'{sorted(recorder.steps[0])}'
        )
        assert probe.baseline is not None, 'the probe should have run once at the start'
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_sanity_samples_are_optional():
    '''The default sanity_samples=0 builds no probe and requires nothing extra.'''
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_sanity_off_')
    try:
        folder = _folder([_chat(index) for index in range(4)])
        pipe = _sft_pipeline(
            folder, os.path.join(base, 'ckpt'), tokenizer,
            callbacks=[StopAfter(1)], sanity_samples=0,
        )
        assert not any(isinstance(callback, SanityProbe) for callback in pipe.callbacks)
        pipe.train()
        assert pipe.state.global_step == 1
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_the_probe_survives_a_resume():
    '''The probe keeps no state: on resume it is rebuilt in the same order and re-baselines.'''
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_sanity_resume_')
    try:
        folder = _folder([_chat(index) for index in range(4)])
        ckpt_dir = os.path.join(base, 'ckpt')
        first = _sft_pipeline(folder, ckpt_dir, tokenizer, callbacks=[StopAfter(1)],
                              sanity_samples=1)
        first.train()
        assert first.state.global_step == 1
        assert os.path.exists(os.path.join(ckpt_dir, 'last.pt'))

        recorder = Recorder()
        resumed = _sft_pipeline(folder, ckpt_dir, tokenizer,
                                callbacks=[recorder, StopAfter(2)], sanity_samples=1,
                                auto_resume=True)
        assert isinstance(resumed.callbacks[0], SanityProbe)
        assert resumed.callbacks[0].baseline is None, 'the new probe starts from scratch'
        resumed.train()
        assert resumed.state.global_step == 2, 'resumed training continues the original steps'
        assert resumed.callbacks[0].baseline is not None, 'the probe works normally after resuming'
        assert 'sanity/loss' in recorder.steps[-1], sorted(recorder.steps[-1])
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_bad_sanity_settings_are_rejected():
    tokenizer = make_tokenizer()
    base = tempfile.mkdtemp(prefix='motif_sanity_bad_')
    try:
        folder = _folder([_chat(0)])
        for overrides, fragment in (
            ({'sanity_samples': -1}, 'must be >= 0'),
            ({'sanity_samples': 1, 'probe_every_steps': 0}, 'probe_every_steps'),
        ):
            try:
                _sft_pipeline(folder, os.path.join(base, 'ckpt'), tokenizer, **overrides)
            except ValueError as exc:
                assert fragment in str(exc), str(exc)
            else:
                raise AssertionError(f'{overrides} should raise ValueError')
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == '__main__':
    for name, function in sorted(globals().items()):
        if name.startswith('test_') and callable(function):
            print(f'--- {name}')
            function()
    print('all sanity probe tests passed')
