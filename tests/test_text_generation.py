import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn

from motif.model.generate import Generation, TextGeneration
from motif.pipeline.probe import generate_reply


class BytePieceTokenizer:
    '''
    Miniature stand-in for byte-level BPE: every id is one run of bytes, and decode
    splices the bytes together and reads them back as UTF-8 (`errors='replace'`), so
    half a character turns into `\\ufffd` — exactly what a real tokenizer does while
    streaming.
    '''

    def __init__(self, pieces=None, eos='<|eos|>', eos_id=99):
        self.pieces = dict(pieces or {
            1: b'\xe4',
            2: b'\xbd',
            3: b'\xa0',
            10: '你'.encode('utf-8'),
            11: '好'.encode('utf-8'),
            eos_id: eos.encode('utf-8'),
        })
        self._reverse = {}
        for token_id, raw in self.pieces.items():
            self._reverse.setdefault(raw, token_id)
        self.eos = eos
        self.eos_id = eos_id

    @property
    def token_eos(self) -> str:
        return self.eos

    def token_to_id(self, token):
        return self.eos_id if token == self.eos else None

    def encode(self, text, add_special_tokens=False, apply_safe_rule=True):
        return [self._reverse[char.encode('utf-8')] for char in text]

    def decode(self, ids, skip_special_tokens=False):
        ids = [int(i) for i in ids]
        if skip_special_tokens:
            ids = [i for i in ids if i != self.eos_id]
        raw = b''.join(self.pieces[i] for i in ids)
        return raw.decode('utf-8', errors='replace')


class PlainDecodeTokenizer(BytePieceTokenizer):
    '''Tokenizer whose decode takes no skip_special_tokens; exercises the fallback branch.'''

    def decode(self, ids):
        return super().decode(ids, skip_special_tokens=False)


class _ScriptedOutput:
    '''Minimal stand-in for `ModelOutput`: next_token returns the token the script planned.'''

    def __init__(self, token: int, cache):
        self.token = token
        self.cache = cache

    def next_token(self, temperature=1.0, sampler=None, input_ids=None) -> torch.Tensor:
        return torch.tensor([[self.token]], dtype=torch.long)


class ScriptedModel:
    '''
    Fake model that emits the scripted tokens and honours the
    `model(input_ids, cache=...)` -> `.next_token(...)` contract `Generation` needs;
    it also carries a mini-motif shaped `generate`, so that
    `probe.build_generate_kwargs` recognises the "sampler but no temperature" path.
    '''

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.training = True
        self.sampler = None

    def __call__(self, input_ids, cache=None, mask=None):
        token = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return _ScriptedOutput(token, cache)

    def generate(self, input_ids, max_new_tokens=32, eos_token_id=None, sampler=None,
                 cache=None):
        raise AssertionError('the mini-motif branch must not call model.generate')

    def parameters(self):
        '''TextGeneration reads the device off the parameters; this fake has none.'''
        return iter(())

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self


class CodonModel:
    '''Codon style fake model whose generate returns a token tensor directly.'''

    def __init__(self, out_ids):
        self.out_ids = list(out_ids)
        self.training = True
        self.seen = {}

    def generate(self, input_ids, max_new_tokens=32, temperature=0.8, eos_token_id=None):
        self.seen = dict(max_new_tokens=max_new_tokens, temperature=temperature,
                         eos_token_id=eos_token_id)
        return torch.tensor([self.out_ids], dtype=torch.long)

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self


def make_text_gen(script, prompt='你好', **kwargs) -> TextGeneration:
    return TextGeneration(ScriptedModel(script), BytePieceTokenizer(), prompt, **kwargs)


def test_half_decoded_character_is_held_back():
    '''With one character's three bytes arriving in three steps, the first two steps must not emit \\ufffd; only the third emits that character.'''
    tokenizer = BytePieceTokenizer()
    gen = TextGeneration(ScriptedModel([1, 2, 3]), tokenizer, '你好', max_new_tokens=3)

    assert '\ufffd' in tokenizer.decode([1, 2])

    chunks = list(gen)
    assert chunks == ['', '', '你'], chunks
    assert ''.join(chunks) == gen.text == '你'
    assert '\ufffd' not in gen.text


def test_include_prompt_also_holds_back_a_half_character():
    '''The same holds with a prompt: the prompt occupies _emitted first, so a half character must not add an \\ufffd.'''
    gen = make_text_gen([1, 2, 3], max_new_tokens=3, include_prompt=True)
    assert gen.text == '你好'
    chunks = list(gen)
    assert chunks == ['', '', '你'], chunks
    assert gen.text == '你好你'


def test_stream_matches_text_at_every_step():
    '''After every step the chunks joined together in text must match a one-shot decode.'''
    gen = make_text_gen([1, 2, 3, 10, 11], max_new_tokens=5)
    emitted = ''
    for chunk in gen:
        emitted += chunk
        assert emitted == gen.text
        assert '\ufffd' not in gen.text
    assert gen.text == '你你好'


def test_text_is_empty_until_it_runs():
    '''Before it runs, text is the empty string.'''
    gen = make_text_gen([10], max_new_tokens=1)
    assert gen.text == ''
    assert gen.steps == 0


def test_zero_max_new_tokens_generates_nothing():
    gen = make_text_gen([10], max_new_tokens=0)
    assert list(gen) == []
    assert gen.text == ''


def test_text_is_the_continuation_by_default():
    gen = make_text_gen([10], max_new_tokens=1)
    gen.run()
    assert gen.prompt_text == '你好'
    assert gen.text == '你'


def test_include_prompt_puts_the_prompt_in_front():
    gen = make_text_gen([10], max_new_tokens=1, include_prompt=True)
    assert gen.text == '你好', 'before anything is generated text is the prompt'
    chunks = list(gen)
    assert chunks == ['你'], 'the prompt must not be emitted again as new text'
    assert gen.text == '你好你'


def test_prompt_can_be_token_ids():
    ids = torch.tensor([[10, 11]])
    gen = TextGeneration(ScriptedModel([10]), BytePieceTokenizer(), ids, max_new_tokens=1)
    assert gen.prompt_text == '你好'
    gen.run()
    assert gen.text == '你'


def test_batch_prompt_is_rejected():
    ids = torch.tensor([[10], [11]])
    try:
        TextGeneration(ScriptedModel([10]), BytePieceTokenizer(), ids)
    except ValueError as exc:
        assert '[1, T]' in str(exc)
    else:
        raise AssertionError('a multi-row prompt did not raise')


def test_eos_is_taken_from_the_tokenizer():
    '''With no eos given the tokenizer's is used: the 11 in the script must never be generated.'''
    gen = make_text_gen([1, 2, 3, 99, 11], max_new_tokens=5, skip_special_tokens=False)
    gen.run()
    assert gen.eos_ids == {99}
    assert gen.steps == 4
    assert gen.generated[0].tolist() == [1, 2, 3, 99]
    assert gen.text == '你<|eos|>'


def test_skip_special_tokens_drops_the_eos_text():
    gen = make_text_gen([1, 2, 3, 99, 11], max_new_tokens=5, skip_special_tokens=True)
    gen.run()
    assert gen.text == '你'


def test_empty_eos_disables_early_stopping():
    '''eos_token_id=() states clearly that there is no early stop, so the run completes even after 99 appears.'''
    gen = make_text_gen([99, 10], max_new_tokens=2, eos_token_id=(),
                        skip_special_tokens=False)
    gen.run()
    assert gen.steps == 2
    assert gen.text == '<|eos|>你'


def test_decode_without_skip_special_tokens_argument():
    '''Must not blow up when the tokenizer decode does not accept skip_special_tokens.'''
    gen = TextGeneration(ScriptedModel([10, 99]), PlainDecodeTokenizer(), '你好',
                         max_new_tokens=2, skip_special_tokens=True)
    gen.run()
    assert gen.text == '你<|eos|>'


def test_generate_reply_uses_text_generation_for_mini_motif():
    '''The mini-motif branch goes through TextGeneration: the prompt is included, eos stops it, and no half character is emitted.'''
    model = ScriptedModel([1, 2, 3, 99, 11])
    text = generate_reply(model, BytePieceTokenizer(), '你好', torch.device('cpu'),
                          max_new_tokens=5)
    assert text == '你好你<|eos|>', text
    assert model.calls == 4, 'it should stop on the tokenizer eos'
    assert model.training is True, 'the eval state must be restored'


def test_generate_reply_keeps_the_codon_style_path():
    '''Codon style models still take the old generate + decode path.'''
    model = CodonModel([10, 11])
    text = generate_reply(model, BytePieceTokenizer(), '你好', torch.device('cpu'),
                          max_new_tokens=7, temperature=1.3, eos_token_id=2)
    assert text == '你好'
    assert model.seen == dict(max_new_tokens=7, temperature=1.3, eos_token_id=2)


def test_text_prompt_follows_the_model_device():
    '''A text prompt is encoded onto the model's device, and a tensor prompt on another one is refused with the fix in the message.'''
    if not torch.cuda.is_available():
        pytest.skip('needs a CUDA device to have a second device to be wrong about')
    model = nn.Linear(4, 4).to('cuda')
    gen = TextGeneration(model, BytePieceTokenizer(), '你好', max_new_tokens=1)
    assert gen.sequences.device.type == 'cuda'

    with pytest.raises(ValueError) as excinfo:
        TextGeneration(model, BytePieceTokenizer(), torch.tensor([[10]]))
    message = str(excinfo.value)
    assert 'cpu' in message and 'cuda' in message
    assert '.to(cuda' in message, 'the message has to say how to fix it'


def test_eos_can_be_spelled_as_a_token_name():
    '''`tokenizer.token_eos` is a name, so Generation has to resolve a name through the tokenizer.'''
    gen = TextGeneration(ScriptedModel([1, 2, 99, 11]), BytePieceTokenizer(), '你好',
                         max_new_tokens=4, eos_token_id='<|eos|>', skip_special_tokens=False)
    assert gen.eos_ids == {99}
    gen.run()
    assert gen.steps == 3, 'the named eos has to stop the run'

    # A name cannot be resolved without a tokenizer, and the error has to say so.
    from motif.model.generate import _resolve_eos

    with pytest.raises(ValueError) as excinfo:
        _resolve_eos('<|eos|>')
    assert '<|eos|>' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        TextGeneration(ScriptedModel([1]), BytePieceTokenizer(), '你好',
                       eos_token_id='<|unknown|>')
    assert 'does not know' in str(excinfo.value)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'[OK] {name}')
