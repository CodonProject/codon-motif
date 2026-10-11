import pytest
import torch

from motif.model.generate import Generation
from motif.model.motif import ModelOutput
from motif.model.sampler import Sampler, make_sampler

from support import make_model


VOCAB = 64
BLOCK = ('gdn', 'chord')
BLOCK_N = 2


def decode_model():
    return make_model(vocab_size=VOCAB, model_dim=64, head_k_dim=16, block=BLOCK,
                      block_n=BLOCK_N).eval()


def assert_raises(exc_type, call) -> None:
    with pytest.raises(exc_type):
        call()


class ForceEos:
    '''Let the first `after` steps through, then force the token to eos.'''

    def __init__(self, eos_token_id: int, after: int = 2):
        self.eos_token_id = eos_token_id
        self.after = after
        self.calls = 0

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def apply_forcing(self, tokens: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls > self.after:
            return torch.full_like(tokens, self.eos_token_id)
        return tokens

    def advance(self, tokens: torch.Tensor) -> None:
        pass


class BlockTokenZero:
    '''Disables token 0 and counts the forcing / advance calls.'''

    def __init__(self, eos_token_id: int = 5):
        self.eos_token_id = eos_token_id
        self.forced = 0
        self.advanced = []

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        logits[..., 0] = float('-inf')
        return logits

    def apply_forcing(self, tokens: torch.Tensor) -> torch.Tensor:
        self.forced += 1
        return tokens

    def advance(self, tokens: torch.Tensor) -> None:
        self.advanced.append(tokens.clone())


def test_generate_returns_lazy_generator():
    model = decode_model()
    ids = torch.randint(0, VOCAB, (1, 5))

    gen = model.generate(ids, max_new_tokens=3)
    assert isinstance(gen, Generation)
    assert iter(gen) is gen, 'it is its own iterator'
    assert gen.steps == 0 and gen.cache is None, 'nothing runs before iteration starts'

    tokens = list(gen)
    assert len(tokens) == 3
    assert all(t.shape == (1, 1) for t in tokens)
    assert gen.steps == 3 and gen.done
    assert torch.equal(torch.cat(tokens, dim=1), gen.generated)
    assert gen.sequences.shape == (1, 8)
    assert torch.equal(gen.sequences[:, :5], ids)
    assert gen.cache is not None
    assert next(gen, None) is None, 'calling next after it finished should raise StopIteration'


def test_generate_matches_manual_decode():
    model = decode_model()
    ids = torch.randint(0, VOCAB, (2, 5))

    gen = model.generate(ids, max_new_tokens=4).run()
    assert gen.generated.shape == (2, 4)
    assert gen.sequences.shape == (2, 9)
    assert gen.finished.shape == (2,) and not gen.finished.any()

    with torch.no_grad():
        hold = model(ids)
        tokens = [hold.next_token()]
        for _ in range(3):
            hold = model(tokens[-1], cache=hold.cache)
            tokens.append(hold.next_token())
    assert torch.equal(torch.cat(tokens, dim=1), gen.generated)

    empty = model.generate(ids, max_new_tokens=0).run()
    assert empty.generated.shape == (2, 0) and empty.steps == 0


def test_generate_stops_at_eos():
    model = decode_model()
    ids = torch.randint(0, VOCAB, (1, 4))
    eos = 7

    constraint = ForceEos(eos, after=2)
    sampler = Sampler(temperature=1.0, repetition_penalty=1.0).with_constraint(
        constraint, eos_token_id=eos,
    )
    gen = model.generate(ids, max_new_tokens=10, eos_token_id=eos, sampler=sampler).run()

    assert gen.steps == 3, 'it stops at eos and must not run the full max_new_tokens'
    assert bool(gen.finished.all())
    assert gen.generated[0, -1].item() == eos
    assert gen.generated.shape[1] == 3


def test_cache_continuation():
    model = decode_model()
    ids = torch.randint(0, VOCAB, (1, 5))

    full = model.generate(ids, max_new_tokens=6).run()
    head = model.generate(ids, max_new_tokens=3).run()

    assert head.cache.seq_len == 5 + 2
    tail = model.generate(head.sequences, max_new_tokens=3, cache=head.cache).run()

    assert torch.equal(tail.sequences, full.sequences)
    assert torch.equal(torch.cat([head.generated, tail.generated], dim=1), full.generated)
    assert tail.generated.shape[1] == 3

    cache = model.create_cache()
    chunks = list(model.generate(ids, max_new_tokens=3, cache=cache))
    assert cache.seq_len == 5 + 2
    assert torch.equal(torch.cat(chunks, dim=1), full.generated[:, :3])

    again = model.generate(torch.cat([ids] + chunks, dim=1), max_new_tokens=3, cache=cache).run()
    assert torch.equal(again.generated, full.generated[:, 3:])


def test_model_level_sampler_and_errors():
    model = decode_model()
    ids = torch.randint(0, VOCAB, (1, 5))

    greedy = model.generate(ids, max_new_tokens=3).run().generated
    model.set_sampler(temperature=1.0, top_k=1, repetition_penalty=1.0)
    assert torch.equal(model.generate(ids, max_new_tokens=3).run().generated, greedy)

    penalized = model.generate(
        ids, max_new_tokens=3, sampler=Sampler(temperature=1.0, repetition_penalty=2.0),
    ).run()
    assert penalized.generated.shape == (1, 3)

    cached = model.generate(ids, max_new_tokens=2).run().cache
    for bad in (
        lambda: model.generate(ids.unsqueeze(0), max_new_tokens=1),
        lambda: model.generate(ids, max_new_tokens=-1),
        lambda: model.generate(ids, max_new_tokens=1, cache=cached),
    ):
        assert_raises(ValueError, bad)


def test_top_k_and_top_p_collapse_to_greedy():
    torch.manual_seed(0)
    logits = torch.randn(4, 32)

    top_1 = Sampler(temperature=1.0, top_k=1, repetition_penalty=1.0)(logits.clone())
    assert torch.equal(top_1, logits.argmax(dim=-1, keepdim=True))

    tiny_p = Sampler(temperature=1.0, top_p=1e-6, repetition_penalty=1.0)(logits.clone())
    assert torch.equal(tiny_p, logits.argmax(dim=-1, keepdim=True))


def test_repetition_penalty_does_not_mutate_input():
    logits = torch.zeros(1, 8)
    logits[0, 3] = 2.0
    before = logits.clone()

    Sampler(temperature=1.0, repetition_penalty=2.0)(logits, input_ids=torch.tensor([[3, 3]]))

    assert torch.equal(logits, before)


def test_constraint_is_applied_before_sampling():
    constraint = BlockTokenZero()
    sampler = Sampler(temperature=1.0, repetition_penalty=1.0).with_constraint(
        constraint, eos_token_id=7,
    )
    assert sampler.constraint is constraint
    assert constraint.eos_token_id == 7

    logits = torch.zeros(4, 8)
    logits[..., 0] = 100.0
    picked = sampler(logits)

    assert (picked != 0).all(), 'a token banned by the constraint must not be sampled'
    assert constraint.forced == 1, 'apply_forcing must be called once after sampling'
    assert len(constraint.advanced) == 1, 'advance must record the tokens of this step'

    unbound = sampler.with_constraint(None)
    assert unbound.constraint is None
    assert sampler.constraint is constraint, 'with_constraint must not mutate the sampler itself'


def test_model_output_next_token_uses_sampler():
    model = decode_model()
    ids = torch.randint(0, VOCAB, (2, 8))

    with torch.no_grad():
        out = model(ids)

    greedy = out.next_token()
    assert greedy.shape == (2, 1)
    assert torch.equal(greedy, out.last_logits.argmax(dim=-1, keepdim=True))

    top_1 = out.next_token(sampler=Sampler(temperature=1.0, top_k=1, repetition_penalty=1.0))
    assert torch.equal(top_1, greedy)

    before = out.logits.clone()
    picked = out.next_token(
        sampler=Sampler(temperature=1.0, repetition_penalty=2.0), input_ids=ids,
    )
    assert picked.shape == (2, 1)
    assert torch.equal(out.logits, before)

    assert ModelOutput(hidden_states=torch.zeros(1, 1, 4)).next_token() is None


def test_model_output_set_sampler():
    model = decode_model()
    ids = torch.randint(0, VOCAB, (2, 8))

    with torch.no_grad():
        out = model(ids)

    greedy = out.next_token()
    assert out.sampler is None, 'no built-in sampler by default, next_token stays greedy'

    assert out.set_sampler(temperature=1.0, top_k=1, repetition_penalty=1.0) is out
    assert isinstance(out.sampler, Sampler)
    assert torch.equal(out.next_token(), greedy)

    override = Sampler(temperature=1.0, top_p=1e-6, repetition_penalty=1.0)
    assert torch.equal(out.next_token(sampler=override), greedy)

    custom = Sampler(temperature=1.0, top_k=1, repetition_penalty=1.0)
    out.set_sampler(custom)
    assert out.sampler is custom
    assert torch.equal(out.next_token(), greedy)

    out.set_sampler()
    assert out.sampler is None
    assert torch.equal(out.next_token(), greedy), 'after unbinding it falls back to greedy'

    out.set_sampler(temperature=1.0, repetition_penalty=2.0)
    before = out.logits.clone()
    out.next_token(input_ids=ids)
    assert torch.equal(out.logits, before)

    assert_raises(ValueError, lambda: out.set_sampler(Sampler(), top_p=0.9))


def test_model_level_sampler_is_set_once():
    model = decode_model()
    ids = torch.randint(0, VOCAB, (2, 8))

    with torch.no_grad():
        assert model(ids).sampler is None, 'there is no model-level sampler by default'

    assert model.set_sampler(temperature=1.0, top_k=1, repetition_penalty=1.0) is model
    assert isinstance(model.sampler, Sampler)

    for _ in range(3):
        with torch.no_grad():
            step = model(ids)
        assert step.sampler is model.sampler, 'it does not need to be reset at every step'
        assert torch.equal(step.next_token(), step.last_logits.argmax(dim=-1, keepdim=True))

    one_off = model(ids)
    override = Sampler(temperature=1.0, top_p=1e-6, repetition_penalty=1.0)
    assert torch.equal(one_off.next_token(sampler=override),
                       one_off.last_logits.argmax(dim=-1, keepdim=True))
    assert one_off.next_token(sampler=override) is not None
    assert model.sampler is not None and model.sampler is not override

    model.set_sampler()
    assert model.sampler is None
    assert model(ids).sampler is None

    model.set_sampler(temperature=0.5)
    assert 'sampler' not in model.state_dict()


def test_make_sampler():
    assert make_sampler() is None

    instance = Sampler(temperature=0.5)
    assert make_sampler(instance) is instance

    built = make_sampler(top_k=5, temperature=0.5)
    assert isinstance(built, Sampler) and built.top_k == 5 and built.temperature == 0.5

    assert_raises(ValueError, lambda: make_sampler(instance, top_k=5))
