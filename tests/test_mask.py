import torch
import pytest

from motif.model.cache import ModelCache

from support import make_model


VOCAB = 64
BLOCK = ('gdn', 'gdn', 'gdn', 'chord')
BLOCK_N = 2
DIM = 64


def test_mask_must_be_two_dimensional():
    '''A mask that is not `[B, T]` is rejected.'''
    model = make_model(vocab_size=VOCAB, model_dim=DIM, head_k_dim=16, block=BLOCK, block_n=BLOCK_N).eval()
    ids = torch.randint(0, VOCAB, (2, 5))

    with pytest.raises(ValueError, match='mask must be'):
        model(ids, mask=torch.ones(5))


def test_mask_must_cover_every_key():
    '''A fresh forward pass needs one mask column per input token.'''
    model = make_model(vocab_size=VOCAB, model_dim=DIM, head_k_dim=16, block=BLOCK, block_n=BLOCK_N).eval()
    ids = torch.randint(0, VOCAB, (2, 5))

    with pytest.raises(ValueError, match='does not cover'):
        model(ids, mask=torch.ones(2, 4))


def test_mask_covers_cached_keys_plus_new_inputs():
    '''A continued forward pass needs `cached + T` columns, not `T`.'''
    model = make_model(vocab_size=VOCAB, model_dim=DIM, head_k_dim=16, block=BLOCK, block_n=BLOCK_N).eval()
    ids = torch.randint(0, VOCAB, (2, 5))

    with torch.no_grad():
        out = model(ids, mask=torch.ones(2, 5))

    step = torch.randint(0, VOCAB, (2, 1))
    with pytest.raises(ValueError, match='does not cover'):
        model(step, cache=out.cache, mask=torch.ones(2, 1))

    with torch.no_grad():
        continued = model(step, cache=out.cache, mask=torch.ones(2, 6))
    assert continued.logits.shape == (2, 1, VOCAB)


def test_mask_changes_the_output_and_none_does_not():
    '''Masking a key changes the rows that can see it; no mask changes nothing.'''
    model = make_model(vocab_size=VOCAB, model_dim=DIM, head_k_dim=16, block=BLOCK, block_n=BLOCK_N).eval()
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB, (1, 6))

    with torch.no_grad():
        plain = model(ids, cache=model.create_cache()).logits
        ones = model(ids, cache=model.create_cache(),
                     mask=torch.ones(1, 6)).logits
        blocked = model(ids, cache=model.create_cache(),
                        mask=torch.tensor([[True, True, False, True, True, True]])).logits

    assert torch.allclose(plain, ones, atol=1e-5)
    assert not torch.allclose(plain, blocked, atol=1e-5)
    assert torch.allclose(plain[:, :2], blocked[:, :2], atol=1e-5)


def test_additive_and_bool_masks_agree():
    '''A `0 / -inf` float mask is equivalent to the boolean one.'''
    model = make_model(vocab_size=VOCAB, model_dim=DIM, head_k_dim=16, block=BLOCK, block_n=BLOCK_N).eval()
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB, (1, 6))

    keep = torch.tensor([[True, True, False, False, True, True]])
    additive = torch.where(keep, 0.0, float('-inf'))

    with torch.no_grad():
        boolean = model(ids, mask=keep)
        additive = model(ids, mask=additive)

    assert torch.allclose(boolean.logits, additive.logits, atol=1e-6)


def test_generation_needs_eval_mode():
    '''Cached decoding only works outside training mode, where `cache.attn` is empty.'''
    model = make_model(vocab_size=VOCAB, model_dim=DIM, head_k_dim=16, block=BLOCK, block_n=BLOCK_N)
    ids = torch.randint(0, VOCAB, (1, 6))
    mask = torch.ones(1, 6)

    model.train()
    with pytest.raises(RuntimeError):
        list(model.generate(ids, max_new_tokens=3, eos_token_id=(), mask=mask))

    model.eval()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=3, eos_token_id=()).run()
    assert out.sequences.shape[1] == 9


def test_generation_threads_the_mask():
    '''Each decoding step widens the mask by the token it just appended.'''
    model = make_model(vocab_size=VOCAB, model_dim=DIM, head_k_dim=16, block=BLOCK, block_n=BLOCK_N).eval()
    ids = torch.randint(0, VOCAB, (2, 6))

    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=3, eos_token_id=(),
                             mask=torch.ones(2, 6))
        width = gen._key_mask.shape[1]
        assert width == 6
        for step in range(3):
            next(gen)
            width += 1
            assert gen._key_mask.shape == (2, width)

    assert isinstance(gen.cache, ModelCache)
