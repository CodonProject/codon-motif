import torch

from motif.model.cache import (
    AttnResCache,
    BasicLayerCache,
    DecoderBlockCache,
    GatedDeltaAttentionLayerCache,
    KVLayerCache,
    ModelCache,
)
from motif.model.motif import MotifChordModel

from support import make_config, make_model


VOCAB = 64
BLOCK = ('gdn', 'chord')
BLOCK_N = 2


def test_training_forward_runs_and_gradients_flow():
    model = make_model(vocab_size=VOCAB, model_dim=64, head_k_dim=16, block=BLOCK, block_n=BLOCK_N).train()
    ids = torch.randint(0, VOCAB, (2, 8))

    out = model(ids, labels=ids)

    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.logits.shape == (2, 8, VOCAB)

    out.loss.backward()

    for bi, block in enumerate(model.blocks):
        for li, decoder in enumerate(block.features):
            grad = decoder.attn.q_proj.weight.grad
            assert grad is not None, f'block{bi}.layer{li} q_proj has no gradient'
            assert grad.abs().sum() > 0, f'block{bi}.layer{li} q_proj gradient is all zeros'

    for block_cache in out.cache:
        assert len(block_cache.attn) == 0
        assert block_cache.attn_res.is_empty is False


def test_create_cache_instantiates_every_cache_in_eval_mode():
    '''Outside train mode, every cache should be genuinely instantiated before forward.'''
    model = make_model(vocab_size=VOCAB, model_dim=64, head_k_dim=16, block=BLOCK, block_n=BLOCK_N).eval()

    cache = model.create_cache()
    assert isinstance(cache, ModelCache)
    assert len(cache) == BLOCK_N

    for block_cache in cache:
        assert isinstance(block_cache, DecoderBlockCache)
        assert isinstance(block_cache.attn_res, AttnResCache)
        assert len(block_cache.attn) == len(BLOCK)
        assert all(c is not None for c in block_cache.attn)
        for layer_cache, attn_type in zip(block_cache.attn, BLOCK):
            expected = (KVLayerCache if attn_type == 'chord'
                        else GatedDeltaAttentionLayerCache)
            assert isinstance(layer_cache, expected)
        assert block_cache.attn_res.is_empty

    model.train()
    train_cache = model.create_cache()
    for block_cache in train_cache:
        assert len(block_cache.attn) == 0

    forced = model.create_cache(use_attn=True)
    for block_cache in forced:
        assert len(block_cache.attn) == len(BLOCK)
        assert all(c is not None for c in block_cache.attn)

    model.eval()
    shells = ModelCache.create(len(model.blocks))
    for block_cache in shells:
        assert len(block_cache.attn) == 0
    with torch.no_grad():
        model(torch.randint(0, VOCAB, (2, 4)), cache=shells)
    for block_cache in shells:
        assert len(block_cache.attn) == len(BLOCK)
        assert all(c is not None for c in block_cache.attn)


def test_train_mode_and_eval_mode_forward_match():
    '''Enabling layer caches must not change the numerical result of a single forward.'''
    model = make_model(vocab_size=VOCAB, model_dim=64, head_k_dim=16, block=BLOCK, block_n=BLOCK_N)
    ids = torch.randint(0, VOCAB, (2, 8))

    model.train()
    logits_train = model(ids).logits

    model.eval()
    with torch.no_grad():
        out_eval = model(ids)

    assert torch.allclose(logits_train, out_eval.logits, atol=1e-6)

    for block_cache in out_eval.cache:
        assert len(block_cache.attn) == len(BLOCK)
        for cache, attn_type in zip(block_cache.attn, BLOCK):
            expected = (KVLayerCache if attn_type == 'chord'
                        else GatedDeltaAttentionLayerCache)
            assert isinstance(cache, expected)


def test_cache_structure():
    model = make_model(vocab_size=VOCAB, model_dim=64, head_k_dim=16, block=BLOCK, block_n=BLOCK_N).eval()
    ids = torch.randint(0, VOCAB, (2, 6))

    with torch.no_grad():
        out = model(ids)

    assert isinstance(out.cache, ModelCache)
    assert len(out.cache) == BLOCK_N

    for block_cache in out.cache:
        assert isinstance(block_cache, DecoderBlockCache)
        assert isinstance(block_cache.attn_res, AttnResCache)
        assert len(block_cache.attn) == len(BLOCK)
        assert all(isinstance(c, BasicLayerCache) for c in block_cache.attn)
        assert not block_cache.is_empty()
        assert block_cache.seq_len == ids.shape[1]
        assert len(block_cache.blocks) == 1 + len(BLOCK)
        assert block_cache.blocks[0].shape == (2, 6, 64)

    cached = out.cache
    cached.reset()
    for block_cache in cached:
        assert block_cache.is_empty()
        assert block_cache.attn_res.blocks == []

    with torch.no_grad():
        again = model(ids, cache=cached)
    assert torch.allclose(again.logits, out.logits, atol=1e-6)


def test_output_reset_caches():
    model = make_model(vocab_size=VOCAB, model_dim=64, head_k_dim=16, block=BLOCK, block_n=BLOCK_N).eval()
    ids = torch.randint(0, VOCAB, (2, 6))

    with torch.no_grad():
        out = model(ids)
        first_logits = out.logits
        out.reset_caches()
        assert out.cache[0].is_empty()
        out2 = model(ids, cache=out.cache)

    assert torch.allclose(first_logits, out2.logits, atol=1e-6)


def test_prefill_decode_matches_full_forward():
    '''prefill plus step-by-step decode should match a one-shot full forward numerically.'''
    model = make_model(vocab_size=VOCAB, model_dim=64, head_k_dim=16, block=BLOCK, block_n=BLOCK_N).eval()
    prompt_len = 8
    ids = torch.randint(0, VOCAB, (2, 12))

    with torch.no_grad():
        full = model(ids)
        out = model(ids[:, :prompt_len])

        assert out.cache[0].attn_res.seq_len == prompt_len

        for t in range(prompt_len, ids.shape[1]):
            out = model(ids[:, t:t + 1], cache=out.cache)
            assert out.logits.shape == (2, 1, VOCAB)
            assert torch.allclose(
                out.logits[:, 0, :], full.logits[:, t, :], atol=1e-4
            ), f'decode step t={t} does not match the full forward'
            assert out.cache[0].attn_res.seq_len == t + 1


def test_chunked_prefill_matches_full_forward():
    '''Chunked prefill (multiple chunks reusing the same cache) must match a full forward.'''
    model = make_model(vocab_size=VOCAB, model_dim=64, head_k_dim=16, block=BLOCK, block_n=BLOCK_N).eval()
    chunk = 4
    ids = torch.randint(0, VOCAB, (2, 12))

    with torch.no_grad():
        full = model(ids)
        cache = model.create_cache()
        assert isinstance(cache, ModelCache)
        for start in range(0, ids.shape[1], chunk):
            out = model(ids[:, start:start + chunk], cache=cache)
            cache = out.cache
            assert torch.allclose(
                out.logits, full.logits[:, start:start + chunk, :], atol=1e-4
            ), f'chunk start={start} does not match the full forward'


def test_decode_after_two_independent_prefills():
    '''After reset_caches, a new sequence must not be affected by the old one.'''
    model = make_model(vocab_size=VOCAB, model_dim=64, head_k_dim=16, block=BLOCK, block_n=BLOCK_N).eval()
    a = torch.randint(0, VOCAB, (2, 10))
    b = torch.randint(0, VOCAB, (2, 7))

    with torch.no_grad():
        out_a = model(a)
        out_a.reset_caches()
        out_b = model(b, cache=out_a.cache)
        ref_b = model(b)

        assert torch.allclose(out_b.logits, ref_b.logits, atol=1e-6)
        assert out_b.cache[0].attn_res.seq_len == b.shape[1]


def test_attn_res_continuation_keeps_current_input_gradient():
    '''Continuation only detaches history; the input of the current chunk must keep its computation graph.'''
    cache = AttnResCache()

    x1 = torch.randn(2, 4, 8, requires_grad=True)
    cache.init_embedding(x1)
    cache.update_increment(torch.randn(2, 4, 8))
    assert cache.seq_len == 4 and len(cache.blocks) == 2
    assert cache.partial_block is not None

    x2 = torch.randn(2, 1, 8, requires_grad=True)
    cache.init_embedding(x2)

    assert cache.seq_len == 5
    assert cache.chunk_len == 1
    assert len(cache.blocks) == 1
    assert cache.blocks[0].shape == (2, 5, 8)
    assert torch.allclose(cache.blocks[0][:, -1:, :], x2)

    values = cache.values()
    assert values.shape == (1, 2, 1, 8)

    values.sum().backward()
    assert x2.grad is not None and x2.grad.abs().sum() > 0, 'the current chunk input gradient was detached'
    assert x1.grad is None, 'history must not backpropagate gradients anymore'


def test_cache_to_migrates_all_caches():
    '''cache.to(...) should migrate attn_res and every layer's attn cache at the same time.'''
    torch.manual_seed(0)
    chord_only = ('chord', 'chord')
    model = MotifChordModel(make_config(vocab_size=VOCAB, model_dim=64, head_k_dim=16, block=chord_only, block_n=2)).eval()
    ids = torch.randint(0, VOCAB, (2, 8))

    with torch.no_grad():
        full = model(ids)
        out = model(ids[:, :6])

    cache = out.cache.to(dtype=torch.float64)
    for block_cache in cache:
        assert block_cache.attn_res.blocks[0].dtype == torch.float64
        assert len(block_cache.attn) == len(chord_only)
        for layer_cache in block_cache.attn:
            assert isinstance(layer_cache, KVLayerCache)
            assert layer_cache.k.dtype == torch.float64
            assert layer_cache.v.dtype == torch.float64

    model64 = MotifChordModel(make_config(vocab_size=VOCAB, model_dim=64, head_k_dim=16, block=chord_only, block_n=2))
    model64.load_state_dict(model.state_dict())
    model64.double().eval()

    with torch.no_grad():
        step = model64(ids[:, 6:7], cache=cache)
    assert step.logits.dtype == torch.float64
    assert torch.allclose(step.logits, full.logits[:, 6:7].double(), atol=1e-4)


def _run_all():
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith('test_') and callable(fn)
    ]
    for name, fn in tests:
        fn()
        print(f'  ok  {name}')
    print(f'\n{len(tests)} passed')


if __name__ == '__main__':
    _run_all()
