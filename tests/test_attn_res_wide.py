import warnings

import torch

import motif.ops.attn_res as attn_res_op
from motif.block.decoder import DecoderBlock
from motif.model.cache import AttnResCache, DecoderBlockCache, ModelCache
from motif.model.motif import MotifChordModel
from motif.ops.attn_res import MAX_SRC, apply_block_attn_res
from motif.ops.rms_norm import RMSNorm

from support import make_config


VOCAB = 64
BLOCK = ('gdn', 'gdn', 'gdn', 'chord')
BLOCK_N = 2
DIM = 64
SRC_DIM = 8


def cache_with(x: torch.Tensor, increments) -> AttnResCache:
    '''A block cache whose history is `[x, *increments]`.'''
    cache = AttnResCache()
    cache.init_embedding(x)
    for f_h in increments:
        cache.update_increment(f_h)
    return cache


def test_default_block_width_fits_the_fused_kernel():
    '''The default block width (4 decoders) makes the widest read exactly MAX_SRC.'''
    assert len(make_config(vocab_size=VOCAB, model_dim=DIM, head_k_dim=16, block=BLOCK, block_n=BLOCK_N).block) == MAX_SRC


def test_history_holds_one_increment_per_decoder():
    '''After one block the history is the input plus one increment per decoder.'''
    torch.manual_seed(0)
    model = MotifChordModel(make_config(vocab_size=VOCAB, model_dim=DIM, head_k_dim=16, block=BLOCK, block_n=BLOCK_N)).train()
    ids = torch.randint(0, VOCAB, (2, 7))

    out = model(ids)

    assert isinstance(out.cache, ModelCache)
    for block_cache in out.cache:
        blocks = block_cache.attn_res.blocks
        assert len(blocks) == 1 + len(BLOCK)
        assert blocks[0].shape == (2, 7, DIM)
        assert all(e.shape == (2, 7, DIM) for e in blocks[1:])
        assert block_cache.attn_res.num_sources() == 1 + len(BLOCK)
        assert len({id(e) for e in blocks}) == len(blocks)


def test_block_representation_is_the_sum_of_the_increments():
    '''Widening the read does not change the residual stream: the block output is
    still the sum of every increment.'''
    torch.manual_seed(0)
    block = DecoderBlock(make_config(vocab_size=VOCAB, model_dim=DIM, head_k_dim=16, block=BLOCK, block_n=BLOCK_N))
    x = torch.randn(2, 7, DIM, requires_grad=True)
    cache = DecoderBlockCache()

    out = block(x, cache)

    attn_res = cache.attn_res
    increments = attn_res.blocks[1:]
    assert len(increments) == len(BLOCK)
    assert torch.allclose(attn_res.partial_block, torch.stack(increments).sum(dim=0))
    assert torch.allclose(out.hidden_states, attn_res.partial_block)


def test_wide_read_is_not_a_function_of_the_running_sum_alone():
    '''Two runs whose increments sum to the same total still read differently.'''
    torch.manual_seed(0)
    x = torch.randn(1, 4, SRC_DIM)
    a = torch.randn(1, 4, SRC_DIM)
    b = torch.randn(1, 4, SRC_DIM)
    d = torch.randn(1, 4, SRC_DIM)

    paired = cache_with(x, [a, b])
    shuffled = cache_with(x, [a + d, b - d])
    query = torch.randn(SRC_DIM) * 0.05
    norm = RMSNorm(SRC_DIM)

    wide_paired = apply_block_attn_res(paired, query, norm)
    wide_shuffled = apply_block_attn_res(shuffled, query, norm)
    assert not torch.allclose(wide_paired, wide_shuffled)


def test_alignment_trims_history_to_the_current_chunk():
    '''A continued run trims the history to the current chunk, whose only source
    is the step just fed in.'''
    torch.manual_seed(0)
    cache = AttnResCache()
    cache.init_embedding(torch.randn(1, 6, SRC_DIM))

    step = torch.randn(1, 1, SRC_DIM)
    cache.init_embedding(step)

    aligned = cache.aligned_values()
    assert len(aligned) == 1
    assert aligned[0].shape == (1, 1, SRC_DIM)
    assert torch.allclose(aligned[0], step)


def test_above_max_src_falls_back_to_eager_with_one_warning():
    '''Reads wider than MAX_SRC fall back to the eager path and warn once.'''
    torch.manual_seed(0)
    x = torch.randn(1, 4, SRC_DIM)
    cache = cache_with(x, [torch.randn(1, 4, SRC_DIM) for _ in range(MAX_SRC + 1)])
    query = torch.randn(SRC_DIM) * 0.05
    norm = RMSNorm(SRC_DIM)
    assert len(cache.aligned_values()) > MAX_SRC

    was_warned = attn_res_op._warned_too_many
    attn_res_op._warned_too_many = False
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out = apply_block_attn_res(cache, query, norm)
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)

        with warnings.catch_warnings(record=True) as again:
            warnings.simplefilter('always')
            apply_block_attn_res(cache, query, norm)
        assert not again
    finally:
        attn_res_op._warned_too_many = was_warned

    assert out.shape == x.shape and torch.isfinite(out).all()


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
