import contextlib
import io

from motif.utils.units import count_params, format_count

from support import make_model


def test_format_count_units():
    assert format_count(0) == '0'
    assert format_count(512) == '512'
    assert format_count(999) == '999'
    assert format_count(1000) == '1.000 K'
    assert format_count(1234) == '1.234 K'
    assert format_count(12_345) == '12.345 K'
    assert format_count(999_999) == '999.999 K'
    assert format_count(1_000_000) == '1.000 M'
    assert format_count(1_387_000) == '1.387 M'
    assert format_count(242_700_000) == '242.700 M'
    assert format_count(2_250_000_000) == '2.250 B'
    assert format_count(2_250_000_000_000) == '2.250 T'
    assert format_count(9_000_000_000_000_000).endswith('T')


def test_format_count_options():
    assert format_count(1_387_000, decimals=1) == '1.4 M'
    assert format_count(1_387_000, space=False) == '1.387M'


def test_count_params_helper():
    model = make_model()
    assert count_params(model) == sum(p.numel() for p in model.parameters())
    assert isinstance(count_params(model), int)
    assert count_params(model) > count_params(model, trainable_only=True)
    assert count_params(model, trainable_only=True) == sum(
        p.numel() for p in model.parameters() if p.requires_grad)


def test_count_param_returns_int_without_printing():
    model = make_model()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        total = model.count_param()
    assert total == sum(p.numel() for p in model.parameters())
    assert isinstance(total, int)
    assert buffer.getvalue() == '', 'print=False (the default) must not print anything'


def test_count_param_prints_human_readable():
    model = make_model()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        total = model.count_param(print=True)
    out = buffer.getvalue()
    assert 'MotifChordModel params:' in out, f'printed output is wrong: {out!r}'
    assert f'{total:,}' in out, 'the exact value with thousands separators should be printed too'
    assert format_count(total) in out, 'the human-readable unit string should be printed too'


def test_count_param_uses_units_for_big_models():
    model = make_model(vocab_size=40960, model_dim=256)
    total = model.count_param()
    assert total > 1_000_000, f'this configuration should reach the M range, got {total}'
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        model.count_param(print=True)
    assert ' M (' in buffer.getvalue(), f'the M unit should be present: {buffer.getvalue()!r}'


def test_existing_configs_count():
    '''
    Parameter counts of the dashboard recipe (block_n=6 / dim 768) and the 8GB
    preset, as a regression baseline.
    '''
    board = make_model(vocab_size=40960, model_dim=768, num_heads=12, num_kv_heads=3,
                       head_k_dim=64, block=('gdn', 'gdn', 'gdn', 'chord'), block_n=6)
    assert 240e6 < board.count_param() < 245e6, \
        f'the block_n=6 default configuration should be about 242.7M'

    eight = make_model(vocab_size=40960, model_dim=320, num_heads=4, num_kv_heads=2,
                       head_k_dim=64, block=('gdn', 'chord'), block_n=8)
    assert 34e6 < eight.count_param() < 37e6, f'the 8GB preset should be about 35.5M'


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            print(f'--- {name}')
            fn()
    print('all count_param tests passed')
