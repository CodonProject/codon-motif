import dataclasses
import json
import os
import shutil
import tempfile

import pytest
import torch

from motif.model.checkpoint import (
    HAS_SAFETENSORS,
    config_from_state_dict,
    load_checkpoint,
)
from motif.model.motif import MotifChordModel

from support import make_config, make_model


def workdir():
    path = tempfile.mkdtemp(prefix='motif_from_pretrained_')
    return path


def assert_same_weights(model, reference):
    '''Compare tensor by tensor and report the first key that differs.'''
    left, right = model.state_dict(), reference.state_dict()
    assert sorted(left) == sorted(right), 'state_dict keys differ'
    for key in sorted(left):
        assert torch.equal(left[key], right[key]), f'{key} weights differ'


def assert_loads(path, reference, config=None, **kwargs) -> MotifChordModel:
    '''Load a checkpoint and check the weights against `reference`.'''
    model = MotifChordModel.from_pretrained(path, config=config, verbose=False, **kwargs)
    assert_same_weights(model, reference)
    return model


def test_infer_config_from_state_dict():
    config = make_config()
    inferred = config_from_state_dict(MotifChordModel(config).state_dict())
    assert inferred == config, f'inferred config is wrong: {inferred}'


def test_infer_config_keeps_block_recipe_order():
    config = make_config(block=('chord', 'gdn', 'gdn'), block_n=3, model_dim=64,
                         num_heads=8, num_kv_heads=2, head_k_dim=16)
    inferred = config_from_state_dict(MotifChordModel(config).state_dict())
    assert inferred.block == ('chord', 'gdn', 'gdn'), f'the block order was lost: {inferred.block}'
    assert inferred.block_n == 3
    assert inferred.num_kv_heads == 2 and inferred.head_k_dim == 16
    assert inferred == config


def test_infer_config_detects_untied_weights():
    config = make_config(tied_weight=False)
    model = MotifChordModel(config)
    with torch.no_grad():
        model.lm_head.weight.add_(1.0)
    inferred = config_from_state_dict(model.state_dict())
    assert inferred.tied_weight is False, 'an untied checkpoint must not be judged tied'
    assert inferred == config


def test_infer_config_rejects_foreign_state_dict():
    try:
        config_from_state_dict({'some.other.weight': torch.zeros(3, 3)})
    except ValueError as exc:
        assert 'embed.weight' in str(exc), f'error message is not specific enough: {exc}'
    else:
        raise AssertionError('a state_dict that is not from motif should raise ValueError')


def test_from_pretrained_bare_state_dict():
    root = workdir()
    try:
        reference = make_model()
        path = os.path.join(root, 'step_000012345.pt')
        torch.save(reference.state_dict(), path)

        model = assert_loads(path, reference)
        assert model.config == reference.config, \
            'without a config the shapes must be used to infer it'
        assert not model.training, 'the model should be in eval() mode after from_pretrained'
        assert load_checkpoint(path).source == 'bare state_dict'
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_from_pretrained_without_config_json_uses_shapes():
    '''With no config.json beside a bare state_dict, only the shapes can be used.'''
    root = workdir()
    try:
        reference = make_model(block=('gdn', 'gdn', 'chord'), block_n=3)
        path = os.path.join(root, 'last.pt')
        torch.save(reference.state_dict(), path)

        model = MotifChordModel.from_pretrained(path, verbose=False)
        assert model.config == reference.config
        assert_same_weights(model, reference)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_from_pretrained_pipeline_checkpoint():
    root = workdir()
    try:
        reference = make_model()
        path = os.path.join(root, 'last.pt')
        payload = {
            '__pipeline__': 'PretrainPipeline',
            '__version__': '1.1',
            'state': {'global_step': 1234, 'phase': 'training'},
            'metrics': {'loss/train': 3.37},
            'callbacks': [{}, {}],
            'payload': {
                'model': reference.state_dict(),
                'optimizer': {'state': {}},
                'scheduler': {'last_epoch': 12},
            },
            'data': {'plan': None},
        }
        torch.save(payload, path)

        model = assert_loads(path, reference)
        assert model.config == reference.config
        loaded = load_checkpoint(path)
        assert 'PretrainPipeline' in loaded.source, f'format tag is wrong: {loaded.source}'
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_from_pretrained_pipeline_prefers_embedded_config():
    '''
    When the payload or the top level carries a config, it wins (no more
    guessing from shapes).
    '''
    root = workdir()
    try:
        reference = make_model()
        path = os.path.join(root, 'last.pt')
        torch.save({
            '__pipeline__': 'PretrainPipeline',
            'payload': {'model': reference.state_dict(),
                        'config': dataclasses.asdict(reference.config)},
        }, path)

        model = MotifChordModel.from_pretrained(path, verbose=False)
        assert model.config == reference.config, \
            f'the config in the payload was not read: {model.config}'
        assert_same_weights(model, reference)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_from_pretrained_safetensors():
    if not HAS_SAFETENSORS:
        pytest.skip('safetensors is not installed')
    from safetensors.torch import save_file

    root = workdir()
    try:
        reference = make_model(tied_weight=False)
        weights = reference.state_dict()
        path = os.path.join(root, 'model.safetensors')
        save_file(weights, path, metadata={'format': 'pt'})
        with open(os.path.join(root, 'config.json'), 'w', encoding='utf-8') as handle:
            json.dump(dataclasses.asdict(reference.config), handle)

        model = assert_loads(path, reference)
        assert model.config == reference.config, 'the sidecar config.json did not take effect'
        assert load_checkpoint(path).source == 'safetensors'
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_from_pretrained_safetensors_directory():
    '''A directory (or a *.index.json) instead of a single .safetensors must also load.'''
    if not HAS_SAFETENSORS:
        pytest.skip('safetensors is not installed')
    from safetensors.torch import save_file

    root = workdir()
    try:
        reference = make_model(tied_weight=False)
        save_file(reference.state_dict(), os.path.join(root, 'model.safetensors'))
        model = assert_loads(root, reference)
        assert model.config == reference.config
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_from_pretrained_safetensors_from_metadata_config():
    if not HAS_SAFETENSORS:
        pytest.skip('safetensors is not installed')
    from safetensors.torch import save_file

    root = workdir()
    try:
        reference = make_model(tied_weight=False, block_n=3,
                               block=('gdn', 'chord', 'gdn'))
        path = os.path.join(root, 'model.safetensors')
        save_file(reference.state_dict(), path, metadata={
            'config': json.dumps(dataclasses.asdict(reference.config)),
        })
        model = MotifChordModel.from_pretrained(path, verbose=False)
        assert model.config == reference.config, \
            'the config in the safetensors metadata did not take effect'
        assert_same_weights(model, reference)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_from_pretrained_strips_wrapper_prefixes():
    root = workdir()
    try:
        reference = make_model()
        weights = reference.state_dict()
        weights.pop('lm_head.weight')
        path = os.path.join(root, 'ddp_step_000000123.pt')
        torch.save({'module.' + key: value for key, value in weights.items()}, path)

        model = assert_loads(path, reference)
        assert model.config.tied_weight is True
        assert not any(key.startswith('module.') for key in model.state_dict())
        assert model.embed.weight is model.lm_head.weight, 'the tied relation should be restored'
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_from_pretrained_only_strips_shared_prefixes():
    '''When only some keys carry the `model.` prefix, it must not be stripped blindly.'''
    root = workdir()
    try:
        reference = make_model(tied_weight=False)
        weights = reference.state_dict()
        weights = {('model.' + k if k.startswith('embed') else k): v
                   for k, v in weights.items()}
        path = os.path.join(root, 'partial_prefix.pt')
        torch.save(weights, path)

        loaded = load_checkpoint(path)
        assert 'model.embed.weight' in loaded.state_dict, \
            'a prefix that appears only once must not be stripped'
        assert 'embed.weight' not in loaded.state_dict
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_from_pretrained_tied_checkpoint_strict():
    '''A tied checkpoint that stores only embed.weight must still pass strict=True.'''
    root = workdir()
    try:
        reference = make_model()
        weights = reference.state_dict()
        weights.pop('lm_head.weight')
        path = os.path.join(root, 'tied.pt')
        torch.save(weights, path)

        model = MotifChordModel.from_pretrained(path, strict=True, verbose=False)
        assert_same_weights(model, reference)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_from_pretrained_strict_raises_on_mismatch():
    '''When a genuinely missing weight (not the tied lm_head) is absent, strict=True must raise.'''
    root = workdir()
    try:
        reference = make_model()
        weights = reference.state_dict()
        weights.pop('norm.weight')
        path = os.path.join(root, 'missing_norm.pt')
        torch.save(weights, path)

        try:
            MotifChordModel.from_pretrained(path, strict=True, verbose=False)
        except RuntimeError as exc:
            assert 'missing' in str(exc), f'error message is not specific enough: {exc}'
        else:
            raise AssertionError('strict=True should raise on a missing key')
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_from_pretrained_explicit_config_argument():
    root = workdir()
    try:
        reference = make_model()
        path = os.path.join(root, 'last.pt')
        torch.save(reference.state_dict(), path)

        model = assert_loads(path, reference, config=reference.config)
        assert model.config is reference.config
        assert_loads(path, reference, config=dataclasses.asdict(reference.config))
        json_path = os.path.join(root, 'config.json')
        with open(json_path, 'w', encoding='utf-8') as handle:
            json.dump(dataclasses.asdict(reference.config), handle)
        assert_loads(path, reference, config=json_path)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_from_pretrained_overrides_beat_inference():
    '''A chord-only checkpoint cannot infer head_k_dim, so an override supplies it.'''
    root = workdir()
    try:
        reference = make_model(block=('chord', 'chord'), head_k_dim=32)
        path = os.path.join(root, 'chord_only.pt')
        torch.save(reference.state_dict(), path)

        guessed = MotifChordModel.from_pretrained(path, verbose=False)
        assert guessed.config.head_k_dim == 64, \
            'with no gdn layer the default head_k_dim should be kept'

        fixed = MotifChordModel.from_pretrained(path, verbose=False, head_k_dim=32)
        assert fixed.config.head_k_dim == 32, 'overrides should win over the inferred result'
        assert fixed.config == reference.config
        assert_same_weights(fixed, reference)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_from_pretrained_rejects_unknown_override():
    root = workdir()
    try:
        reference = make_model()
        path = os.path.join(root, 'last.pt')
        torch.save(reference.state_dict(), path)
        try:
            MotifChordModel.from_pretrained(path, verbose=False, hidden_size=768)
        except TypeError as exc:
            assert 'hidden_size' in str(exc)
        else:
            raise AssertionError('a misspelled override should raise TypeError')
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_from_pretrained_missing_file():
    try:
        MotifChordModel.from_pretrained('definitely/not/here.pt', verbose=False)
    except FileNotFoundError as exc:
        assert 'here.pt' in str(exc)
    else:
        raise AssertionError('a missing file should raise FileNotFoundError')


def test_from_pretrained_verbose_prints_summary():
    import contextlib
    import io

    root = workdir()
    try:
        reference = make_model()
        path = os.path.join(root, 'last.pt')
        torch.save(reference.state_dict(), path)

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            MotifChordModel.from_pretrained(path)
        out = buffer.getvalue()
        assert '[MotifChordModel]' in out, f'missing the summary line: {out!r}'
        assert 'bare state_dict' in out, f'the summary should name the detected format: {out!r}'

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            MotifChordModel.from_pretrained(path, verbose=False)
        assert buffer.getvalue() == '', 'verbose=False must not print anything'
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            print(f'--- {name}')
            fn()
    print('all from_pretrained tests passed')
