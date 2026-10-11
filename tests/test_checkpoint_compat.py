import os

import pytest
import torch

from motif.model.motif import MotifChordModel


#: Point this at a published checkpoint (or a directory holding last.pt) to check real weights.
CHECKPOINT = os.environ.get('MOTIF_REAL_CHECKPOINT', '')


def _real_checkpoint():
    if not CHECKPOINT:
        pytest.skip('set MOTIF_REAL_CHECKPOINT to a checkpoint path to run this test')
    path = CHECKPOINT
    if os.path.isdir(path):
        path = os.path.join(path, 'last.pt')
    if not os.path.isfile(path):
        pytest.skip(f'no checkpoint at {path}')
    return path


def test_published_weights_still_load():
    path = _real_checkpoint()
    blob = torch.load(path, map_location='cpu', weights_only=False)
    saved = blob.get('payload', {}).get('model', blob)
    saved = {name: value for name, value in saved.items() if hasattr(value, 'shape')}

    model = MotifChordModel.from_pretrained(path, verbose=False)
    config = model.config

    assert config.vocab_size == 40960, 'the published tokenizer reserves 40,960 rows'
    assert (config.model_dim, config.num_heads, config.num_kv_heads, config.head_k_dim) == (768, 12, 3, 64)
    assert config.block == ('gdn', 'gdn', 'gdn', 'chord') and config.block_n == 4
    assert config.tied_weight is True
    assert not hasattr(config, 'vision'), 'vision was dropped from the library'

    assert len(saved) == 299, f'expected the 299 tensors of the published trunk, got {len(saved)}'
    assert not [name for name in saved if 'vision' in name], 'published weights carry no vision tensors'
    assert model.count_param() == 172_256_480

    report = model.load_state_dict(saved, strict=False)
    assert not report.missing_keys, report.missing_keys
    assert not report.unexpected_keys, report.unexpected_keys

    pointers = {}
    for name, value in saved.items():
        pointers.setdefault(value.data_ptr(), []).append(name)
    tied = [sorted(names) for names in pointers.values() if len(names) > 1]
    assert tied == [['embed.weight', 'lm_head.weight']], f'unexpected weight sharing: {tied}'
    unique = sum(value.numel() for value in saved.values())
    unique -= sum(saved[name].numel() for name in tied[0][1:])
    assert unique == model.count_param(), 'stored tensors minus the tied copy must match the model'

    logits = model(torch.tensor([[1, 2, 3]])).logits
    assert tuple(logits.shape) == (1, 3, 40960)
