'''Shared helpers for the test suite, so that 15 test modules do not each redefine the
same tokenizer / model / dataset builders.

Everything here is a plain module-level function rather than a pytest fixture, because
test modules in this repository import what they need directly and subdirectories do not
see `conftest.py` fixtures without importing them.
'''

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from motif.data.base import MotifDataset
from motif.model.config import MotifChordConfig
from motif.model.motif import MotifChordModel
from motif.tokenizer.tokenizer import create_tokenizer_trainer
from motif.utils.session import TEMPLATE_PATH

CORPUS = [f'the quick brown fox jumps over the lazy dog number {i} hello 世界'
          for i in range(64)]

_TOKENIZER = None


def make_tokenizer():
    '''
    Train a small vocabulary holding every special token with this library's own
    trainer, and install the chat template on it.

    Returns:
        PackedTokenizer: The tokenizer, trained once and cached for the whole session.
    '''
    global _TOKENIZER
    if _TOKENIZER is None:
        trainer = create_tokenizer_trainer(vocab_size=512)
        trainer.train_from_iterator(iter(CORPUS))
        _TOKENIZER = trainer.packed_tokenizer
        _TOKENIZER.set_chat_template(open(TEMPLATE_PATH, encoding='utf-8').read())
    return _TOKENIZER


def make_config(**overrides) -> MotifChordConfig:
    '''
    A small `MotifChordConfig`, with any field replaceable through keyword arguments.

    Returns:
        MotifChordConfig: The configuration.
    '''
    fields = dict(vocab_size=512, model_dim=32, num_heads=4, num_kv_heads=2,
                  head_k_dim=8, block=('gdn', 'chord'), block_n=2)
    fields.update(overrides)
    return MotifChordConfig(**fields)


def make_model(seed: int = 0, **overrides) -> MotifChordModel:
    '''
    A freshly initialised tiny model, seeded so that two instances are comparable.

    Args:
        seed (int): Torch seed used before the parameters are created.
        **overrides: Configuration field values replacing the small defaults.

    Returns:
        MotifChordModel: The model.
    '''
    torch.manual_seed(seed)
    return MotifChordModel(make_config(**overrides))


class ListDataset(MotifDataset):
    '''A map-style dataset for tests: each record is a sequence of token ids.'''

    def __init__(self, records):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def write_jsonl(folder: str, rows, name: str = 'part.jsonl') -> str:
    '''
    Write `rows` into one JSONL file inside `folder` and return its path (creating the
    folder when needed).

    Args:
        folder (str): Target directory.
        rows: Iterable of JSON-serialisable rows.
        name (str): File name inside `folder`.

    Returns:
        str: Path of the written file.
    '''
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    with open(path, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    return path
