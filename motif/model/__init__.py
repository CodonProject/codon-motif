from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from motif.model.checkpoint import (
        LoadedCheckpoint,
        config_to_dict,
        load_checkpoint,
        save_safetensors,
    )
    from motif.model.config import MotifChordConfig
    from motif.model.generate import EosTokenId, Generation, TextGeneration
    from motif.model.motif import ModelOutput, MotifChordModel
    from motif.model.sampler import Constraint, Sampler, make_sampler

__all__ = [
    'MotifChordConfig',
    'MotifChordModel',
    'ModelOutput',
    'Generation',
    'TextGeneration',
    'EosTokenId',
    'Sampler',
    'Constraint',
    'make_sampler',
    'LoadedCheckpoint',
    'load_checkpoint',
    'save_safetensors',
    'config_to_dict',
]

_EXPORTS = {
    'MotifChordConfig': 'motif.model.config',
    'MotifChordModel': 'motif.model.motif',
    'ModelOutput': 'motif.model.motif',
    'Generation': 'motif.model.generate',
    'TextGeneration': 'motif.model.generate',
    'EosTokenId': 'motif.model.generate',
    'Sampler': 'motif.model.sampler',
    'Constraint': 'motif.model.sampler',
    'make_sampler': 'motif.model.sampler',
    'LoadedCheckpoint': 'motif.model.checkpoint',
    'load_checkpoint': 'motif.model.checkpoint',
    'save_safetensors': 'motif.model.checkpoint',
    'config_to_dict': 'motif.model.checkpoint',
}


def __getattr__(name: str) -> Any:
    '''
    Resolve one public name on first access.

    Args:
        name (str): Attribute being looked up.

    Returns:
        Any: The object re-exported under `name`.

    Raises:
        AttributeError: If `name` is not part of the public API.
    '''
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    return getattr(import_module(module), name)


def __dir__() -> list:
    '''
    Returns:
        list: The public names of this package.
    '''
    return sorted(__all__)
