from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from motif.block.attention import (
        AttentionOutput,
        BasicAttention,
        ChordAttention,
        GatedDeltaAttention,
    )
    from motif.block.decoder import Decoder, DecoderBlock, DecoderBlockOutput
    from motif.block.mlp import SwiGLU

__all__ = [
    'Decoder',
    'DecoderBlock',
    'DecoderBlockOutput',
    'SwiGLU',
    'BasicAttention',
    'AttentionOutput',
    'ChordAttention',
    'GatedDeltaAttention',
]

_EXPORTS = {
    'Decoder': 'motif.block.decoder',
    'DecoderBlock': 'motif.block.decoder',
    'DecoderBlockOutput': 'motif.block.decoder',
    'SwiGLU': 'motif.block.mlp',
    'BasicAttention': 'motif.block.attention',
    'AttentionOutput': 'motif.block.attention',
    'ChordAttention': 'motif.block.attention',
    'GatedDeltaAttention': 'motif.block.attention',
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
