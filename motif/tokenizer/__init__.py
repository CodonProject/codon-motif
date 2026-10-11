from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from motif.tokenizer.spec_token import SPEC_TOKEN, token
    from motif.tokenizer.tokenizer import (
        PackedTokenizer,
        TokenizerTrainerResult,
        create_tokenizer_trainer,
    )

__all__ = [
    'SPEC_TOKEN',
    'token',
    'PackedTokenizer',
    'TokenizerTrainerResult',
    'create_tokenizer_trainer',
]

_EXPORTS = {
    'SPEC_TOKEN': 'motif.tokenizer.spec_token',
    'token': 'motif.tokenizer.spec_token',
    'PackedTokenizer': 'motif.tokenizer.tokenizer',
    'TokenizerTrainerResult': 'motif.tokenizer.tokenizer',
    'create_tokenizer_trainer': 'motif.tokenizer.tokenizer',
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
