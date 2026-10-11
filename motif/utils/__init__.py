from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from motif.utils.session import (
        EFFORT_ALIAS,
        EFFORT_DEFAULT,
        TEMPLATE_PATH,
        TOKEN_NAMES,
        MaskPolicy,
        resolve_effort,
        resolve_token_id,
    )
    from motif.utils.units import UNITS, count_params, format_count

__all__ = [
    'count_params',
    'format_count',
    'UNITS',
    'resolve_effort',
    'resolve_token_id',
    'TOKEN_NAMES',
    'TEMPLATE_PATH',
    'EFFORT_ALIAS',
    'EFFORT_DEFAULT',
    'MaskPolicy',
]

_EXPORTS = {
    'count_params': 'motif.utils.units',
    'format_count': 'motif.utils.units',
    'UNITS': 'motif.utils.units',
    'resolve_effort': 'motif.utils.session',
    'resolve_token_id': 'motif.utils.session',
    'TOKEN_NAMES': 'motif.utils.session',
    'TEMPLATE_PATH': 'motif.utils.session',
    'EFFORT_ALIAS': 'motif.utils.session',
    'EFFORT_DEFAULT': 'motif.utils.session',
    'MaskPolicy': 'motif.utils.session',
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
