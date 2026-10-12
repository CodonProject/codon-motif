from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from motif.onnx.export_onnx import (
        DEFAULT_PREFILL_LENGTHS,
        ExportResult,
        export_onnx,
    )
    from motif.onnx.reader import load_model, resolve_source

__all__ = [
    'export_onnx',
    'ExportResult',
    'DEFAULT_PREFILL_LENGTHS',
    'load_model',
    'resolve_source',
]

_EXPORTS = {
    'export_onnx': 'motif.onnx.export_onnx',
    'ExportResult': 'motif.onnx.export_onnx',
    'DEFAULT_PREFILL_LENGTHS': 'motif.onnx.export_onnx',
    'load_model': 'motif.onnx.reader',
    'resolve_source': 'motif.onnx.reader',
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
