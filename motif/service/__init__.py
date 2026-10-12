from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from motif.service.app import create_app, torch_imported
    from motif.service.backends import (
        DEFAULT_CONTEXT_LENGTH,
        DEFAULT_MAX_NEW_TOKENS,
        DEFAULT_REPETITION_PENALTY,
        DEFAULT_TEMPERATURE,
        OnnxBackend,
        SamplingParams,
        TorchBackend,
        build_backend,
    )
    from motif.service.schemas import ChatCompletionRequest, ChatMessage, CompletionRequest
    from motif.service.server import main

__all__ = [
    'create_app',
    'torch_imported',
    'build_backend',
    'OnnxBackend',
    'TorchBackend',
    'SamplingParams',
    'main',
    'ChatMessage',
    'ChatCompletionRequest',
    'CompletionRequest',
    'DEFAULT_MAX_NEW_TOKENS',
    'DEFAULT_TEMPERATURE',
    'DEFAULT_REPETITION_PENALTY',
    'DEFAULT_CONTEXT_LENGTH',
]

_EXPORTS = {
    'create_app': 'motif.service.app',
    'torch_imported': 'motif.service.app',
    'build_backend': 'motif.service.backends',
    'OnnxBackend': 'motif.service.backends',
    'TorchBackend': 'motif.service.backends',
    'SamplingParams': 'motif.service.backends',
    'DEFAULT_MAX_NEW_TOKENS': 'motif.service.backends',
    'DEFAULT_TEMPERATURE': 'motif.service.backends',
    'DEFAULT_REPETITION_PENALTY': 'motif.service.backends',
    'DEFAULT_CONTEXT_LENGTH': 'motif.service.backends',
    'main': 'motif.service.server',
    'ChatMessage': 'motif.service.schemas',
    'ChatCompletionRequest': 'motif.service.schemas',
    'CompletionRequest': 'motif.service.schemas',
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
