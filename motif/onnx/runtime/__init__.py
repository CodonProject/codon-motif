from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from motif.onnx.runtime.engine import OrtEngine, graph_path, prefill_graph_paths
    from motif.onnx.runtime.no_torch import block_torch, torch_blocked
    from motif.onnx.runtime.paths import default_bundle, default_tokenizer
    from motif.onnx.runtime.sampler import NumpySampler, softmax
    from motif.onnx.runtime.template import (
        EFFORT_DEFAULT,
        ThoughtSplit,
        render_prompt,
        resolve_effort,
        split_thought,
        stop_token_ids,
    )
    from motif.onnx.runtime.tokenizer import PackedTokenizer

__all__ = [
    'OrtEngine',
    'graph_path',
    'prefill_graph_paths',
    'NumpySampler',
    'softmax',
    'PackedTokenizer',
    'block_torch',
    'torch_blocked',
    'default_bundle',
    'default_tokenizer',
    'render_prompt',
    'stop_token_ids',
    'split_thought',
    'ThoughtSplit',
    'resolve_effort',
    'EFFORT_DEFAULT',
]

_EXPORTS = {
    'OrtEngine': 'motif.onnx.runtime.engine',
    'graph_path': 'motif.onnx.runtime.engine',
    'prefill_graph_paths': 'motif.onnx.runtime.engine',
    'NumpySampler': 'motif.onnx.runtime.sampler',
    'softmax': 'motif.onnx.runtime.sampler',
    'PackedTokenizer': 'motif.onnx.runtime.tokenizer',
    'block_torch': 'motif.onnx.runtime.no_torch',
    'torch_blocked': 'motif.onnx.runtime.no_torch',
    'default_bundle': 'motif.onnx.runtime.paths',
    'default_tokenizer': 'motif.onnx.runtime.paths',
    'render_prompt': 'motif.onnx.runtime.template',
    'stop_token_ids': 'motif.onnx.runtime.template',
    'split_thought': 'motif.onnx.runtime.template',
    'ThoughtSplit': 'motif.onnx.runtime.template',
    'resolve_effort': 'motif.onnx.runtime.template',
    'EFFORT_DEFAULT': 'motif.onnx.runtime.template',
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
