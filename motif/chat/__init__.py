from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from motif.chat.session import (
        ROLE_ALIASES,
        ChatMessage,
        ChatReply,
        ChatSession,
    )
    from motif.chat.template import (
        DIALOG_TOKEN_NAMES,
        EFFORT_TOKENS,
        ThoughtSplit,
        generation_prompt_ids,
        render_prompt,
        split_thought,
        stop_token_ids,
        strip_dialog_tokens,
    )

__all__ = [
    'ROLE_ALIASES',
    'ChatMessage',
    'ChatReply',
    'ChatSession',
    'DIALOG_TOKEN_NAMES',
    'EFFORT_TOKENS',
    'ThoughtSplit',
    'generation_prompt_ids',
    'render_prompt',
    'split_thought',
    'stop_token_ids',
    'strip_dialog_tokens',
]

_EXPORTS = {
    'ROLE_ALIASES': 'motif.chat.session',
    'ChatMessage': 'motif.chat.session',
    'ChatReply': 'motif.chat.session',
    'ChatSession': 'motif.chat.session',
    'DIALOG_TOKEN_NAMES': 'motif.chat.template',
    'EFFORT_TOKENS': 'motif.chat.template',
    'ThoughtSplit': 'motif.chat.template',
    'generation_prompt_ids': 'motif.chat.template',
    'render_prompt': 'motif.chat.template',
    'split_thought': 'motif.chat.template',
    'stop_token_ids': 'motif.chat.template',
    'strip_dialog_tokens': 'motif.chat.template',
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
