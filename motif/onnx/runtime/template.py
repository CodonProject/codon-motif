from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

__all__ = [
    'EFFORT_ALIAS',
    'EFFORT_DEFAULT',
    'EFFORT_TOKENS',
    'TOKEN_NAMES',
    'ThoughtSplit',
    'render_prompt',
    'resolve_effort',
    'resolve_token_id',
    'split_thought',
    'stop_token_ids',
]

#: Special-token text by logical name, the same table the library's sessions use.
TOKEN_NAMES = {
    'bos': '<|bos|>', 'eos': '<|eos|>', 'end_of_text': '<|end_of_text|>',
    'im_start': '<|im_start|>', 'im_end': '<|im_end|>',
    'system': '<|system|>', 'user': '<|human|>', 'model': '<|model|>',
    'tool': '<|tool_response|>',
    'thought_start': '<|thought_start|>', 'thought_end': '<|thought_end|>',
    'effort_low': '<|effort_low|>', 'effort_high': '<|effort_high|>', 'effort_max': '<|effort_max|>',
    'tool_call_start': '<|tool_call_start|>', 'tool_call_end': '<|tool_call_end|>',
    'tool_name_divider': '<|tool_name_divider|>',
    'fim_pre': '<|fim_prefix|>', 'fim_mid': '<|fim_middle|>', 'fim_suf': '<|fim_suffix|>',
    'pad': '<|pad|>',
}

#: Thinking-effort alias table, kept in sync with `render_effort` in the template.
EFFORT_ALIAS = {
    'minimal': 'low', 'low': 'low',
    'medium': 'high', 'high': 'high', 'xhigh': 'high',
    'max': 'max', 'ultra': 'max',
}
EFFORT_DEFAULT = 'high'

#: Effort marker text -> canonical level.
EFFORT_TOKENS = {
    TOKEN_NAMES[f'effort_{level}']: level for level in ('low', 'high', 'max')
}

_THOUGHT_START = TOKEN_NAMES['thought_start']
_THOUGHT_END = TOKEN_NAMES['thought_end']


def resolve_effort(effort: Optional[str] = None) -> str:
    '''
    Collapse a thinking-effort level name to its canonical level
    ('medium' -> 'high').

    Args:
        effort (Optional[str]): Requested level; None uses `EFFORT_DEFAULT`.

    Returns:
        str: The canonical level ('low', 'high' or 'max').

    Raises:
        ValueError: If the level is not one of the known names.
    '''
    level = str(effort or EFFORT_DEFAULT).strip().lower()
    if level not in EFFORT_ALIAS:
        raise ValueError(
            f'unexpected effort level: {effort!r} '
            f"(expected 'minimal', 'low', 'medium', 'high', 'xhigh', 'max' or 'ultra')"
        )
    return EFFORT_ALIAS[level]


def resolve_token_id(tokenizer, name: str) -> Optional[int]:
    '''
    Resolve a logical name or the literal vocabulary text into a token id.

    Args:
        tokenizer: Tokenizer to query.
        name (str): Logical name or literal token text.

    Returns:
        Optional[int]: The token id, or None when it is not in the vocabulary.
    '''
    return tokenizer.token_to_id(TOKEN_NAMES.get(name, name))


def render_prompt(
    tokenizer,
    messages: Sequence[Dict[str, Any]],
    add_generation_prompt: bool = True,
    **template_kwargs,
) -> List[int]:
    '''
    Render a conversation through the chat template carried by the archive.

    Args:
        tokenizer: Tokenizer exposing `apply_chat_template`.
        messages (Sequence[Dict[str, Any]]): OpenAI-style messages; every message
            has a `role` and a `content`, and may carry `thought` / `effort` /
            `tool_calls` / `name`.
        add_generation_prompt (bool): Append the assistant generation prompt.
        **template_kwargs: Extra template switches (for example `effort` and
            `enable_thinking`).

    Returns:
        List[int]: Token ids of the rendered conversation.

    Raises:
        ValueError: If `messages` is empty.
    '''
    if not messages:
        raise ValueError('the conversation is empty: add a message first')
    return list(
        tokenizer.apply_chat_template(
            list(messages), add_generation_prompt=add_generation_prompt, **template_kwargs
        )
    )


def stop_token_ids(tokenizer, extra: Iterable[Union[str, int]] = ()) -> Tuple[int, ...]:
    '''
    Collect the token ids that end one assistant turn.

    Every message of the template ends with `<|im_end|>`, and a turn that emits a
    tool call ends with `<|tool_call_end|>`; both are included when the vocabulary
    has them.

    Args:
        tokenizer: Tokenizer to resolve the tokens against.
        extra (Iterable[Union[str, int]]): Extra token ids or token names.

    Returns:
        Tuple[int, ...]: The stop ids, in a stable order and without duplicates.
    '''
    ids: List[int] = []
    for name in ('im_end', 'tool_call_end'):
        token_id = resolve_token_id(tokenizer, name)
        if token_id is not None and token_id not in ids:
            ids.append(token_id)

    for item in extra:
        token_id = resolve_token_id(tokenizer, item) if isinstance(item, str) else int(item)
        if token_id is not None and token_id not in ids:
            ids.append(token_id)

    if not ids:
        eos = getattr(tokenizer, 'token_eos', None)
        fallback = tokenizer.token_to_id(eos) if isinstance(eos, str) else None
        if fallback is not None:
            ids.append(int(fallback))
    return tuple(ids)


@dataclass
class ThoughtSplit:
    '''
    One assistant turn taken apart at the reasoning markers.

    Attributes:
        reasoning (str): Text inside `<|thought_start|>` / `<|thought_end|>`, with
            the effort marker removed. Empty when the turn has no reasoning
            section.
        content (str): Everything outside the reasoning section - the answer plus
            the text before the reasoning opener.
        effort (Optional[str]): Canonical effort level parsed from the effort
            marker ('low' / 'high' / 'max'), or None when there is none.
        complete (bool): False when the reasoning section was never closed, which
            is what a turn looks like in the middle of a stream.
    '''

    reasoning: str = ''
    content: str = ''
    effort: Optional[str] = None
    complete: bool = True


def split_thought(text: str, inside_thought: bool = False) -> ThoughtSplit:
    '''
    Split a decoded assistant turn into its reasoning section and its answer.

    A generated turn usually starts inside the reasoning section, because the
    generation prompt already emitted `<|thought_start|><|effort_...|>`: text
    before `<|thought_end|>` is then reasoning, and the effort marker is not part
    of the turn at all. An unterminated reasoning section is tolerated (streaming,
    or a model that never emitted the end marker), and `inside_thought` says
    whether the text begins inside one.

    Args:
        text (str): Decoded turn, special tokens included.
        inside_thought (bool): Whether `text` starts inside an open reasoning
            section.

    Returns:
        ThoughtSplit: The two parts, the effort level and whether the section
        closed.
    '''
    start = text.find(_THOUGHT_START)
    if start < 0:
        end = text.find(_THOUGHT_END)
        if end < 0:
            if inside_thought:
                return ThoughtSplit(reasoning=text, content='', complete=False)
            return ThoughtSplit(reasoning='', content=text, complete=True)
        return ThoughtSplit(
            reasoning=text[:end],
            content=text[end + len(_THOUGHT_END):],
            complete=True,
        )

    head = text[:start]
    rest = text[start + len(_THOUGHT_START):]

    effort: Optional[str] = None
    for marker, level in EFFORT_TOKENS.items():
        if rest.startswith(marker):
            effort = level
            rest = rest[len(marker):]
            break

    end = rest.find(_THOUGHT_END)
    if end < 0:
        return ThoughtSplit(reasoning=rest, content=head, effort=effort, complete=False)

    return ThoughtSplit(
        reasoning=rest[:end],
        content=head + rest[end + len(_THOUGHT_END):],
        effort=effort,
        complete=True,
    )
