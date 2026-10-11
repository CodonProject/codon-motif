import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from motif.tokenizer.tokenizer import PackedTokenizer
from motif.utils.session import TOKEN_NAMES, resolve_effort, resolve_token_id


__all__ = [
    'ThoughtSplit',
    'DIALOG_TOKEN_NAMES',
    'EFFORT_TOKENS',
    'render_prompt',
    'generation_prompt_ids',
    'stop_token_ids',
    'split_thought',
    'strip_dialog_tokens',
]


#: The markers a conversation layer owns. They are dropped from a reply before it is
#: shown or stored; every other special token (tool calls, modalities, ...) is left
#: alone, so nothing is ever swallowed silently.
DIALOG_TOKEN_NAMES = (
    'im_start', 'im_end',
    'system', 'user', 'tool', 'model',
    'thought_start', 'thought_end', 'effort_low', 'effort_high', 'effort_max',
)

#: Effort marker text -> canonical level, kept in sync with `render_effort` in template.j2.
EFFORT_TOKENS = {
    TOKEN_NAMES[f'effort_{level}']: level for level in ('low', 'high', 'max')
}

_THOUGHT_START = TOKEN_NAMES['thought_start']
_THOUGHT_END = TOKEN_NAMES['thought_end']

_DIALOG_PATTERN = re.compile(
    '|'.join(
        re.escape(text)
        for text in sorted(
            {TOKEN_NAMES[name] for name in DIALOG_TOKEN_NAMES}, key=len, reverse=True
        )
    )
)


def render_prompt(
    tokenizer: PackedTokenizer,
    messages: Sequence[Dict[str, Any]],
    add_generation_prompt: bool = True,
    **template_kwargs,
) -> List[int]:
    '''
    Render a conversation through this library's chat template.

    Args:
        tokenizer (PackedTokenizer): Tokenizer exposing `apply_chat_template`.
        messages (Sequence[Dict[str, Any]]): OpenAI-style messages; every message has
            a `role` and a `content`, and may carry `thought` / `effort` /
            `tool_calls` / `name`.
        add_generation_prompt (bool): Append the assistant generation prompt.
        **template_kwargs: Extra template switches (for example `effort`,
            `enable_thinking`, `enable_think` / `enable_effort` or the modality
            capability flags).

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


def generation_prompt_ids(
    tokenizer: PackedTokenizer,
    effort: Optional[str] = None,
    enable_thinking: bool = True,
    enable_think: bool = True,
    enable_effort: bool = True,
) -> List[int]:
    '''
    Build the assistant generation prompt (`<|im_start|><|model|>` plus the reasoning
    opener) as token ids, without going through the template.

    This is what `ChatSession` feeds when it continues a sequence it already holds
    (see `ChatSession._continued_ids`), so a caller can also compare it with what the
    template itself rendered.

    Args:
        tokenizer (PackedTokenizer): Tokenizer to resolve the tokens against.
        effort (Optional[str]): Thinking effort level; None uses the default level.
        enable_thinking (bool): Open a reasoning section with an effort token when
            True, and an empty `<|thought_start|><|thought_end|>` section when False -
            the same thing `enable_effort=False` means.
        enable_think (bool): Render the reasoning section at all; False mirrors the
            template's `enable_think=false` and drops `<|thought_start|>` as well, so
            the prompt is only `<|im_start|><|model|>`. It wins over
            `enable_thinking` / `enable_effort`.
        enable_effort (bool): Render the effort token inside the section; False
            mirrors the template's `enable_effort=false` and leaves
            `<|thought_start|><|thought_end|>` without a level.

    Returns:
        List[int]: The ids of the generation prompt.

    Raises:
        ValueError: If the vocabulary misses one of the tokens.
    '''
    keys = ['im_start', 'model']
    if not enable_think:
        pass
    elif enable_thinking and enable_effort:
        keys += ['thought_start', f'effort_{resolve_effort(effort)}']
    else:
        keys += ['thought_start', 'thought_end']

    ids: List[int] = []
    for key in keys:
        token_id = resolve_token_id(tokenizer, key)
        if token_id is None:
            raise ValueError(f'the vocabulary has no {TOKEN_NAMES[key]} token')
        ids.append(token_id)
    return ids


def stop_token_ids(tokenizer: PackedTokenizer, extra: Iterable[Union[str, int]] = ()) -> Tuple[int, ...]:
    '''
    Collect the token ids that end one assistant turn.

    Every message of this library's template ends with `<|im_end|>`, and a turn that
    emits a tool call ends with `<|tool_call_end|>`; both are included when the
    vocabulary has them. Callers can add their own ids (or token names).

    Args:
        tokenizer: Tokenizer to resolve the tokens against.
        extra (Iterable[Union[str, int]]): Extra token ids or token names.

    Returns:
        Tuple[int, ...]: The stop ids, in a stable order and without duplicates.
    '''
    names: List[str] = ['im_end', 'tool_call_end']
    ids: List[int] = []
    for name in names:
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
        reasoning (str): Text inside `<|thought_start|>` / `<|thought_end|>`, with the
            effort marker removed. Empty when the turn has no reasoning section.
        content (str): Everything outside the reasoning section - the answer plus the
            text before the reasoning opener.
        effort (Optional[str]): Canonical effort level parsed from the effort marker
            ('low' / 'high' / 'max'), or None when there is none.
        complete (bool): False when the reasoning section was never closed, which is
            what a turn looks like in the middle of a stream.
    '''

    reasoning: str = ''
    content: str = ''
    effort: Optional[str] = None
    complete: bool = True


def split_thought(text: str, inside_thought: bool = False) -> ThoughtSplit:
    '''
    Split a decoded assistant turn into its reasoning section and its answer.

    A generated turn usually *starts inside* the reasoning section, because the
    generation prompt already emitted `<|thought_start|><|effort_...|>`: text before
    `<|thought_end|>` is then reasoning, and the effort marker is not part of the turn
    at all. Text that carries the opener as well (a full render, for instance) is
    handled too.

    While such a turn is still streaming there is no marker to go by at all, so
    `inside_thought` says whether the text begins inside the reasoning section (the
    generation prompt was built with `enable_thinking=True`). Without it, a half
    finished reasoning section is indistinguishable from an answer.

    Tolerates an unterminated reasoning section (streaming, or a model that never
    emitted `<|thought_end|>`): the reasoning keeps growing and `content` stays empty
    until the marker shows up.

    Args:
        text (str): Decoded turn, special tokens included.
        inside_thought (bool): Whether `text` starts inside an open reasoning section.

    Returns:
        ThoughtSplit: The two parts, the effort level and whether the section closed.
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


def strip_dialog_tokens(text: str) -> str:
    '''
    Remove the conversation markers (`<|im_start|>`, `<|im_end|>`, role tags, the
    reasoning section markers and the effort markers) from a decoded turn.

    Tool-call and modality tokens are deliberately kept, so a caller still sees that
    something structural was emitted.

    Args:
        text (str): Decoded text, special tokens included.

    Returns:
        str: The same text without the conversation markers.
    '''
    return _DIALOG_PATTERN.sub('', text)
