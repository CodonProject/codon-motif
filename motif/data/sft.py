import json
import os
import random
from glob import glob
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch

from motif.data.base import MotifDataset
from motif.tokenizer.tokenizer import PackedTokenizer
from motif.utils.session import Session, resolve_effort


_DEFAULT_SYSTEM_PROMPTS = [
    'You are a helpful assistant.',
    'You are a AI.',
    'Answer the user concisely and accurately.',
]

#: Message roles mapped onto the vocabulary the chat template knows. Extend it when a
#: corpus uses another spelling, otherwise `MotifSFT` refuses the row loudly.
ROLE_ALIASES: Dict[str, str] = {
    'human': 'user',
    'user': 'user',
    'assistant': 'model',
    'model': 'model',
    'gpt': 'model',
    'bot': 'model',
    'ai': 'model',
    'system': 'system',
    'developer': 'system',
    'tool': 'tool',
    'function': 'tool',
    'tool_response': 'tool',
    'observation': 'tool',
}

#: Row keys that carry a ready-made conversation instead of a single QA pair.
#: `session` is the schema of the TinySFT corpus: one list of turns, each carrying
#: `role` / `content` / `cot` / `reasoning_effort` / `tool_call_json`.
_CONVERSATION_KEYS = ('conversations', 'messages', 'session')

#: Row keys that make a row a legacy QA pair (`input` + `content`, optionally `cot`).
_PAIR_KEYS = ('input', 'content')

#: Row keys carrying the tool (function) declarations of a conversation.
_TOOL_KEYS = ('tools', 'tool', 'tool_json', 'functions')

#: Message keys carrying the tool calls a turn emitted.
_TOOL_CALL_KEYS = ('tool_calls', 'tool_call', 'tool_call_json')

#: Keys carrying the arguments of one tool call, most specific first.
_TOOL_ARGUMENT_KEYS = ('arguments', 'param', 'parameters', 'args')

#: Message / row keys spelling the reasoning effort of a turn.
EFFORT_KEYS = ('effort', 'reasoning_effort')

#: Effort spelling meaning "this turn carries no reasoning section at all". It is not
#: a level - `motif.utils.session.resolve_effort` does not know it - so it is consumed
#: before a level is collapsed.
EFFORT_NONE = 'none'


def _normalize_role(role: Any) -> str:
    '''
    Normalise one message role onto the vocabulary of the chat template.

    Args:
        role (Any): The role of a row, or None.

    Returns:
        str: One of 'user' / 'model' / 'system' / 'tool'.

    Raises:
        ValueError: If the role is unknown; `template.j2` silently renders nothing for
            an unknown role, which would drop a whole message, so the row is refused
            instead.
    '''
    name = str('user' if role is None else role).strip().lower()
    if not name:
        return 'user'
    alias = ROLE_ALIASES.get(name)
    if alias is None:
        raise ValueError(
            f'unsupported message role {name!r}: add it to motif.data.sft.ROLE_ALIASES '
            f'(known roles: {", ".join(sorted(ROLE_ALIASES))})'
        )
    return alias


def _pick_first(mapping: dict, *keys: str) -> Any:
    '''
    Return the first value of `mapping` that is not None.

    Args:
        mapping (dict): The mapping to read.
        *keys (str): Candidate keys, most specific first.

    Returns:
        Any: The first value that is present and not None, else None.
    '''
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _loads(value: Any) -> Any:
    '''
    Parse a JSON string; return anything else unchanged.

    Args:
        value (Any): A candidate JSON string, or an already-parsed value.

    Returns:
        Any: The parsed value, the original value when it is not a string, or None for
        an empty / blank string.
    '''
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _pick_effort(*values: Any) -> Optional[str]:
    '''
    Return the first non-empty reasoning effort among `values`.

    Args:
        *values (Any): Candidate spellings, most specific first (a message's own
            effort, then the row default, then the dataset default).

    Returns:
        Optional[str]: The spelling, lower-cased and stripped, or None when the turn
        says nothing about effort.
    '''
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def _collapse_effort(raw: Optional[str], where: str) -> Optional[str]:
    '''
    Collapse one effort spelling onto what the chat template understands.

    Args:
        raw (Optional[str]): The spelling carried by a message / row, or None.
        where (str): Human-readable location, used in the error message.

    Returns:
        Optional[str]: None when nothing was said (the template then falls back to its
        own default level), `EFFORT_NONE` for a turn without a reasoning section, else
        'low' / 'high' / 'max'.

    Raises:
        ValueError: If the spelling is neither `EFFORT_NONE` nor a known level or alias
            (`minimal` / `low` / `medium` / `high` / `xhigh` / `max` / `ultra`).
    '''
    if raw is None:
        return None
    if raw == EFFORT_NONE:
        return EFFORT_NONE
    try:
        return resolve_effort(raw)
    except ValueError as exc:
        raise ValueError(f'{where}: {exc}') from exc


def _attach_effort(rendered: dict, raw: Optional[str], thought: Any, where: str) -> None:
    '''
    Write the effort of one model turn onto its rendered message.

    A level becomes `effort`, which the template renders as `<|effort_low|>` /
    `<|effort_high|>` / `<|effort_max|>` after that turn's `<|thought_start|>`.
    `EFFORT_NONE` writes nothing: a model turn without reasoning content renders an
    empty `<|thought_start|><|thought_end|>` section instead.

    Args:
        rendered (dict): The message being built, updated in place.
        raw (Optional[str]): The effort spelling of that turn, if any.
        thought (Any): The reasoning content of that turn.
        where (str): Human-readable location, used in the error messages.

    Raises:
        ValueError: On an unknown spelling, or on `none` next to reasoning content:
            `none` means "no reasoning section", so the two contradict each other and
            the corpus would silently train the model on a lie.
    '''
    effort = _collapse_effort(raw, where)
    if effort == EFFORT_NONE:
        if isinstance(thought, str) and thought.strip():
            raise ValueError(
                f"{where}: reasoning effort 'none' contradicts a non-empty reasoning "
                f'content (drop the reasoning content, or label the turn with a level)'
            )
        return
    if effort is not None:
        rendered['effort'] = effort


def _tool_declarations(row: dict) -> List[dict]:
    '''
    Tool (function) declarations carried by one row.

    Args:
        row (dict): One jsonl / parquet row. The keys `tools` / `tool` / `tool_json` /
            `functions` are accepted, and every entry may be a mapping already or a
            JSON string (TinySFT stores `tool_json` as a list of JSON strings).

    Returns:
        List[dict]: The declarations, ready for the template's `tools` variable (the
        template JSON-encodes every entry itself); an empty list when the row declares
        none. Entries that do not resolve to a mapping are skipped.
    '''
    for key in _TOOL_KEYS:
        value = _loads(row.get(key))
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            continue
        tools = [item for item in (_loads(entry) for entry in value) if isinstance(item, dict)]
        if tools:
            return tools
    return []


def _tool_calls(message: dict) -> List[dict]:
    '''
    Tool calls emitted by one message, in the shape `template.j2` renders.

    The corpus spellings disagree about both the key and the nesting - `tool_calls` /
    `tool_call` / `tool_call_json`, `{name, arguments}` vs
    `{type: 'function', function: {name, arguments}}`, and the arguments key is
    `arguments` or `param` or `parameters` or `args` - so every call is normalised onto
    `{'type': 'function', 'function': {'name', 'arguments'}}`.

    Args:
        message (dict): One message of a ready-made conversation.

    Returns:
        List[dict]: The normalised calls, in order; an empty list when the message
        emits none. An entry without a name is skipped.
    '''
    raw = None
    for key in _TOOL_CALL_KEYS:
        if message.get(key) is not None:
            raw = _loads(message[key])
            break
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    calls: List[dict] = []
    for entry in raw:
        entry = _loads(entry)
        if not isinstance(entry, dict):
            continue
        inner = entry.get('function')
        source = inner if isinstance(inner, dict) else entry
        name = _pick_first(source, 'name') or entry.get('name')
        if not name:
            continue
        arguments = _pick_first(source, *_TOOL_ARGUMENT_KEYS)
        calls.append({
            'type': 'function',
            'function': {'name': name, 'arguments': {} if arguments is None else arguments},
        })
    return calls


def _check_pair_row(row: dict) -> None:
    '''
    Refuse a row that is neither a ready-made conversation nor a legacy QA pair.

    Without this guard such a row renders as an empty user turn plus an empty model
    turn: a silently wasted sample that no loss mask can flag.

    Args:
        row (dict): One row that did not carry a conversation.

    Raises:
        ValueError: If the row carries neither `input` nor `content`.
    '''
    if any(key in row for key in _PAIR_KEYS):
        return
    keys = ', '.join(sorted(str(key) for key in row)[:8])
    raise ValueError(
        f'the row is neither a conversation ({" / ".join(_CONVERSATION_KEYS)}) nor a '
        f'QA pair ({" / ".join(_PAIR_KEYS)}); it only carries: {keys}'
    )


def _truncate_sample(tensors: Dict[str, Any], length: int) -> Dict[str, Any]:
    '''
    Cut one rendered sample down to `length` positions.

    Args:
        tensors (Dict[str, Any]): Fields of one `Session.to_tensors()` dict.
        length (int): Number of positions kept.

    Returns:
        Dict[str, Any]: The truncated fields; `tensors` itself when it already fits.
    '''
    if tensors['input_ids'].size(0) <= length:
        return tensors
    return {
        key: (value[:length].contiguous() if torch.is_tensor(value) else value)
        for key, value in tensors.items()
    }


class MotifSFT(MotifDataset):
    '''
    Build SFT batches from the jsonl files under `folder`.

    `MotifSFT` turns jsonl rows into conversations and renders them, through
    `motif.utils.session.Session`, into pre-batched tensors carrying a loss mask. Four
    row schemas are accepted, and one folder may mix them freely:

        {"input": "what is 2+2?", "content": "4", "cot": "two plus two"}
        {"system": "...", "conversations": [{"role": "human", "content": "..."}, ...]}
        {"messages": [{"role": "system", "content": "..."}, {"role": "user", ...}]}
        {"system": "...", "tool": [...],
         "session": [{"role": "human", "content": "..."},
                     {"role": "model", "cot": "...", "reasoning_effort": "high"}, ...]}

    `conversations` (ShareGPT), `messages` (LLaMA-Factory) and `session` rows are
    ready-made conversations: such a row is kept whole and becomes exactly one sample,
    never merged with a neighbour, so a corpus that is already conversational
    (role-play data, chat distillations) is only rendered, never re-grouped. Legacy
    `input` / `content` rows are shuffled and merged into 1-3 turn conversations by
    `two_turn_prob` / `three_turn_prob`. Every sample gets a system turn: the row's own
    top-level `system`, or its leading `system` message, when it has one, otherwise a
    random entry of `system_prompts`. Message roles are normalised through
    `ROLE_ALIASES` (`human` -> `user`, `assistant` -> `model`, ...); a role that is not
    known raises instead of being silently dropped by the chat template.

    Every model turn may carry its **own** reasoning effort (`effort` /
    `reasoning_effort` on the message, else on the row, else `MotifSFT(effort=...)`,
    else the template's own default). A level is rendered as `<|effort_low|>` /
    `<|effort_high|>` / `<|effort_max|>` after that turn's `<|thought_start|>`;
    `EFFORT_NONE` is not a level: it means "this turn has no reasoning section", and a
    turn without reasoning content renders an empty
    `<|thought_start|><|thought_end|>` section, so every model turn has one. Tool
    declarations (`tools` / `tool` / `tool_json`) are rendered into the leading system
    block, and the tool calls of a turn (`tool_calls` / `tool_call` / `tool_call_json`)
    are normalised onto `{name, arguments}` before the template renders them.

    The class doubles as the pipeline's resumable data flow: the batch order of every
    epoch is derived deterministically from `seed + epoch` (not truly random, so a whole
    permutation never has to be stored), and the cursor only records "which batch of the
    current epoch was produced last". `state_dict()` therefore stores `{epoch, cursor}`.
    It is a training stream rather than a general map-style dataset:

        ds = MotifSFT(folder, tokenizer, pad_length=2048, batch_size=8)
        for batch in ds.iter_epoch(0):      # run epoch 0 from the beginning
            ...
        ds.state_dict()                     # {'epoch': 0, 'cursor': batches consumed}
        ds.load_state_dict(saved)           # continue in place, neither replay nor skip

    Plain `for batch in ds` is equivalent to the **remaining part** of
    `for batch in ds.iter_epoch(ds.epoch)`: once an epoch is finished, `iter()` yields
    nothing until `set_epoch` / `iter_epoch` switches to the next epoch explicitly.
    Rendering is reachable on its own, without consuming the stream:
    `build_messages(group)` returns the chat messages of a group,
    `session_for(messages)` renders them, and `render_group(group)` is one padded
    sample. The periodic sanity probe of `motif/pipeline/sanity.py` uses them to score
    the very tokens training scores.

    Args:
        folder (str): Directory holding the jsonl files.
        tokenizer (PackedTokenizer): Tokenizer already carrying the chat template.
        pad_length (int): Length every sample is padded / truncated to.
        batch_size (int): Number of conversation groups per batch.
        two_turn_prob (float): Probability of merging two lines into a two-turn
            conversation (pair rows only).
        three_turn_prob (float): Probability of merging three lines into a three-turn
            conversation (pair rows only).
        system_prompts (Optional[Sequence[str]]): Candidate system prompts, used when
            a row does not bring its own.
        pattern (str): File-name glob pattern.
        recursive (bool): Whether to scan sub-directories as well.
        limit (Optional[int]): Maximum number of rows read from **each** matching file;
            None reads them whole. Useful to subset a multi-GB corpus without loading
            all of it (the rows of every file are read in file-name order anyway).
        seed (int): Seed of the shuffling / grouping / per-epoch order.
    '''

    def __init__(
        self,
        folder: str,
        tokenizer: PackedTokenizer,
        pad_length: int,
        batch_size: int,
        two_turn_prob: float = 0.2,
        three_turn_prob: float = 0.1,
        system_prompts: Optional[Sequence[str]] = None,
        pattern: str = '*.jsonl',
        recursive: bool = True,
        limit: Optional[int] = None,
        seed: int = 42,
        effort: Optional[str] = None,
    ) -> None:
        '''
        Load the jsonl corpus, shuffle it and pre-build the conversation groups.

        Args:
            folder (str): Directory holding the jsonl files.
            tokenizer (PackedTokenizer): Tokenizer carrying the chat template.
            pad_length (int): Length every sample is padded / truncated to.
            batch_size (int): Number of conversation groups per batch.
            two_turn_prob (float): Probability of building a two-turn conversation.
            three_turn_prob (float): Probability of building a three-turn
                conversation.
            system_prompts (Optional[Sequence[str]]): Candidate system prompts;
                defaults to `_DEFAULT_SYSTEM_PROMPTS`.
            pattern (str): File-name glob pattern.
            recursive (bool): Whether to scan sub-directories as well.
            seed (int): Seed used for shuffling, grouping and the epoch order.
            effort (Optional[str]): Default reasoning effort for turns that do not
                carry one (`low` / `high` / `max`, or `none` for "no reasoning
                section"); None leaves the chat template's own default in place.

        Raises:
            ValueError: If a turn probability is negative, if the two probabilities
                exceed 1.0, if `pad_length` / `batch_size` is not positive, if a row
                carries a message role that is not in `ROLE_ALIASES`, or if `effort` is
                not a known level.
            RuntimeError: If no jsonl sample could be loaded.
        '''
        if two_turn_prob < 0 or three_turn_prob < 0:
            raise ValueError('turn probabilities must be non-negative')
        if two_turn_prob + three_turn_prob > 1.0:
            raise ValueError('two_turn_prob + three_turn_prob must not exceed 1.0')
        if pad_length <= 0:
            raise ValueError('pad_length must be positive')
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')

        self.folder = folder
        self.tokenizer = tokenizer
        self.pad_length = pad_length
        self.batch_size = batch_size
        self.two_turn_prob = two_turn_prob
        self.three_turn_prob = three_turn_prob
        self.system_prompts = (
            list(system_prompts) if system_prompts else list(_DEFAULT_SYSTEM_PROMPTS)
        )
        self.pattern = pattern
        self.recursive = recursive
        self.limit = limit
        self.seed = seed
        self.effort = effort
        if effort is not None:
            _collapse_effort(str(effort).strip().lower(), 'MotifSFT(effort=...)')

        self.samples = self._load_jsonl(folder, pattern, recursive, limit)
        if not self.samples:
            raise RuntimeError(f'no jsonl samples loaded from {folder!r}')
        random.Random(seed).shuffle(self.samples)
        self.groups = self._build_groups(random.Random(seed + 1))

        self._n_batches = len(self.groups) // self.batch_size
        self._epoch = 0
        self._cursor = 0
        self._order_cache: Dict[int, List[int]] = {}

    @staticmethod
    def _load_jsonl(folder: str, pattern: str, recursive: bool, limit: Optional[int] = None) -> List[dict]:
        '''
        Read every matching file into a flat list of rows.

        A `.parquet` file (the shape TinySFT ships) is read with pyarrow, imported
        lazily so a jsonl corpus never needs it; every other suffix is read as jsonl.
        Both paths end up as the same rows: one dict per sample, its list columns (for
        example TinySFT's `session`) as plain Python lists again.

        Args:
            folder (str): Directory to scan.
            pattern (str): File-name glob pattern.
            recursive (bool): Whether to scan sub-directories as well.
            limit (Optional[int]): Maximum number of rows read from each file; None
                reads every row.

        Returns:
            List[dict]: One dict per row, in file-name order.
        '''
        if recursive:
            paths = sorted(glob(os.path.join(folder, '**', pattern), recursive=True))
        else:
            paths = sorted(glob(os.path.join(folder, pattern)))

        rows: List[dict] = []
        for path in paths:
            if path.lower().endswith('.parquet'):
                rows.extend(MotifSFT._load_parquet(path, limit))
                continue
            taken = 0
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
                    taken += 1
                    if limit is not None and taken >= limit:
                        break
        return rows

    @staticmethod
    def _load_parquet(path: str, limit: Optional[int] = None) -> List[dict]:
        '''
        Read one parquet file into row dicts.

        Args:
            path (str): The parquet file.
            limit (Optional[int]): Maximum number of rows to read; None reads them all.

        Returns:
            List[dict]: One dict per row, in file order.

        Raises:
            ImportError: If pyarrow is not installed.
        '''
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                f'reading {os.path.basename(path)!r} needs pyarrow: pip install pyarrow'
            ) from exc

        rows: List[dict] = []
        handle = pq.ParquetFile(path)
        for batch in handle.iter_batches(batch_size=4096):
            rows.extend(batch.to_pylist())
            if limit is not None and len(rows) >= limit:
                del rows[limit:]
                break
        return rows

    @staticmethod
    def _conversation(row: dict) -> Optional[List[dict]]:
        '''
        Return the ready-made conversation of a row, when it carries one.

        Args:
            row (dict): One jsonl row.

        Returns:
            Optional[List[dict]]: The messages of a `conversations` / `messages` row
            (non-empty, dict entries only), or None for a legacy pair row.
        '''
        for key in _CONVERSATION_KEYS:
            value = row.get(key)
            if isinstance(value, list):
                messages = [m for m in value if isinstance(m, dict)]
                if messages:
                    return messages
        return None

    @staticmethod
    def _render_message(
        message: dict,
        row: Optional[dict] = None,
        default: Optional[str] = None,
    ) -> dict:
        '''
        Convert one corpus message into the shape the chat template expects.

        `role` / `content` are the canonical keys; the ShareGPT spellings `from` /
        `value` are accepted as fallbacks, and the role is normalised (`human` ->
        `user`, `assistant` -> `model`, ...). The reasoning content is accepted as
        `reasoning_content`, `thought` or `cot`; the effort comes from the message
        first, then from `row`, then from `default`; tool calls are normalised onto the
        shape the template renders.

        Args:
            message (dict): One message of a `conversations` / `messages` / `session`
                row.
            row (Optional[dict]): The row the message belongs to, used as the effort
                fallback.
            default (Optional[str]): Dataset-wide effort fallback.

        Returns:
            Dict[str, Any]: `{role, content}`, plus `reasoning_content` / `effort` /
            `tool_calls` / `name` when the message carries them.

        Raises:
            ValueError: If the effort spelling is unknown, or if a turn labelled `none`
                still carries reasoning content.
        '''
        content = message.get('content')
        if content is None:
            content = message.get('value')
        role = _normalize_role(message.get('role') or message.get('from'))
        rendered: Dict[str, Any] = {
            'role': role,
            'content': '' if content is None else content,
        }
        thought = message.get('reasoning_content')
        if thought is None:
            thought = message.get('thought')
        if thought is None:
            thought = message.get('cot')
        if thought:
            rendered['reasoning_content'] = thought

        if role == 'model':
            _attach_effort(
                rendered,
                _pick_effort(
                    *(message.get(key) for key in EFFORT_KEYS),
                    *((row or {}).get(key) for key in EFFORT_KEYS),
                    default,
                ),
                thought,
                'MotifSFT: a conversation message',
            )
            calls = _tool_calls(message)
            if calls:
                rendered['tool_calls'] = calls
        elif role == 'tool':
            name = message.get('name')
            if name:
                rendered['name'] = name
        return rendered

    def _chat_group(self, row: dict, messages: List[dict], rng: random.Random) -> dict:
        '''
        Wrap one ready-made conversation into a group.

        The conversation's own system turn wins: either the top-level `system` string
        or a leading `system` message. Without one, a random entry of `system_prompts`
        is prepended, so a chat sample reaches the model with a system turn exactly
        like a pair sample does.

        Args:
            row (dict): The jsonl row.
            messages (List[dict]): Its message list, as returned by `_conversation`.
            rng (random.Random): Generator used when a default prompt is needed.

        Returns:
            dict: A group holding `kind`, `system` and the rendered `messages`.
        '''
        system = row.get('system')
        system = system.strip() if isinstance(system, str) else ''
        if _normalize_role(messages[0].get('role') or messages[0].get('from')) == 'system':
            system = ''
        elif not system:
            system = rng.choice(self.system_prompts)
        return {
            'kind': 'chat',
            'system': system,
            'tools': _tool_declarations(row),
            'messages': [
                self._render_message(m, row=row, default=self.effort) for m in messages
            ],
        }

    def _build_groups(self, rng: random.Random) -> List[dict]:
        '''
        Merge the shuffled rows into `(system, turns)` conversation groups.

        Pair rows are merged in runs of one to three, as before; a row that already
        carries a conversation becomes exactly one group and is never merged with a
        neighbour.

        Args:
            rng (random.Random): Generator used for the turn counts and the system
                prompts.

        Returns:
            List[dict]: One dict per group, holding a `system` prompt and either a
            list of `turns` (pair rows) or `kind='chat'` plus `messages` (ready-made
            conversations).
        '''
        groups: List[dict] = []
        total = len(self.samples)
        i = 0
        while i < total:
            conversation = self._conversation(self.samples[i])
            if conversation is not None:
                groups.append(self._chat_group(self.samples[i], conversation, rng))
                i += 1
                continue

            _check_pair_row(self.samples[i])

            r = rng.random()
            remain = total - i
            if r < self.three_turn_prob and remain >= 3:
                n = 3
            elif r < self.three_turn_prob + self.two_turn_prob and remain >= 2:
                n = 2
            else:
                n = 1
            turns: List[dict] = []
            while len(turns) < n and i + len(turns) < total:
                candidate = self.samples[i + len(turns)]
                if self._conversation(candidate) is not None:
                    break
                turns.append(candidate)
            tools: List[dict] = []
            for turn in turns:
                tools = _tool_declarations(turn)
                if tools:
                    break
            groups.append({
                'system': rng.choice(self.system_prompts),
                'tools': tools,
                'turns': turns,
            })
            i += len(turns)
        return groups

    def session_for(self, messages: Sequence[dict]) -> Session:
        '''
        Open a `Session` on this dataset's tokenizer and add `messages`.

        This is the public render seam: training, the tests and the periodic sanity
        probe all build their conversations here, so a probe sees exactly the tokens
        the optimiser does.

        Args:
            messages (Sequence[dict]): Chat messages (`{'role', 'content'}`, plus
                `reasoning_content` / `effort` / `tool_calls` / `name` when the corpus
                carries them). A `tools` key on the **first** message is handed to the
                chat template, which renders the declarations in its leading system
                block.

        Returns:
            Session: The rendered conversation.
        '''
        session = Session(self.tokenizer)
        for index, message in enumerate(messages):
            message = dict(message)
            tools = message.pop('tools', None)
            session.add_message(message, tools=tools if index == 0 else None)
        return session

    def build_messages(self, group: dict) -> List[dict]:
        '''
        The chat messages of one group, in render order.

        The system prompt comes first, when the group has one. A pair group then adds
        every turn as a user message followed by a model message; a `cot` field becomes
        the model's `reasoning_content`. A chat group adds its already-rendered
        messages in order.

        Args:
            group (dict): Group produced by `_build_groups`.

        Returns:
            List[dict]: Fresh message dicts; mutating them leaves the group untouched.
            The tool declarations of the group (if any) ride along on the first
            message, under `tools`.
        '''
        messages: List[dict] = []
        system = group.get('system')
        if system:
            messages.append({'role': 'system', 'content': system})
        if group.get('kind') == 'chat':
            messages.extend(dict(message) for message in group['messages'])
        else:
            for turn in group['turns']:
                messages.append({'role': 'user', 'content': turn.get('input', '')})
                messages.append(self._pair_model_message(turn))
        tools = group.get('tools')
        if tools and messages:
            messages[0]['tools'] = list(tools)
        return messages

    def _pair_model_message(self, turn: dict) -> dict:
        '''
        Build the model message of one legacy pair row.

        Args:
            turn (dict): A row carrying `input` / `content` / `cot`, and optionally its
                own effort and tool calls.

        Returns:
            Dict[str, Any]: `{role, content}`, plus `reasoning_content` / `effort` /
            `tool_calls` when the row carries them.

        Raises:
            ValueError: If the row's effort spelling is unknown, or if it is labelled
                `none` while the row still carries reasoning content.
        '''
        model_msg: dict = {'role': 'model', 'content': turn.get('content', '')}
        cot = turn.get('cot')
        if cot is None:
            cot = turn.get('reasoning_content')
        if cot:
            model_msg['reasoning_content'] = cot
        _attach_effort(
            model_msg,
            _pick_effort(*(turn.get(key) for key in EFFORT_KEYS), self.effort),
            cot,
            'MotifSFT: a QA pair row',
        )
        calls = _tool_calls(turn)
        if calls:
            model_msg['tool_calls'] = calls
        return model_msg

    def _build_session(self, group: dict) -> Session:
        '''
        Render one conversation group as a `Session`.

        Args:
            group (dict): Group produced by `_build_groups`.

        Returns:
            Session: The rendered conversation.
        '''
        return self.session_for(self.build_messages(group))

    def render_group(self, group: dict) -> Dict[str, Any]:
        '''
        Render one group into the padded tensor dict training consumes.

        Args:
            group (dict): Group produced by `_build_groups`.

        Returns:
            Dict[str, Any]: The tensor fields of that sample.
        '''
        tensors = self._build_session(group).to_tensors(pad_to=self.pad_length)
        return _truncate_sample(tensors, self.pad_length)

    def _build_sample(self, group: dict) -> Dict[str, Any]:
        '''
        Render one group into a padded tensor dict (alias of `render_group`).

        Args:
            group (dict): Group produced by `_build_groups`.

        Returns:
            Dict[str, Any]: The tensor fields of that sample.
        '''
        return self.render_group(group)

    def _build_batch(self, batch_idx: int) -> Dict[str, Any]:
        '''
        Merge the samples of one batch.

        Every field is a tensor holding one entry per row, so the batch stacks them
        along a new leading dimension.

        Args:
            batch_idx (int): Index into `self.groups`.

        Returns:
            Dict[str, Any]: The batched fields.
        '''
        start = batch_idx * self.batch_size
        samples = [self._build_sample(g) for g in self.groups[start:start + self.batch_size]]
        return {key: torch.stack([s[key] for s in samples]) for key in samples[0]}

    @property
    def epoch(self) -> int:
        '''
        Returns:
            int: Index of the current epoch (or of the one `set_epoch` switched to
            last).
        '''
        return self._epoch

    @property
    def cursor(self) -> int:
        '''
        Returns:
            int: Number of batches already consumed inside the current epoch.
        '''
        return self._cursor

    def _order(self, epoch: int) -> List[int]:
        '''
        Batch order of epoch `epoch`, derived deterministically from seed + epoch
        (reproducible, so it never has to be checkpointed).

        Args:
            epoch (int): Epoch index.

        Returns:
            List[int]: The order the batches are visited in.
        '''
        cached = self._order_cache.get(epoch)
        if cached is not None:
            return cached
        order = list(range(self._n_batches))
        random.Random(self.seed * 1_000_003 + epoch).shuffle(order)
        self._order_cache[epoch] = order
        return order

    def set_epoch(self, epoch: int) -> 'MotifSFT':
        '''
        Switch to epoch `epoch` and reset the cursor (the order follows from
        seed + epoch).

        Args:
            epoch (int): Epoch index to switch to.

        Returns:
            MotifSFT: This dataset, for chaining.
        '''
        self._epoch = int(epoch)
        self._cursor = 0
        return self

    def rewind(self) -> 'MotifSFT':
        '''
        Reset only the cursor, leaving the epoch alone (to re-run the current epoch).

        Returns:
            MotifSFT: This dataset, for chaining.
        '''
        self._cursor = 0
        return self

    def state_dict(self) -> Dict[str, Any]:
        '''
        Returns:
            Dict[str, Any]: `{epoch, cursor}`, enough to continue exactly where the
            run stopped.
        '''
        return {'epoch': int(self._epoch), 'cursor': int(self._cursor)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        '''
        Restore the epoch and the cursor.

        Args:
            state (Dict[str, Any]): State produced by `state_dict()`.
        '''
        self._epoch = int(state.get('epoch', 0))
        self._cursor = int(state.get('cursor', 0))

    def __len__(self) -> int:
        '''
        Returns:
            int: Number of batches in one epoch.
        '''
        return self._n_batches

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        '''
        Fetch batch `idx` following the order of the current epoch (for debugging and
        sampling).

        Args:
            idx (int): Batch index; negative values count from the end.

        Returns:
            Dict[str, Any]: The batched fields.

        Raises:
            IndexError: If `idx` is outside the epoch.
        '''
        if idx < 0:
            idx += self._n_batches
        if idx < 0 or idx >= self._n_batches:
            raise IndexError(f'index {idx} out of range for {self._n_batches} batches')
        return self._build_batch(self._order(self._epoch)[idx])

    def iter_epoch(self, epoch: int) -> Iterator[Dict[str, Any]]:
        '''
        Move the cursor to the start of epoch `epoch` and iterate it completely.

        Args:
            epoch (int): Epoch index.

        Returns:
            Iterator[Dict[str, Any]]: The batches of that epoch.
        '''
        return iter(self.set_epoch(epoch))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        '''
        Iterate the remaining batches of the current epoch, starting from the cursor
        (finishing it does not switch to the next epoch automatically).

        Yields:
            Dict[str, Any]: The batched fields.
        '''
        order = self._order(self._epoch)
        i = self._cursor
        while i < len(order):
            batch = self._build_batch(order[i])
            self._cursor = i + 1
            yield batch
            i += 1
