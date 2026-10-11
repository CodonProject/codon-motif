import json
import os
import shutil
import tempfile

import torch

from motif.data import MotifSFT
from motif.data.sft import ROLE_ALIASES, _DEFAULT_SYSTEM_PROMPTS, _normalize_role
from motif.utils.session import Session

from support import make_tokenizer, write_jsonl

def _folder(rows, name: str = 'part.jsonl') -> str:
    return os.path.dirname(write_jsonl(tempfile.mkdtemp(prefix='motif_sft_chat_'), rows, name))

def _dataset(rows, tokenizer=None, name: str = 'part.jsonl', **kwargs) -> MotifSFT:
    tokenizer = tokenizer or make_tokenizer()
    kwargs.setdefault('pad_length', 128)
    kwargs.setdefault('batch_size', 2)
    kwargs.setdefault('seed', 7)
    return MotifSFT(_folder(rows, name), tokenizer, **kwargs)

def _roles(dataset: MotifSFT, group: dict) -> list:
    return [message.role for message in dataset._build_session(group).messages]

CHAT_ROW = {
    'conversations': [
        {'role': 'human', 'content': 'who are you?'},
        {'role': 'model', 'content': 'a small test model'},
        {'role': 'human', 'content': 'say hi'},
        {'role': 'model', 'content': 'hi there'},
    ],
}
SYSTEM_ROW = {
    'system': 'you are a pirate',
    'conversations': [
        {'role': 'human', 'content': 'ahoy'},
        {'role': 'model', 'content': 'arr, matey'},
    ],
}
MESSAGES_ROW = {
    'messages': [
        {'role': 'system', 'content': 'be nice'},
        {'role': 'user', 'content': 'how are you'},
        {'role': 'assistant', 'content': 'fine thanks'},
    ],
}

def test_conversations_row_becomes_one_group():
    dataset = _dataset([CHAT_ROW] * 3)
    assert len(dataset.groups) == 3, f'3 conversations should make exactly 3 groups, got {len(dataset.groups)}'
    for group in dataset.groups:
        assert group.get('kind') == 'chat'
        assert 'messages' in group and 'turns' not in group
    assert _roles(dataset, dataset.groups[0]) == ['system', 'user', 'model', 'user', 'model']

def test_top_level_system_is_used():
    dataset = _dataset([SYSTEM_ROW] * 2)
    for group in dataset.groups:
        assert group['system'] == 'you are a pirate', group['system']
        session = dataset._build_session(group)
        assert session.messages[0].role == 'system'
        assert [m.role for m in session.messages] == ['system', 'user', 'model']

def test_messages_row_keeps_its_own_system_turn():
    dataset = _dataset([MESSAGES_ROW] * 2)
    for group in dataset.groups:
        assert group['system'] == ''
        assert [m['role'] for m in group['messages']] == ['system', 'user', 'model']
    assert _roles(dataset, dataset.groups[0]) == ['system', 'user', 'model']

def test_embedded_system_wins_over_top_level():
    row = dict(MESSAGES_ROW, system='ignored top level prompt')
    dataset = _dataset([row])
    group = dataset.groups[0]
    assert group['system'] == ''
    session = dataset._build_session(group)
    assert [m.role for m in session.messages] == ['system', 'user', 'model']

def test_chat_row_without_system_gets_a_candidate_prompt():
    dataset = _dataset([CHAT_ROW] * 4, system_prompts=['S-ONE', 'S-TWO'])
    drawn = {group['system'] for group in dataset.groups}
    assert drawn and drawn <= {'S-ONE', 'S-TWO'}, drawn
    assert _roles(dataset, dataset.groups[0]) == ['system', 'user', 'model', 'user', 'model']

def test_sharegpt_from_value_rows():
    row = {
        'conversations': [
            {'from': 'human', 'value': 'say hi'},
            {'from': 'gpt', 'value': 'hi there'},
        ],
    }
    dataset = _dataset([row])
    group = dataset.groups[0]
    assert [m['role'] for m in group['messages']] == ['user', 'model']
    assert group['messages'][1]['content'] == 'hi there'

def test_empty_conversation_list_falls_back_to_pair_row():
    row = {'conversations': [], 'input': 'what is 2+2?', 'content': '4'}
    dataset = _dataset([row])
    group = dataset.groups[0]
    assert group.get('kind') != 'chat'
    assert group['turns'][0]['input'] == 'what is 2+2?'
    assert _roles(dataset, group) == ['system', 'user', 'model']

def test_role_aliases():
    assert _normalize_role('human') == 'user'
    assert _normalize_role('User') == 'user'
    assert _normalize_role('assistant') == 'model'
    assert _normalize_role('gpt') == 'model'
    assert _normalize_role('developer') == 'system'
    assert _normalize_role('function') == 'tool'
    assert _normalize_role(None) == 'user'
    assert _normalize_role('') == 'user'
    assert ROLE_ALIASES['model'] == 'model'

def test_unknown_role_raises():
    row = {'conversations': [{'role': 'narrator', 'content': 'once upon a time'},
                             {'role': 'model', 'content': 'the end'}]}
    try:
        _dataset([row])
    except ValueError as exc:
        assert 'unsupported message role' in str(exc), str(exc)
        assert 'narrator' in str(exc), str(exc)
    else:
        raise AssertionError('an unknown role should raise ValueError')

PAIR_ROWS = [
    {'input': 'what is 2+2?', 'cot': 'two plus two', 'content': '4'},
    {'input': 'say hi', 'content': 'hi there'},
    {'input': 'name a color', 'content': 'blue'},
    {'input': 'capital of France?', 'cot': 'France', 'content': 'Paris'},
] + [{'input': f'question {i}?', 'content': f'answer {i}'} for i in range(12)]

def test_pair_rows_keep_merging_and_cot():
    dataset = _dataset(PAIR_ROWS, batch_size=4, two_turn_prob=0.4, three_turn_prob=0.2)
    assert all('messages' not in group for group in dataset.groups), 'pair rows skip the chat path'
    assert sum(len(group['turns']) for group in dataset.groups) == len(dataset.samples)
    assert {group['system'] for group in dataset.groups} <= set(_DEFAULT_SYSTEM_PROMPTS)
    assert max(len(group['turns']) for group in dataset.groups) <= 3
    assert max(len(group['turns']) for group in dataset.groups) >= 2

def test_pair_row_cot_becomes_reasoning():
    dataset = _dataset(PAIR_ROWS, batch_size=1, two_turn_prob=0.0, three_turn_prob=0.0)
    assert all(len(group['turns']) == 1 for group in dataset.groups)
    session = Session(make_tokenizer())
    thought_start, thought_end = session.token_id('thought_start'), session.token_id('thought_end')

    cot_group = next(g for g in dataset.groups if g['turns'][0].get('cot'))
    model_msg = dataset._build_session(cot_group).messages[-1]
    assert model_msg.role == 'model'
    assert thought_start in model_msg.ids and thought_end in model_msg.ids, 'cot not rendered'
    assert cot_group['turns'][0]['cot'] in make_tokenizer().decode(model_msg.ids)

    plain_group = next(g for g in dataset.groups if not g['turns'][0].get('cot'))
    plain_ids = dataset._build_session(plain_group).messages[-1].ids
    assert thought_start in plain_ids and thought_end in plain_ids, 'empty thought block missing'
    for level in ('low', 'high', 'max'):
        assert session.token_id(f'effort_{level}') not in plain_ids, 'effort token in empty block'

def test_load_jsonl_returns_raw_rows():
    rows = [CHAT_ROW, PAIR_ROWS[0]]
    folder = _folder(rows)
    loaded = MotifSFT._load_jsonl(folder, '*.jsonl', True)
    assert [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in loaded] == \
           [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows]
    assert 'conversations' in loaded[0]

def test_limit_is_per_file_for_chat_rows():
    base = tempfile.mkdtemp(prefix='motif_sft_chat_limit_')
    try:
        for name in ('a.jsonl', 'b.jsonl'):
            write_jsonl(base, [CHAT_ROW] * 5, name)
        dataset = MotifSFT(base, make_tokenizer(), pad_length=64, batch_size=2, limit=2)
        assert len(dataset.samples) == 4, len(dataset.samples)
        assert len(dataset.groups) == 4
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_mixed_folder_keeps_both_kinds_apart():
    rows = [PAIR_ROWS[0], CHAT_ROW, PAIR_ROWS[1], SYSTEM_ROW, MESSAGES_ROW, PAIR_ROWS[2]]
    dataset = _dataset(rows, batch_size=2, two_turn_prob=1.0, three_turn_prob=0.0)
    chats = [group for group in dataset.groups if group.get('kind') == 'chat']
    pairs = [group for group in dataset.groups if group.get('kind') != 'chat']

    assert len(chats) == 3, f'3 ready-made conversations should make 3 chat groups, got {len(chats)}'
    assert all('messages' in g and 'turns' not in g for g in chats)
    assert all('turns' in g and 'messages' not in g for g in pairs)
    contents = sorted(json.dumps(g['messages'], sort_keys=True, ensure_ascii=False) for g in chats)
    assert contents == sorted([
        json.dumps([dict(role='user', content='who are you?'),
                    dict(role='model', content='a small test model'),
                    dict(role='user', content='say hi'),
                    dict(role='model', content='hi there')], sort_keys=True, ensure_ascii=False),
        json.dumps([dict(role='user', content='ahoy'),
                    dict(role='model', content='arr, matey')], sort_keys=True, ensure_ascii=False),
        json.dumps([dict(role='system', content='be nice'),
                    dict(role='user', content='how are you'),
                    dict(role='model', content='fine thanks')], sort_keys=True, ensure_ascii=False),
    ]), contents

    batches = list(dataset.iter_epoch(0))
    assert len(batches) == len(dataset) == len(dataset.groups) // 2
    for batch in batches:
        assert batch['input_ids'].shape == (2, 128)
        assert batch['labels'].shape == (2, 128)
        assert batch['attention_mask'].dtype == torch.bool or batch['attention_mask'].dtype == torch.long

def test_chat_batches_carry_supervision():
    dataset = _dataset([CHAT_ROW, MESSAGES_ROW], pad_length=256, batch_size=1)
    for index, batch in enumerate(dataset.iter_epoch(0)):
        supervised = int((batch['labels'] != -100).sum())
        assert supervised > 0, f'chat sample {index} carries no supervision signal at all'
        padding = batch['attention_mask'] == 0
        assert bool((batch['labels'][padding] == -100).all()), 'padding carries supervision'

def test_long_conversation_is_truncated_to_pad_length():
    long_row = {
        'conversations': [
            item
            for turn in range(6)
            for item in ({'role': 'human', 'content': f'turn {turn} ' + 'word ' * 30},
                         {'role': 'model', 'content': f'reply {turn} ' + 'word ' * 30})
        ],
    }
    short = _dataset([long_row], pad_length=32, batch_size=1)
    batch = next(iter(short.iter_epoch(0)))
    assert batch['input_ids'].shape == (1, 32)
    assert batch['labels'].shape == (1, 32)

    wide = _dataset([long_row], pad_length=512, batch_size=1)
    wide_batch = next(iter(wide.iter_epoch(0)))
    assert int((wide_batch['labels'] != -100).sum()) > 0

def test_chat_resume_is_exact():
    rows = [CHAT_ROW, MESSAGES_ROW, SYSTEM_ROW, dict(CHAT_ROW, system='another prompt')]
    reference = _dataset(rows, batch_size=1)
    full = list(reference.iter_epoch(0))
    assert len(full) == 4

    dataset = _dataset(rows, batch_size=1)
    iterator = iter(dataset.set_epoch(0))
    partial = [next(iterator) for _ in range(2)]
    state = dataset.state_dict()
    del iterator
    assert state == {'epoch': 0, 'cursor': 2}

    resumed = _dataset(rows, batch_size=1)
    resumed.load_state_dict(state)
    rest = list(resumed)
    assert len(partial) + len(rest) == len(full)
    for got, want in zip(partial + rest, full):
        assert torch.equal(got['input_ids'], want['input_ids']), 'chat batches misaligned at the break'
        assert torch.equal(got['labels'], want['labels'])


# --- effort / tool-call schema ---
import os
import tempfile

from motif.data import MotifSFT
from motif.data.sft import EFFORT_NONE, _collapse_effort, _tool_calls, _tool_declarations
from motif.utils.session import Session

from support import make_tokenizer, write_jsonl

def _ids(dataset: MotifSFT, group: dict) -> list:
    session = dataset._build_session(group)
    return [token for message in session.messages for token in message.ids]

def _tok(name: str) -> int:
    return Session(make_tokenizer()).token_id(name)

def _model_messages(dataset: MotifSFT, group: dict) -> list:
    return [m for m in dataset.build_messages(group) if m['role'] == 'model']

def _raises(rows, needle: str, where: str, **kwargs) -> str:
    try:
        _dataset(rows, **kwargs)
    except ValueError as exc:
        text = str(exc)
        assert needle in text, f'{where}: error message does not contain {needle!r}: {text}'
        return text
    raise AssertionError(f'{where}: a ValueError should have been raised')

def test_effort_spellings_collapse():
    assert _collapse_effort(None, 'x') is None
    assert _collapse_effort('none', 'x') == EFFORT_NONE
    assert _collapse_effort('low', 'x') == 'low'
    assert _collapse_effort('minimal', 'x') == 'low'
    assert _collapse_effort('medium', 'x') == 'high'
    assert _collapse_effort('high', 'x') == 'high'
    assert _collapse_effort('xhigh', 'x') == 'high'
    assert _collapse_effort('ultra', 'x') == 'max'

def test_tool_declarations_parse_json_strings():
    decl = {'name': 'bash', 'parameters': {'type': 'object'}}
    assert _tool_declarations({'tool_json': [json.dumps(decl)]}) == [decl]
    assert _tool_declarations({'tool': [decl]}) == [decl]
    assert _tool_declarations({'tools': []}) == []
    assert _tool_declarations({}) == []

def test_tool_calls_are_normalised():
    calls = _tool_calls({'tool_call_json': json.dumps([
        {'name': 'bash', 'param': {'cmd': 'ls'}},
        {'type': 'function', 'function': {'name': 'read', 'arguments': {'path': 'a'}}},
        {'name': 'noargs'},
    ])})
    assert calls == [
        {'type': 'function', 'function': {'name': 'bash', 'arguments': {'cmd': 'ls'}}},
        {'type': 'function', 'function': {'name': 'read', 'arguments': {'path': 'a'}}},
        {'type': 'function', 'function': {'name': 'noargs', 'arguments': {}}},
    ], calls
    assert _tool_calls({'tool_calls': [{'name': 'x', 'arguments': {}}]}) == [
        {'type': 'function', 'function': {'name': 'x', 'arguments': {}}}
    ]
    assert _tool_calls({}) == []

TOOL_DECL = {'type': 'function',
             'function': {'name': 'bash', 'description': 'run a shell command',
                          'parameters': {'type': 'object', 'properties': {}}}}

SESSION_ROW = {
    'type': 'SFT',
    'system': 'you are a careful assistant',
    'tool_json': [json.dumps(TOOL_DECL)],
    'session': [
        {'role': 'human', 'content': 'list the files', 'cot': '', 'reasoning_effort': ''},
        {'role': 'model', 'content': '', 'cot': 'the user wants a listing',
         'reasoning_effort': 'low',
         'tool_call_json': json.dumps([{'name': 'bash', 'param': {'cmd': 'ls'}}])},
        {'role': 'tool_response', 'content': 'a.txt b.txt', 'cot': '', 'reasoning_effort': ''},
        {'role': 'model', 'content': 'two files', 'cot': 'a and b, two entries',
         'reasoning_effort': 'high'},
        {'role': 'model', 'content': 'done', 'cot': '', 'reasoning_effort': 'none'},
    ],
}

def test_session_row_becomes_one_conversation():
    dataset = _dataset([SESSION_ROW, SESSION_ROW])
    assert len(dataset.groups) == 2, f'2 session rows should give exactly 2 groups, got {len(dataset.groups)}'
    for group in dataset.groups:
        assert group.get('kind') == 'chat'
        assert group['system'] == 'you are a careful assistant'
        assert [m['role'] for m in dataset.build_messages(group)] == [
            'system', 'user', 'model', 'tool', 'model', 'model'
        ]

def test_session_row_per_turn_effort():
    dataset = _dataset([SESSION_ROW] * 2)
    group = dataset.groups[0]
    models = _model_messages(dataset, group)
    assert [m.get('effort') for m in models] == ['low', 'high', None], models
    assert models[1]['reasoning_content'] == 'a and b, two entries'

    ids = _ids(dataset, group)
    assert _tok('effort_low') in ids, 'effort_low was not rendered'
    assert _tok('effort_high') in ids, 'effort_high was not rendered'
    assert _tok('effort_max') not in ids
    assert ids.index(_tok('effort_low')) < ids.index(_tok('effort_high')), 'effort token order is wrong'
    assert ids.count(_tok('effort_low')) == 1, ids.count(_tok('effort_low'))
    assert ids.count(_tok('effort_high')) == 1, ids.count(_tok('effort_high'))

def test_none_turn_renders_empty_thought_section():
    dataset = _dataset([SESSION_ROW])
    ids = _ids(dataset, dataset.groups[0])
    assert _tok('thought_start') in ids and _tok('thought_end') in ids
    assert ids.count(_tok('thought_start')) == 3, 'every model turn should carry one thought block'

def test_tool_declarations_reach_the_system_block():
    dataset = _dataset([SESSION_ROW, SESSION_ROW])
    group = dataset.groups[0]
    ids = _ids(dataset, group)
    text = make_tokenizer().decode(ids)
    assert '# You have access to the following functions' in text, text[:400]
    assert 'run a shell command' in text, 'tool declarations were not rendered into the system block'
    assert _tok('tool_call_start') in ids and _tok('tool_call_end') in ids, 'tool call was not rendered'
    assert _tok('tool_name_divider') in ids
    assert text.count('run a shell command') == 1, text.count('run a shell command')
    assert len(group['messages']) == 5

def test_tool_calls_rendered_once_per_turn():
    dataset = _dataset([SESSION_ROW])
    group = dataset.groups[0]
    model = _model_messages(dataset, group)[0]
    assert model['tool_calls'] == [
        {'type': 'function', 'function': {'name': 'bash', 'arguments': {'cmd': 'ls'}}}
    ], model
    ids = _ids(dataset, group)
    text = make_tokenizer().decode(ids)
    assert text.count('<|tool_call_start|>bash') == 1, text.count('<|tool_call_start|>bash')
    assert text.count('<|tool_call_start|>tool_name') == 1, 'the usage text should appear once'

def test_row_level_effort_is_the_fallback():
    row = {'system': 's', 'effort': 'max',
           'conversations': [{'role': 'user', 'content': 'hi'},
                             {'role': 'model', 'content': 'yo', 'reasoning_content': 'thinking'}]}
    dataset = _dataset([row] * 2)
    model = _model_messages(dataset, dataset.groups[0])[0]
    assert model['effort'] == 'max', model

def test_message_effort_wins_over_the_row():
    row = {'system': 's', 'effort': 'max',
           'conversations': [{'role': 'user', 'content': 'hi'},
                             {'role': 'model', 'content': 'yo', 'reasoning_content': 'thinking',
                              'effort': 'low'}]}
    dataset = _dataset([row] * 2)
    assert _model_messages(dataset, dataset.groups[0])[0]['effort'] == 'low'

def test_dataset_default_effort():
    dataset = _dataset([{'input': 'hi', 'content': 'yo', 'cot': 'thinking'}] * 4,
                       effort='max', two_turn_prob=0.0, three_turn_prob=0.0)
    assert all(m['effort'] == 'max'
               for group in dataset.groups for m in _model_messages(dataset, group))

def test_pair_row_keeps_its_own_effort_and_tools():
    rows = [{'input': 'run it', 'content': 'ok', 'cot': 'let me run it', 'reasoning_effort': 'low',
             'tool': [TOOL_DECL],
             'tool_call': [{'name': 'bash', 'param': {'cmd': 'ls'}}]}] * 2
    dataset = _dataset(rows, two_turn_prob=0.0, three_turn_prob=0.0)
    group = dataset.groups[0]
    model = _model_messages(dataset, group)[0]
    assert model['effort'] == 'low'
    assert model['tool_calls'][0]['function']['name'] == 'bash'
    assert group['tools'] == [TOOL_DECL]
    text = make_tokenizer().decode(_ids(dataset, group))
    assert 'run a shell command' in text, 'the tool declaration of the pair row was not rendered'

def test_pair_row_without_cot_gets_an_empty_thought_section():
    dataset = _dataset([{'input': 'hi', 'content': 'yo'}] * 4,
                       two_turn_prob=0.0, three_turn_prob=0.0)
    group = dataset.groups[0]
    ids = _ids(dataset, group)
    assert _tok('thought_start') in ids and _tok('thought_end') in ids
    for level in ('low', 'high', 'max'):
        assert _tok(f'effort_{level}') not in ids
    assert 'effort' not in _model_messages(dataset, group)[0]

def test_none_next_to_reasoning_content_is_refused():
    row = {'system': 's',
           'conversations': [{'role': 'user', 'content': 'hi'},
                             {'role': 'model', 'content': 'yo', 'cot': 'thinking',
                              'reasoning_effort': 'none'}]}
    _raises([row], "'none' contradicts", 'none + cot')

def test_unknown_effort_is_refused():
    row = {'system': 's',
           'conversations': [{'role': 'user', 'content': 'hi'},
                             {'role': 'model', 'content': 'yo', 'reasoning_content': 'x',
                              'reasoning_effort': 'extreme'}]}
    _raises([row], 'extreme', '未知 effort')

def test_bad_dataset_default_effort_is_refused():
    _raises([{'input': 'hi', 'content': 'yo'}], 'extreme', '数据集默认 effort', effort='extreme')

def test_row_without_conversation_or_pair_is_refused():
    _raises([{'domain': 'math', 'language': 'en'}], 'neither a conversation',
            '既不是对话也不是配对的行')

def test_unknown_session_role_is_refused():
    row = {'system': 's', 'session': [{'role': 'narrator', 'content': 'hi'},
                                      {'role': 'model', 'content': 'yo'}]}
    _raises([row], 'unsupported message role', '未知 role')

def test_render_group_produces_a_batchable_sample():
    dataset = _dataset([SESSION_ROW] * 3)
    for group in dataset.groups:
        tensors = dataset.render_group(group)
        assert tensors['input_ids'].dim() == 1
        assert tensors['input_ids'].size(0) == dataset.pad_length
    batch = dataset._build_batch(0)
    assert batch['input_ids'].size(0) == dataset.batch_size
