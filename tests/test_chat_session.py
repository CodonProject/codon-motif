import torch
import torch.nn as nn

from motif.chat import (
    ChatSession,
    generation_prompt_ids,
    render_prompt,
    split_thought,
    stop_token_ids,
    strip_dialog_tokens,
)
from motif.model.config import MotifChordConfig
from motif.model.motif import MotifChordModel
from motif.tokenizer.tokenizer import create_tokenizer_trainer
from motif.utils.session import TOKEN_NAMES, Session, resolve_effort, resolve_token_id

from support import CORPUS, make_tokenizer

_THOUGHT_START = TOKEN_NAMES['thought_start']
_THOUGHT_END = TOKEN_NAMES['thought_end']

_ANSWER = 'answer'
_REASONING = 'think'

def token_id(name):
    return resolve_token_id(make_tokenizer(), name)

def assert_raises(exception, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except exception as error:
        return error
    raise AssertionError(f'{exception.__name__} was not raised')

def test_render_prompt_uses_the_chat_template():
    tokenizer = make_tokenizer()

    ids = render_prompt(tokenizer, [{'role': 'user', 'content': 'hello'}])
    prompt = generation_prompt_ids(tokenizer)

    assert ids[:2] == [resolve_token_id(tokenizer, 'im_start'), resolve_token_id(tokenizer, 'user')]
    assert ids[-len(prompt):] == prompt

def test_render_prompt_rejects_an_empty_conversation():
    error = assert_raises(ValueError, render_prompt, make_tokenizer(), [])

    assert 'empty' in str(error)

def test_generation_prompt_matches_the_template_tail():
    tokenizer = make_tokenizer()

    for enable_thinking in (True, False):
        ids = render_prompt(
            tokenizer,
            [{'role': 'user', 'content': 'hello'}],
            enable_thinking=enable_thinking,
        )
        prompt = generation_prompt_ids(tokenizer, enable_thinking=enable_thinking)
        assert ids[-len(prompt):] == prompt

    assert generation_prompt_ids(tokenizer, enable_thinking=False)[-2:] == [
        token_id('thought_start'),
        token_id('thought_end'),
    ]
    assert generation_prompt_ids(tokenizer, effort='max')[-1] == token_id('effort_max')

def test_split_thought_reads_a_generated_turn():
    split = split_thought(f'{_REASONING}{_THOUGHT_END}{_ANSWER}<|im_end|>')

    assert split.reasoning == _REASONING
    assert split.content == f'{_ANSWER}<|im_end|>'
    assert split.effort is None
    assert split.complete is True

def test_split_thought_reads_a_full_render():
    split = split_thought(
        f'{_THOUGHT_START}{TOKEN_NAMES["effort_high"]}{_REASONING}{_THOUGHT_END}{_ANSWER}'
    )

    assert split.reasoning == _REASONING
    assert split.content == _ANSWER
    assert split.effort == 'high'
    assert split.complete is True

def test_split_thought_without_a_section():
    split = split_thought('just an answer')

    assert split.reasoning == ''
    assert split.content == 'just an answer'
    assert split.effort is None
    assert split.complete is True

def test_split_thought_tolerates_an_unterminated_section():
    split = split_thought(f'{_THOUGHT_START}{_REASONING}')

    assert split.reasoning == _REASONING
    assert split.content == ''
    assert split.complete is False

def test_split_thought_knows_when_the_section_is_still_open():
    split = split_thought(_REASONING, inside_thought=True)

    assert split.reasoning == _REASONING
    assert split.content == ''
    assert split.complete is False

    split = split_thought(_REASONING)

    assert split.reasoning == ''
    assert split.content == _REASONING
    assert split.complete is True

def test_strip_dialog_tokens_keeps_tool_markers():
    text = (
        f'<|im_start|>{TOKEN_NAMES["model"]}{_THOUGHT_START}{TOKEN_NAMES["effort_high"]}'
        f'{_REASONING}{_THOUGHT_END}{_ANSWER}<|im_end|>'
        f'{TOKEN_NAMES["tool_call_start"]}{{}}{TOKEN_NAMES["tool_call_end"]}'
    )

    stripped = strip_dialog_tokens(text)

    for marker in ('<|im_start|>', '<|im_end|>', TOKEN_NAMES['model'], _THOUGHT_START,
                   _THOUGHT_END, TOKEN_NAMES['effort_high']):
        assert marker not in stripped
    for kept in (_ANSWER, TOKEN_NAMES['tool_call_start'], TOKEN_NAMES['tool_call_end']):
        assert kept in stripped

def test_stop_token_ids_cover_the_end_of_a_turn():
    tokenizer = make_tokenizer()

    stops = stop_token_ids(tokenizer)

    assert token_id('im_end') in stops
    assert token_id('tool_call_end') in stops
    assert 4321 in stop_token_ids(tokenizer, extra=[4321])
    assert stop_token_ids(tokenizer, extra=[token_id('im_end')]).count(token_id('im_end')) == 1

class _ScriptedCache:

    def __init__(self, prompt_len):
        self.seq_len = 0

class _ScriptedOutput:
    def __init__(self, token, cache):
        self.token = int(token)
        self.cache = cache

    def next_token(self, sampler=None, input_ids=None):
        return torch.tensor([[self.token]], dtype=torch.long)

class ScriptedModel(nn.Module):

    def __init__(self, script):
        super().__init__()
        self.script = [int(token) for token in script]
        self.anchor = nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, input_ids, cache=None, labels=None, mask=None):
        if cache is None:
            cache = _ScriptedCache(prompt_len=int(input_ids.shape[1]))
        cache.seq_len += int(input_ids.shape[1])
        token = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return _ScriptedOutput(token, cache)

def turn_ids(tokenizer, user_text='hello'):
    conversation = [{'role': 'user', 'content': user_text}]
    prompt = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, effort='high', enable_thinking=True
    )
    full = tokenizer.apply_chat_template(
        conversation + [{'role': 'model', 'content': _ANSWER, 'thought': _REASONING, 'effort': 'high'}],
        add_generation_prompt=False,
        effort='high',
        enable_thinking=True,
    )
    assert full[:len(prompt)] == prompt
    return full[len(prompt):]

def make_session(turns=2, **kwargs):
    tokenizer = make_tokenizer()
    model = ScriptedModel(turn_ids(tokenizer) * turns)
    return ChatSession(model, tokenizer, **kwargs), model

def test_send_appends_one_model_message():
    chat, _ = make_session()
    generated = turn_ids(make_tokenizer())
    chat.user('hello')
    ids = chat.render()

    reply = chat.send()

    assert len(chat.messages) == 2
    assert [message.role for message in chat.messages] == ['user', 'model']
    assert chat.messages[-1].content == reply.text == _ANSWER
    assert reply.reasoning == _REASONING
    assert reply.raw == f'{_REASONING}{_THOUGHT_END}{_ANSWER}<|im_end|>'
    assert reply.new_tokens == len(generated)
    assert reply.steps == reply.new_tokens
    assert reply.stopped is True
    assert reply.prompt_tokens == len(ids)
    assert chat.history_ids == ids + generated[:-1]

def test_second_turn_reuses_the_cache():
    chat, _ = make_session()

    chat.send('hello')
    chat.user('again')
    ids = chat.render()
    prefix = len(chat.history_ids)
    reply = chat.send()

    assert reply.reused_cache is True
    assert reply.prompt_tokens == len(ids) - prefix < len(ids)
    assert len(chat.messages) == 4
    assert chat.messages[-1].content == _ANSWER
    assert len(chat.history_ids) == len(ids) + reply.new_tokens - 1

def test_cache_reuse_does_not_change_the_reply():
    reused, _ = make_session(reuse_cache=True)
    fresh, _ = make_session(reuse_cache=False)

    for chat in (reused, fresh):
        chat.send('hello')
        chat.send('again')

    assert reused.messages[-1].content == fresh.messages[-1].content == _ANSWER
    assert reused.last_reply.reused_cache is True
    assert fresh.last_reply.reused_cache is False
    assert reused.last_reply.prompt_tokens < fresh.last_reply.prompt_tokens

def test_a_stale_cache_is_not_reused():
    chat, _ = make_session()
    chat.send('hello')

    chat._prefix_ids = chat._prefix_ids + [token_id('sep')]
    chat.user('again')
    ids = chat.render()
    reply = chat.send()

    assert reply.reused_cache is False
    assert reply.text == _ANSWER
    assert reply.prompt_tokens == len(ids)

def test_stream_yields_the_answer():
    chat, _ = make_session()

    chunks = list(chat.stream('hello'))

    assert chunks and all(chunks)
    assert ''.join(chunks) == chat.last_reply.streamed == _ANSWER
    assert chat.messages[-1].content == _ANSWER

def test_reset_keeps_the_system_message():
    chat, _ = make_session(system='be nice')
    chat.send('hello')

    chat.reset()

    assert [message.role for message in chat.messages] == ['system']
    assert chat.cache is None
    assert chat.history_ids == []

    chat.reset(keep_system=False)
    assert chat.messages == ()

def test_unknown_role_and_override_are_rejected():
    chat, _ = make_session()

    error = assert_raises(ValueError, chat.add, 'wizard', 'hi')
    assert 'wizard' in str(error)
    assert chat.messages == ()

    error = assert_raises(TypeError, chat.send, 'hello', temperature=0.5)
    assert 'temperature' in str(error)

def test_close_restores_the_model_mode():
    tokenizer = make_tokenizer()
    model = ScriptedModel(turn_ids(tokenizer) * 2)

    model.train()
    chat = ChatSession(model, tokenizer)
    assert model.training is False
    chat.close()
    assert model.training is True

    model.eval()
    with ChatSession(model, tokenizer) as chat:
        assert model.training is False
    assert model.training is False

def test_a_real_model_runs_a_turn():
    tokenizer = make_tokenizer()
    torch.manual_seed(0)
    model = MotifChordModel(
        MotifChordConfig(
            vocab_size=tokenizer.vocab_size,
            model_dim=32,
            num_heads=4,
            num_kv_heads=2,
            head_k_dim=8,
            block=('gdn', 'chord'),
            block_n=2,
            tied_weight=True,
        )
    )
    chat = ChatSession(model, tokenizer, max_new_tokens=4)

    reply = chat.send('hello')

    assert model.training is False
    assert reply.steps == reply.new_tokens > 0
    assert chat.messages[-1].content == reply.text
    assert isinstance(reply.streamed, str)

    chat.send('hello again')

    assert len(chat.messages) == 4
    assert chat.last_reply.new_tokens > 0
    chat.close()
    assert model.training is True

def test_a_generated_turn_keeps_its_own_ids():
    tokenizer = make_tokenizer()
    conversation = [{'role': 'user', 'content': 'hello'}]
    prompt = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, effort='high', enable_thinking=False
    )
    generated = tokenizer.encode(_ANSWER, apply_safe_rule=False) + [token_id('im_end')]
    model = ScriptedModel(generated * 2)
    chat = ChatSession(model, tokenizer, enable_thinking=False)

    reply = chat.send('hello')

    rerendered = tokenizer.apply_chat_template(
        conversation + [{'role': 'model', 'content': reply.text}],
        add_generation_prompt=False,
        effort='high',
        enable_thinking=False,
    )
    assert rerendered[:len(prompt)] == prompt
    assert chat.render(add_generation_prompt=False) == prompt + generated

    chat.user('again')
    ids = chat.render()
    second = chat.send()

    assert second.reused_cache is True
    assert second.prompt_tokens < len(ids)
    assert second.text == _ANSWER

def test_a_turn_can_continue_without_a_new_message():
    chat, _ = make_session(turns=2)
    chat.send('hello')
    cached = chat.history_ids
    prefix = len(cached)

    ids = chat.render()
    reply = chat.send()

    assert ids[:prefix] == cached
    assert reply.reused_cache is True
    assert reply.prompt_tokens == len(ids) - prefix < len(ids)
    assert reply.text == _ANSWER


# --- Session helpers and generation prompts ---
from motif.utils.session import Session, resolve_effort, resolve_token_id

from support import CORPUS, make_tokenizer

def test_resolve_helpers():
    tokenizer = make_tokenizer()

    assert resolve_token_id(tokenizer, 'im_end') == tokenizer.token_to_id('<|im_end|>')
    assert resolve_token_id(tokenizer, '<|bbox_start|>') == tokenizer.token_to_id('<|bbox_start|>')
    assert resolve_token_id(tokenizer, 'not_a_token') is None

    assert resolve_effort(None) == 'high'
    assert resolve_effort('minimal') == 'low'
    assert resolve_effort('medium') == 'high'
    assert resolve_effort('xhigh') == 'high'
    assert resolve_effort('ultra') == 'max'

    try:
        resolve_effort('bogus')
    except ValueError:
        pass
    else:
        raise AssertionError('an invalid effort level must raise')

def test_text_roles_and_content_policy():
    tokenizer = make_tokenizer()
    session = Session(tokenizer)
    session.add_message({'role': 'system', 'content': 'You are a helpful assistant.'})
    session.add_message({'role': 'user', 'content': 'hello'})
    session.add_message({'role': 'assistant', 'thought': 'let me think', 'content': 'hi there'})
    session.add_message({'role': 'tool', 'name': 'f', 'content': 'result'})
    session.add_message({'role': 'user', 'content': 'bye'})

    assert [m.role for m in session] == ['system', 'user', 'model', 'tool', 'user']

    for i in (0, 1, 3, 4):
        assert all(session[i].ignore_mask), f'message {i} must not contribute to the loss'

    msg = session[2]
    start, end = msg.find(session.token_id('thought_start')), msg.find(session.token_id('thought_end'))
    assert 0 < start < end
    assert msg.ids[0] == session.token_id('im_start') and msg.ignore_mask[0]
    assert msg.ids[1] == session.token_id('model') and msg.ignore_mask[1]
    assert not any(msg.ignore_mask[start:]), 'everything after thought_start joins the loss'
    assert msg.ids[-1] == session.token_id('im_end') and not msg.ignore_mask[-1]

def test_user_turn_uses_human_token():
    tokenizer = make_tokenizer()
    human = tokenizer.token_to_id('<|human|>')
    assert human is not None
    assert tokenizer.token_to_id('<|user|>') is None, '<|user|> is not in the vocabulary here'

    session = Session(tokenizer)
    session.add_message({'role': 'user', 'content': 'hello'})

    msg = session[0]
    assert msg.ids[0] == session.token_id('im_start')
    assert msg.ids[1] == human, 'the role token of a user turn must be <|human|>'
    assert msg.ids.count(human) == 1, 'the role token must be one special token, not several pieces'
    assert msg.ids[-1] == session.token_id('im_end')
    assert session.token_id('user') == human

    assert Session(tokenizer).add_message({'role': 'human', 'content': 'x'})[0].role == 'user'
    assert Session(tokenizer).add_message({'role': 'human', 'content': 'x'})[0].ids[1] == human

def test_generation_prompt():
    tokenizer = make_tokenizer()
    session = Session(tokenizer).add_message({'role': 'user', 'content': 'hello'})
    session.add_generation_prompt(enable_thinking=True, effort='medium')

    gen = session[-1]
    assert gen.role == '__gen_prompt__' and all(gen.ignore_mask)
    assert gen.ids == [
        session.token_id('im_start'), session.token_id('model'),
        session.token_id('thought_start'), session.token_id('effort_high'),
    ]

    session.add_message({'role': 'assistant', 'content': 'answer'})
    assert [m.role for m in session][-2:] == ['model', '__gen_prompt__']

    session.pop()
    session.add_generation_prompt(disable_thinking=True)
    assert session[-1].ids[-2:] == [
        session.token_id('thought_start'), session.token_id('thought_end'),
    ]

    try:
        session.add_generation_prompt(enable_thinking=True, disable_thinking=True)
    except ValueError:
        pass
    else:
        raise AssertionError('enable_thinking and disable_thinking cannot both be True')

def test_template_fallback_and_pad_to():
    trainer = create_tokenizer_trainer(vocab_size=512)
    trainer.train_from_iterator(iter(CORPUS))
    tokenizer = trainer.packed_tokenizer
    tokenizer.template = ''

    Session(tokenizer)
    assert tokenizer.template.strip()

    session = Session(tokenizer).add_message({'role': 'user', 'content': 'x'})
    length = len(session)
    session.pad_to(length + 4, side='left')
    assert len(session) == length + 4
    assert session[0].role == '__padding__' and all(session[0].ignore_mask)

def _run_all():
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith('test_') and callable(fn)
    ]
    for name, fn in tests:
        fn()
        print(f'  ok  {name}')
    print(f'\n{len(tests)} passed')
