from motif.chat import generation_prompt_ids, render_prompt
from motif.utils.session import TOKEN_NAMES, Session

from support import make_tokenizer

_LEVELS = ('low', 'high', 'max')
_EFFORT_TOKENS = tuple(TOKEN_NAMES[f'effort_{level}'] for level in _LEVELS)

def render(messages, **kwargs) -> str:
    tokenizer = make_tokenizer()
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
    return tokenizer.fast_tokenizer.decode(ids)

def assert_no_effort(text: str) -> None:
    for marker in _EFFORT_TOKENS:
        assert marker not in text, text

def test_defaults_keep_the_full_section() -> None:
    text = render([{'role': 'model', 'content': 'answer', 'reasoning_content': 'reasoning'}])
    assert f'{TOKEN_NAMES["thought_start"]}{TOKEN_NAMES["effort_high"]}' in text, text
    assert 'reasoning' in text, text
    assert TOKEN_NAMES['thought_end'] in text, text

    explicit = render(
        [{'role': 'model', 'content': 'answer', 'reasoning_content': 'reasoning'}],
        enable_think=True, enable_effort=True,
    )
    assert explicit == text, explicit

def test_enable_think_off_drops_the_whole_section() -> None:
    text = render(
        [{'role': 'model', 'content': 'answer', 'reasoning_content': 'reasoning'}],
        enable_think=False,
    )
    assert 'reasoning' not in text, text
    assert TOKEN_NAMES['thought_start'] not in text, text
    assert TOKEN_NAMES['thought_end'] not in text, text

    tokenizer = make_tokenizer()
    ids = render_prompt(
        tokenizer,
        [{'role': 'model', 'content': 'answer', 'reasoning_content': 'reasoning'}],
        add_generation_prompt=False,
        enable_think=False,
    )
    assert ids == [
        tokenizer.token_to_id(TOKEN_NAMES['im_start']),
        tokenizer.token_to_id(TOKEN_NAMES['model']),
    ] + tokenizer.encode('answer') + [
        tokenizer.token_to_id(TOKEN_NAMES['im_end']),
    ], ids

def test_enable_think_off_drops_the_empty_section_too() -> None:
    text = render([{'role': 'model', 'content': 'answer'}], enable_think=False)
    assert TOKEN_NAMES['thought_start'] not in text, text
    assert TOKEN_NAMES['thought_end'] not in text, text
    assert 'answer' in text, text

def test_enable_think_off_applies_to_every_model_turn() -> None:
    text = render([
        {'role': 'model', 'content': 'first', 'reasoning_content': 'one'},
        {'role': 'user', 'content': 'again'},
        {'role': 'model', 'content': 'second', 'reasoning_content': 'two'},
    ], enable_think=False)
    assert TOKEN_NAMES['thought_start'] not in text, text
    assert TOKEN_NAMES['thought_end'] not in text, text
    assert 'one' not in text and 'two' not in text, text
    assert 'first' in text and 'second' in text, text

def test_enable_effort_off_keeps_the_section_without_the_marker() -> None:
    text = render(
        [{'role': 'model', 'content': 'answer', 'reasoning_content': 'reasoning'}],
        enable_effort=False,
    )
    assert_no_effort(text)
    assert TOKEN_NAMES['thought_start'] in text, text
    assert 'reasoning' in text, text
    assert f'reasoning{TOKEN_NAMES["thought_end"]}' in text, text
    assert 'answer' in text, text

def test_enable_effort_off_drops_every_level() -> None:
    for level in ('minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'):
        text = render(
            [{'role': 'model', 'content': 'a', 'reasoning_content': 'r', 'effort': level}],
            enable_effort=False,
        )
        assert_no_effort(text)

def test_enable_effort_does_not_beat_enable_think() -> None:
    text = render(
        [{'role': 'model', 'content': 'answer', 'reasoning_content': 'reasoning'}],
        enable_think=False, enable_effort=True,
    )
    assert_no_effort(text)
    assert TOKEN_NAMES['thought_start'] not in text, text

def test_generation_prompt_combinations() -> None:
    tokenizer = make_tokenizer()
    ids = lambda **kw: generation_prompt_ids(tokenizer, **kw)
    start, end, model, im_start = (
        tokenizer.token_to_id(TOKEN_NAMES['thought_start']),
        tokenizer.token_to_id(TOKEN_NAMES['thought_end']),
        tokenizer.token_to_id(TOKEN_NAMES['model']),
        tokenizer.token_to_id(TOKEN_NAMES['im_start']),
    )

    assert ids() == [im_start, model, start, tokenizer.token_to_id(TOKEN_NAMES['effort_high'])]
    assert ids(enable_thinking=False) == [im_start, model, start, end]
    assert ids(enable_think=False) == [im_start, model]
    assert ids(enable_think=False, enable_thinking=False, enable_effort=False) == [im_start, model]
    assert ids(enable_effort=False) == [im_start, model, start, end]
    assert ids(effort='max')[-1] == tokenizer.token_to_id(TOKEN_NAMES['effort_max'])

def test_generation_prompt_tail_matches_the_template() -> None:
    tokenizer = make_tokenizer()
    messages = [{'role': 'user', 'content': 'hello'}]

    for enable_think in (True, False):
        for enable_effort in (True, False):
            for enable_thinking in (True, False):
                tail = render_prompt(
                    tokenizer,
                    messages,
                    effort='low',
                    enable_think=enable_think,
                    enable_effort=enable_effort,
                    enable_thinking=enable_thinking,
                )
                prompt = generation_prompt_ids(
                    tokenizer,
                    effort='low',
                    enable_think=enable_think,
                    enable_effort=enable_effort,
                    enable_thinking=enable_thinking,
                )
                assert tail[-len(prompt):] == prompt, (
                    enable_think, enable_effort, enable_thinking, tail, prompt
                )

def test_template_keeps_the_older_enable_thinking_switch() -> None:
    tokenizer = make_tokenizer()
    messages = [{'role': 'user', 'content': 'hello'}]
    full = f'{TOKEN_NAMES["thought_start"]}{TOKEN_NAMES["effort_high"]}'
    empty = f'{TOKEN_NAMES["thought_start"]}{TOKEN_NAMES["thought_end"]}'

    def tail(**kwargs):
        kwargs.setdefault('effort', 'high')
        return tokenizer.fast_tokenizer.decode(
            tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kwargs)
        )

    assert tail().endswith(full), tail()
    assert tail(enable_thinking=True) == tail(), tail(enable_thinking=True)
    assert tail(enable_think=True) == tail(), tail(enable_think=True)
    assert tail(enable_thinking=False).endswith(empty), tail(enable_thinking=False)
    assert tail(enable_thinking=False) == tail(enable_effort=False), tail(enable_thinking=False)
    assert tail(enable_think=False).endswith('<|model|>'), tail(enable_think=False)
    assert tail(enable_think=False) == tail(enable_think=False, enable_effort=False), tail(enable_think=False)
    assert tail(enable_thinking=False, enable_effort=False).endswith(empty), tail(
        enable_thinking=False, enable_effort=False
    )
    assert tail(effort=None).endswith(full), tail(effort=None)

def test_template_generation_prompt_switch() -> None:
    tokenizer = make_tokenizer()
    messages = [{'role': 'user', 'content': 'hello'}]
    text = lambda **kw: tokenizer.fast_tokenizer.decode(
        tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kw)
    )

    assert text().endswith(f'{TOKEN_NAMES["thought_start"]}{TOKEN_NAMES["effort_high"]}'), text()
    assert text(enable_think=False).endswith('<|model|>'), text(enable_think=False)
    assert_no_effort(text(enable_think=False))
    assert text(enable_effort=False).endswith(
        f'{TOKEN_NAMES["thought_start"]}{TOKEN_NAMES["thought_end"]}'
    ), text(enable_effort=False)

def test_session_generation_prompt_switch() -> None:
    tokenizer = make_tokenizer()
    gen = lambda **kw: Session(tokenizer).add_generation_prompt(**kw)[-1].ids

    im_start, model = tokenizer.token_to_id(TOKEN_NAMES['im_start']), tokenizer.token_to_id(TOKEN_NAMES['model'])
    start, end = tokenizer.token_to_id(TOKEN_NAMES['thought_start']), tokenizer.token_to_id(TOKEN_NAMES['thought_end'])

    assert gen(enable_thinking=True) == [im_start, model, start, tokenizer.token_to_id(TOKEN_NAMES['effort_high'])]
    assert gen(enable_think=False) == [im_start, model]
    assert gen(enable_thinking=True, enable_think=False) == [im_start, model]
    assert gen(enable_thinking=True, enable_effort=False) == [im_start, model, start, end]
    assert gen(disable_thinking=True, enable_think=False) == [im_start, model]
    assert gen(enable_thinking=False) == [im_start, model]
    assert gen(enable_thinking=False, disable_thinking=False) == [im_start, model]
    assert gen(disable_thinking=True) == [im_start, model, start, end]
    assert gen(enable_effort=False) == [im_start, model, start, end]
    assert gen(enable_thinking=False, enable_effort=False) == [im_start, model, start, end]

def test_session_history_switch_matches_its_generation_prompt() -> None:
    tokenizer = make_tokenizer()

    session = Session(tokenizer)
    session.add_message({'role': 'model', 'content': 'answer', 'reasoning_content': 'reasoning'})
    assert session[0].ids[:4] == [
        tokenizer.token_to_id(TOKEN_NAMES['im_start']),
        tokenizer.token_to_id(TOKEN_NAMES['model']),
        tokenizer.token_to_id(TOKEN_NAMES['thought_start']),
        tokenizer.token_to_id(TOKEN_NAMES['effort_high']),
    ], session[0].ids
    session.add_generation_prompt(enable_thinking=True)
    assert session[-1].role == '__gen_prompt__'
    assert session[-1].ids[-2:] == [
        tokenizer.token_to_id(TOKEN_NAMES['thought_start']),
        tokenizer.token_to_id(TOKEN_NAMES['effort_high']),
    ]

    bare = Session(tokenizer)
    bare.add_generation_prompt(enable_thinking=True, enable_think=False)
    assert bare[-1].ids[-2:] == [
        tokenizer.token_to_id(TOKEN_NAMES['im_start']),
        tokenizer.token_to_id(TOKEN_NAMES['model']),
    ]

def test_chat_session_carries_the_switches() -> None:
    from motif.chat import ChatSession

    class _Model:
        def parameters(self):
            return iter(())

        def eval(self):
            return self

        def train(self):
            return self

    tokenizer = make_tokenizer()
    model = _Model()

    default = ChatSession(model, tokenizer)
    assert default.template_kwargs['enable_think'] is True
    assert default.template_kwargs['enable_effort'] is True
    assert default.inside_thought is True

    no_think = ChatSession(model, tokenizer, enable_think=False)
    assert no_think.template_kwargs['enable_think'] is False
    assert no_think.inside_thought is False
    no_think.add('user', 'hello')
    assert no_think.render()[-2:] == [
        tokenizer.token_to_id(TOKEN_NAMES['im_start']),
        tokenizer.token_to_id(TOKEN_NAMES['model']),
    ]

    no_effort = ChatSession(model, tokenizer, enable_effort=False)
    assert no_effort.template_kwargs['enable_effort'] is False
    assert no_effort.inside_thought is False
    no_effort.add('user', 'hello')
    assert no_effort.render()[-2:] == [
        tokenizer.token_to_id(TOKEN_NAMES['thought_start']),
        tokenizer.token_to_id(TOKEN_NAMES['thought_end']),
    ]


# --- reasoning_content fallback ---
from support import make_tokenizer

def render_via_session(message: dict) -> str:
    tokenizer = make_tokenizer()
    session = Session(tokenizer)
    session.add_message(dict(message))
    ids = [token for one in session.messages for token in one.ids]
    return tokenizer.fast_tokenizer.decode(ids)

def test_reasoning_content_only_reaches_the_thought_section() -> None:
    text = render([
        {'role': 'user', 'content': 'hi'},
        {'role': 'model', 'content': 'the answer', 'reasoning_content': 'the reasoning'},
    ])
    assert '<|thought_start|><|effort_high|>' in text, text
    assert 'the reasoning' in text, text
    assert text.index('the reasoning') < text.index('the answer'), text

def test_thought_wins_over_reasoning_content() -> None:
    text = render([
        {'role': 'model', 'content': 'a', 'thought': 'from thought', 'reasoning_content': 'from reasoning'},
    ])
    assert 'from thought' in text, text
    assert 'from reasoning' not in text, text

def test_explicit_none_thought_falls_back() -> None:
    text = render([{'role': 'model', 'content': 'a', 'thought': None, 'reasoning_content': 'fallback'}])
    assert 'fallback' in text, text

def test_empty_thought_falls_back() -> None:
    text = render([{'role': 'model', 'content': 'a', 'thought': '', 'reasoning_content': 'fallback'}])
    assert 'fallback' in text, text

def test_no_reasoning_renders_the_empty_section() -> None:
    text = render([{'role': 'model', 'content': 'a'}])
    assert '<|thought_start|><|thought_end|>' in text, text
    for level in ('low', 'high', 'max'):
        assert f'<|effort_{level}|>' not in text, text

def test_effort_kwarg_lands_in_the_thought_section() -> None:
    text = render([{'role': 'model', 'content': 'a', 'reasoning_content': 'x'}], effort='max')
    assert '<|effort_max|>' in text, text

def test_per_message_effort_beats_the_kwarg() -> None:
    text = render([{'role': 'model', 'content': 'a', 'reasoning_content': 'x', 'effort': 'low'}], effort='max')
    assert '<|effort_low|>' in text, text

def test_tool_calls_next_to_reasoning_content() -> None:
    text = render([{
        'role': 'model',
        'content': '',
        'reasoning_content': 'let me look',
        'tool_calls': [{'name': 'bash', 'arguments': {'cmd': 'ls'}}],
    }])
    assert 'let me look' in text, text
    assert '<|tool_call_start|>bash<|tool_name_divider|>' in text, text

def test_session_path_is_unchanged() -> None:
    text = render_via_session({'role': 'model', 'content': 'a', 'reasoning_content': 'via session'})
    assert 'via session' in text, text
    assert '<|thought_start|>' in text, text
