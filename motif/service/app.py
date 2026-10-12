import json
import sys
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from motif.service import schemas
from motif.service.backends import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    SamplingParams,
)


def torch_imported() -> bool:
    '''
    Whether `torch` is really loaded.

    `no_torch.block_torch` leaves `sys.modules['torch'] = None`, so a plain
    membership test would report the blocked state as an import.
    '''
    return sys.modules.get('torch') is not None


def _stop_strings(stop) -> List[str]:
    '''`stop` is a string, a list of strings, or nothing.'''
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    return [text for text in stop if text]


class StopFilter:
    '''
    Hold back the tail that could still grow into a stop string.

    A stop string is not part of the reply, so the last `len(stop) - 1`
    characters have to wait until the next delta shows whether they start one.
    '''

    def __init__(self, stops: List[str]):
        self.stops = list(stops)
        self.hold = max((len(stop) for stop in self.stops), default=0)
        self.pending = ''
        self.stopped = False

    def _cut(self, text: str) -> str:
        for stop in self.stops:
            index = text.find(stop)
            if index >= 0:
                self.stopped = True
                return text[:index]
        return text

    def feed(self, delta: str) -> str:
        if self.stopped or not delta:
            return ''
        self.pending += delta
        keep = max(self.hold - 1, 0)
        if len(self.pending) <= keep:
            return ''
        emit = self.pending[:len(self.pending) - keep]
        self.pending = self.pending[len(self.pending) - keep:] if keep else ''
        return self._cut(emit)

    def flush(self) -> str:
        if self.stopped:
            return ''
        self.stopped = True
        return self._cut(self.pending)


class ReplySplitter:
    '''
    Turn decoded deltas into `(reasoning, content)` deltas.

    The accumulated text goes through the runtime's `split_thought` on every
    step, so the markers are handled exactly like they are in `.research/runtime`.
    '''

    def __init__(self, template, thinking: bool):
        self.template = template
        self.inside = bool(thinking)
        self.text = ''
        self.reasoning = ''
        self.content = ''

    def feed(self, delta: str):
        if not delta:
            return '', ''
        self.text += delta
        split = self.template.split_thought(self.text, inside_thought=self.inside)
        reasoning = ''
        if split.reasoning.startswith(self.reasoning):
            reasoning = split.reasoning[len(self.reasoning):]
        content = ''
        if split.content.startswith(self.content):
            content = split.content[len(self.content):]
        self.reasoning, self.content = split.reasoning, split.content
        return reasoning, content


def _chunk(payload: Dict[str, Any]) -> str:
    return f'data: {json.dumps(payload, ensure_ascii=False)}\n\n'


def _completion_id(prefix: str) -> str:
    return f'{prefix}-{uuid.uuid4().hex[:24]}'


def _final_text(result: Dict[str, Any], stops: List[str]) -> str:
    text = result['text']
    for stop in stops:
        index = text.find(stop)
        if index >= 0:
            return text[:index]
    return text


def create_app(backend, api_key: Optional[str] = None, system: Optional[str] = None,
               default_max_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> FastAPI:
    '''Build the FastAPI app that serves `backend`.'''
    app = FastAPI(
        title='codon-motif',
        version='0.1.0',
        description='OpenAI-compatible text generation over MotifChord weights.',
    )
    app.state.backend = backend
    app.state.started = time.time()
    system_prompt = system if system is not None else backend.system

    def authorize(authorization: Optional[str] = Header(default=None)) -> None:
        if not api_key:
            return
        if authorization != f'Bearer {api_key}':
            raise HTTPException(status_code=401, detail='invalid API key',
                                headers={'WWW-Authenticate': 'Bearer'})

    def sampling_params(request, prompt_tokens: int) -> SamplingParams:
        limit = max(int(backend.context_length), 1)
        if prompt_tokens >= limit:
            raise HTTPException(
                status_code=400,
                detail=f'the prompt is {prompt_tokens} tokens, but the context holds {limit}')
        requested = (getattr(request, 'max_completion_tokens', None)
                     or getattr(request, 'max_tokens', None) or default_max_tokens)
        extra: List[int] = []
        for stop in _stop_strings(request.stop):
            ids = backend.stop_string_ids(stop)
            if not ids:
                continue
            if len(ids) != 1:
                raise HTTPException(
                    status_code=400,
                    detail=f'the stop string {stop!r} encodes to {len(ids)} tokens; '
                           f'only a single-token stop string is supported')
            extra.append(ids[0])
        return SamplingParams(
            max_new_tokens=max(1, min(int(requested), limit - prompt_tokens)),
            temperature=(DEFAULT_TEMPERATURE if request.temperature is None
                         else float(request.temperature)),
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=(DEFAULT_REPETITION_PENALTY
                                if request.repetition_penalty is None
                                else float(request.repetition_penalty)),
            seed=request.seed,
            stop_ids=tuple(extra),
        )

    def usage(prompt_tokens: int, completion_tokens: int) -> Dict[str, int]:
        return {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
        }

    def chat_prompt(request) -> tuple:
        '''(prompt ids, thinking) for a chat request, or a 400.'''
        if request.n and request.n != 1:
            raise HTTPException(status_code=400, detail='n > 1 is not supported')
        thinking = True if request.enable_thinking is None else bool(request.enable_thinking)
        messages = [schemas.as_dict(message) for message in request.messages]
        if system_prompt is not None and not any(
                message.get('role') in ('system', 'developer') for message in messages):
            messages = [{'role': 'system', 'content': system_prompt}] + messages
        try:
            prompt_ids = backend.render(messages, thinking=thinking,
                                        effort=request.reasoning_effort,
                                        tools=request.tools)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return prompt_ids, thinking

    @app.get('/health')
    def health() -> Dict[str, Any]:
        return {
            'status': 'ok',
            'torch_imported': torch_imported(),
            'motif_imported': sys.modules.get('motif') is not None,
            'api_key': bool(api_key),
            'system_prompt': system_prompt is not None,
            **backend.info(),
        }

    @app.get('/v1/models')
    def models() -> Dict[str, Any]:
        return {
            'object': 'list',
            'data': [{
                'id': backend.model_id,
                'object': 'model',
                'created': int(app.state.started),
                'owned_by': 'codon',
            }],
        }

    @app.post('/v1/chat/completions')
    def chat_completions(request: schemas.ChatCompletionRequest, _=Depends(authorize)):
        prompt_ids, thinking = chat_prompt(request)
        params = sampling_params(request, len(prompt_ids))
        stops = _stop_strings(request.stop)
        model_id = request.model or backend.model_id
        created = int(time.time())

        if not request.stream:
            try:
                result = backend.generate(prompt_ids, params)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f'generation failed: {exc}')
            text = _final_text(result, stops)
            split = backend.template.split_thought(text, inside_thought=thinking)
            message: Dict[str, Any] = {'role': 'assistant', 'content': split.content}
            if split.reasoning:
                message['reasoning_content'] = split.reasoning
            if split.effort:
                message['reasoning_effort'] = split.effort
            return {
                'id': _completion_id('chatcmpl'),
                'object': 'chat.completion',
                'created': created,
                'model': model_id,
                'choices': [{
                    'index': 0,
                    'message': message,
                    'finish_reason': 'stop' if result['reason'] == 'stop' else 'length',
                }],
                'usage': usage(result['prompt_tokens'], result['completion_tokens']),
            }

        def events() -> Iterator[str]:
            identifier = _completion_id('chatcmpl')
            include_usage = bool((request.stream_options or {}).get('include_usage'))
            filter_ = StopFilter(stops)
            splitter = ReplySplitter(backend.template, thinking)
            produced = 0
            yield _chunk({
                'id': identifier, 'object': 'chat.completion.chunk', 'created': created,
                'model': model_id,
                'choices': [{'index': 0, 'delta': {'role': 'assistant'},
                             'finish_reason': None}],
            })
            try:
                for delta in backend.stream(prompt_ids, params):
                    produced += 1
                    reasoning, content = splitter.feed(filter_.feed(delta))
                    for field, value in (('reasoning_content', reasoning),
                                         ('content', content)):
                        if value:
                            yield _chunk({
                                'id': identifier, 'object': 'chat.completion.chunk',
                                'created': created, 'model': model_id,
                                'choices': [{'index': 0, 'delta': {field: value},
                                             'finish_reason': None}],
                            })
            except Exception as exc:
                yield _chunk({'id': identifier, 'object': 'chat.completion.chunk',
                              'created': created, 'model': model_id,
                              'choices': [{'index': 0, 'delta': {},
                                           'finish_reason': 'error'}],
                              'error': str(exc)})
                yield 'data: [DONE]\n\n'
                return
            reasoning, content = splitter.feed(filter_.flush())
            for field, value in (('reasoning_content', reasoning), ('content', content)):
                if value:
                    yield _chunk({
                        'id': identifier, 'object': 'chat.completion.chunk',
                        'created': created, 'model': model_id,
                        'choices': [{'index': 0, 'delta': {field: value},
                                     'finish_reason': None}],
                    })
            finished = 'stop' if (filter_.stopped or produced < params.max_new_tokens) else 'length'
            yield _chunk({
                'id': identifier, 'object': 'chat.completion.chunk', 'created': created,
                'model': model_id,
                'choices': [{'index': 0, 'delta': {}, 'finish_reason': finished}],
            })
            if include_usage:
                text = splitter.reasoning + splitter.content
                yield _chunk({
                    'id': identifier, 'object': 'chat.completion.chunk',
                    'created': created, 'model': model_id, 'choices': [],
                    'usage': usage(len(prompt_ids), len(backend.encode(text))),
                })
            yield 'data: [DONE]\n\n'

        return StreamingResponse(events(), media_type='text/event-stream')

    @app.post('/v1/completions')
    def completions(request: schemas.CompletionRequest, _=Depends(authorize)):
        prompt = request.prompt
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise HTTPException(status_code=400,
                                    detail='a list prompt must hold exactly one string')
            prompt = prompt[0]
        prompt_ids = backend.encode(prompt)
        params = sampling_params(request, len(prompt_ids))
        stops = _stop_strings(request.stop)
        model_id = request.model or backend.model_id
        created = int(time.time())

        if not request.stream:
            try:
                result = backend.generate(prompt_ids, params)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f'generation failed: {exc}')
            return {
                'id': _completion_id('cmpl'),
                'object': 'text_completion',
                'created': created,
                'model': model_id,
                'choices': [{
                    'index': 0,
                    'text': _final_text(result, stops),
                    'logprobs': None,
                    'finish_reason': 'stop' if result['reason'] == 'stop' else 'length',
                }],
                'usage': usage(result['prompt_tokens'], result['completion_tokens']),
            }

        def events() -> Iterator[str]:
            identifier = _completion_id('cmpl')
            filter_ = StopFilter(stops)
            produced = 0
            yield _chunk({'id': identifier, 'object': 'text_completion', 'created': created,
                          'model': model_id,
                          'choices': [{'index': 0, 'text': '', 'finish_reason': None}]})
            try:
                for delta in backend.stream(prompt_ids, params):
                    produced += 1
                    text = filter_.feed(delta)
                    if text:
                        yield _chunk({'id': identifier, 'object': 'text_completion',
                                      'created': created, 'model': model_id,
                                      'choices': [{'index': 0, 'text': text,
                                                   'finish_reason': None}]})
            except Exception as exc:
                yield _chunk({'id': identifier, 'object': 'text_completion',
                              'created': created, 'model': model_id,
                              'choices': [{'index': 0, 'text': '',
                                           'finish_reason': 'error'}],
                              'error': str(exc)})
                yield 'data: [DONE]\n\n'
                return
            text = filter_.flush()
            if text:
                yield _chunk({'id': identifier, 'object': 'text_completion',
                              'created': created, 'model': model_id,
                              'choices': [{'index': 0, 'text': text, 'finish_reason': None}]})
            finished = 'stop' if (filter_.stopped or produced < params.max_new_tokens) else 'length'
            yield _chunk({'id': identifier, 'object': 'text_completion', 'created': created,
                          'model': model_id,
                          'choices': [{'index': 0, 'text': '', 'finish_reason': finished}]})
            yield 'data: [DONE]\n\n'

        return StreamingResponse(events(), media_type='text/event-stream')

    return app
