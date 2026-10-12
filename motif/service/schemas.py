from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


def as_dict(model) -> Dict[str, Any]:
    '''Dump a pydantic model, on both pydantic 1 and 2.'''
    if hasattr(model, 'model_dump'):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


class ChatMessage(BaseModel):
    '''One turn of a conversation, in the OpenAI shape plus two local fields.'''

    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    thought: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    '''`POST /v1/chat/completions`.'''

    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    n: int = 1
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[str] = None
    enable_thinking: Optional[bool] = None
    tools: Optional[List[Dict[str, Any]]] = None


class CompletionRequest(BaseModel):
    '''`POST /v1/completions`.'''

    model: Optional[str] = None
    prompt: Union[str, List[str]] = ''
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    n: int = 1
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
