# OpenAI-compatible service

A FastAPI server around the exported ONNX bundle. It is OpenAI-compatible
enough for a chat UI or the `openai` client: `/v1/models`,
`/v1/chat/completions` (streaming and not) and `/v1/completions`.

The default backend runs the ONNX bundle through `onnxruntime` and **never
imports `torch` or `motif`**. Passing a model instance (or a checkpoint path)
switches to a torch backend that imports `motif`; that is the only path that
does.

| file | what it holds |
| --- | --- |
| `schemas.py` | the request bodies of the two POST routes (pydantic) |
| `backends.py` | `OnnxBackend`, `TorchBackend`, `build_backend(...)` |
| `app.py` | `create_app(backend, ...)` and the routes, SSE included |
| `server.py` | the command line entry that builds a backend and serves it |

The ONNX runtime it drives is `motif.onnx.runtime` (tokenizer archive reader,
chat template, sampler, two-graph engine), which is torch-free, so the default
path imports neither `torch` nor anything else that needs it.

Install the optional pieces:

```bash
pip install codon-motif[service,onnx]
```

## ONNX backend, without torch

```
python -m motif.onnx.export_onnx --weights D:/MotifChord/pretrained_base.safetensors \
    --precision fp16 --out .research/onnx/out/fp16
python -m motif.service.server --bundle .research/onnx/out/fp16 --no-torch --port 8000
```

`--no-torch` is not required for the ONNX path; it makes importing `torch`
impossible, so `/health` reporting `torch_imported: false` is proof rather than
a promise. The bundle defaults to `$MOTIF_ONNX_BUNDLE` and the tokenizer to
`$MOTIF_TOKENIZER`, then `./tokenizer.zip`, then the one at the root of the
checkout.

## Torch backend

```
python -m motif.service.server --model D:/MotifChord/sft_test.pt --device cuda --port 8000
python -m motif.service.server --model .tmp/demo/pretrain/last.pt --device cpu
```

`--model` takes any checkpoint `MotifChordModel.from_pretrained` understands
(pipeline checkpoint, bare state dict, safetensors file or shard directory), so
the hyper-parameters are inferred when the file carries none. `build_backend`
accepts a ready `MotifChordModel` instance as well, which is the "在传入 model
实例时再使用 motif" case:

```python
from motif.model import MotifChordModel
from motif.service import build_backend, create_app

backend = build_backend(model=MotifChordModel.from_pretrained('sft_test.pt'))
app = create_app(backend)
```

## Routes

| route | notes |
| --- | --- |
| `GET /health` | backend, weights, device, providers, prefill chunks, context, vocab, and whether `torch` / `motif` are imported |
| `GET /v1/models` | one entry, the `--model-id` |
| `POST /v1/chat/completions` | `messages` go through the chat template; `stream` uses SSE |
| `POST /v1/completions` | a raw `prompt`, no template |

Request knobs: `max_tokens` / `max_completion_tokens` (default 512, clamped to
what is left of the 4096-token context), `temperature` (0.7), `top_p`, `top_k`,
`repetition_penalty` (1.15), `seed`, `stop`, `stream`,
`stream_options.include_usage`, `reasoning_effort` and `enable_thinking`.

A reasoning checkpoint spends real budget before the answer starts: the SFT weights
write 170-190 tokens of thought before `<|thought_end|>`, so a small `max_tokens`
fills `reasoning_content` and leaves `content` empty with `finish_reason: "length"`.
Either ask for 512 tokens or more, or send `enable_thinking: false` to skip the
thought section entirely.

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"从前有座山，"}],"max_tokens":24}'
```

```python
from openai import OpenAI

client = OpenAI(base_url='http://127.0.0.1:8000/v1', api_key='not-needed')
stream = client.chat.completions.create(
    model='codon-motif-onnx',
    messages=[{'role': 'user', 'content': '从前有座山，'}],
    max_tokens=64,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    print(delta.reasoning_content or delta.content or '', end='')
```

`--api-key KEY` requires `Authorization: Bearer KEY`; `--system TEXT` prepends a
system prompt when the request carries none.

## What a request maps to

- **Thinking** is on by default: the generation prompt opens
  `<|thought_start|><|effort_high|>`, and the reply is split at
  `<|thought_end|>` into `reasoning_content` and `content`. `enable_thinking:
  false` renders an empty reasoning section instead, so the whole reply is
  `content`.
- **`reasoning_effort`** takes `minimal`, `low`, `medium`, `high`, `xhigh`,
  `max` or `ultra` and is aliased to the three effort tokens `low` / `high` /
  `max`. Anything else is a 400 carrying the template's own message.
- **`stop`** is a string or a list of strings, and each one has to encode to a
  single token, because the runtime stops on token ids; a longer string is a 400
  that says how many tokens it took. Non-streaming cuts the reply at the string,
  streaming holds back the tail that could still become one.
- **`n > 1`** is a 400: the runtime decodes one sequence at a time.
- **Usage** counts real tokens, except in a streaming reply where the completion
  tokens are counted by re-encoding the finished text.

## Measured

Both backends were smoked with `.tmp/service_probe.py` (`--bundle
.research/onnx/out/fp16 --no-torch` on port 8765, `.tmp/demo/pretrain/last.pt
--device cpu` on port 8766). The weights are base models, so the text itself is
word salad; what the run checks is the protocol.

| check | ONNX (`--no-torch`) | torch |
| --- | --- | --- |
| `/health` | `torch_imported: false`, `motif_imported: false`, `providers: ["CPUExecutionProvider"]`, `prefill_chunks: [64, 256, 512, 1024]`, vocab 40960 | `torch_imported: true`, `motif_imported: true`, `model_class: MotifChordModel`, device cpu |
| chat, thinking | 200, reasoning section in `reasoning_content` | 200, same |
| chat, `enable_thinking: false` | 200, answer in `content` | 200, same |
| `reasoning_effort: "low"` | 200 | 200 |
| `/v1/completions` | 200 | 200 |
| streaming | 200 `text/event-stream`, 28 SSE lines for a 24-token reply (role delta + 23 text deltas + finish + usage + `[DONE]`) | 200, 28 lines |
| `stop: "的"` | `finish_reason: "stop"` after 18 tokens, the string absent | `finish_reason: "length"` (it never came up) |
| `stop: "你好世界"` | 400: `encodes to 3 tokens` | 400: same |
| `reasoning_effort: "nonsense"` | 400: `Unexpected effort level: nonsense` | 400: same |
| `n: 2` | 400: `n > 1 is not supported` | 400: same |

The reply text keeps the protocol markers - a base model can emit
`<|unused_26|>` and a client that wants plain prose strips the special tokens it
knows. That is deliberate: `split_thought` needs `<|thought_start|>` and
`<|thought_end|>` to be there.

## Not done here

- Tool calling only renders the declaration block; the model's
  `<|tool_call_start|>...<|tool_call_end|>` output is passed through as text, no
  parsing, no `tool_calls` in the response.
- One sequence per request, no batching, no prefix cache across requests: the
  ONNX graphs carry the state in and out per call.
