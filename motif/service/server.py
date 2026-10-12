import argparse

from motif.onnx.runtime.no_torch import block_torch
from motif.service.app import create_app, torch_imported
from motif.service.backends import DEFAULT_MAX_NEW_TOKENS, build_backend

DESCRIPTION = (
    'Serve a MotifChord ONNX bundle (or a torch checkpoint) in the OpenAI '
    'format: /v1/models, /v1/chat/completions, /v1/completions and /health.'
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bundle', default=None,
                        help='ONNX bundle directory: decode.onnx, prefill_<T>.onnx, weights.bin')
    parser.add_argument('--model', default=None,
                        help='a MotifChord checkpoint; the torch backend is used, so motif is imported')
    parser.add_argument('--tokenizer', default=None,
                        help='tokenizer.zip; defaults to ./tokenizer.zip or MOTIF_TOKENIZER')
    parser.add_argument('--model-id', default=None, help='the id reported by /v1/models')
    parser.add_argument('--system', default=None, help='a system prompt prepended to every chat')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--context-length', type=int, default=4096)
    parser.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help='default completion length when a request sets none; a reasoning model '
                             'needs room for its thought section before the answer starts')
    parser.add_argument('--providers', default=None,
                        help='comma separated ONNX Runtime providers, for example CPUExecutionProvider')
    parser.add_argument('--device', default=None, help='torch backend only, for example cuda or cpu')
    parser.add_argument('--api-key', default=None,
                        help='require this key as `Authorization: Bearer <key>`')
    parser.add_argument('--no-torch', action='store_true',
                        help='make importing torch impossible, to prove the ONNX path never needs it')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.no_torch:
        block_torch()

    providers = None
    if args.providers:
        providers = [name.strip() for name in args.providers.split(',') if name.strip()]

    backend = build_backend(model_path=args.model, bundle=args.bundle, tokenizer=args.tokenizer,
                            providers=providers, model_id=args.model_id,
                            context_length=args.context_length, system=args.system,
                            device=args.device, verbose=args.verbose)

    info = backend.info()
    print(f'[info] backend: {info["backend"]} ({info["weights"]})', flush=True)
    if 'bundle' in info:
        print(f'[info] bundle: {info["bundle"]}', flush=True)
        print(f'[info] prefill chunks: {info["prefill_chunks"]}', flush=True)
        print(f'[info] providers: {info["providers"]}', flush=True)
    if 'device' in info:
        print(f'[info] device: {info["device"]}', flush=True)
    print(f'[info] tokenizer: {info["tokenizer"]}', flush=True)
    print(f'[info] context: {info["context_length"]} tokens, vocab {info["vocab_size"]}',
          flush=True)
    print(f'[info] torch imported: {torch_imported()}', flush=True)
    print(f'[info] serving on http://{args.host}:{args.port}', flush=True)

    app = create_app(backend, api_key=args.api_key, system=args.system,
                     default_max_tokens=args.max_tokens)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
