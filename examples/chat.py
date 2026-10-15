'''
Interactive chat, the smallest end-to-end use of the library: one checkpoint, one
tokenizer, one `ChatSession`.

    python examples/chat.py --ckpt D:/checkpoints/sft/last.pt
    python examples/chat.py --ckpt sft.pt --system "You are a terse assistant." \
        --max-new-tokens 128 --temperature 0.7

Inside the prompt:

    /reset   forget the conversation, keep the system message
    /exit    leave (Ctrl-C and Ctrl-D also work)

`ChatSession` renders the prompt with the tokenizer's chat template, samples through
a `Sampler`, streams the answer as it appears, and keeps the previous turn's cache so
a turn only feeds the new tokens. The reasoning section of the template is printed
before the answer; `--no-think` switches it off.
'''

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples._common import add_common_arguments, load_model, load_tokenizer

from motif.chat import ChatSession
from motif.model.sampler import Sampler


def parse_args() -> argparse.Namespace:
    '''
    Returns:
        argparse.Namespace: The command line of the console chat.
    '''
    parser = argparse.ArgumentParser(description='chat with a codon-motif checkpoint')
    parser.add_argument('--ckpt', required=True, help='checkpoint to talk to')
    parser.add_argument('--system', default=None, help='system message for the session')
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--repetition-penalty', type=float, default=1.1)
    parser.add_argument('--no-think', action='store_true',
                        help='do not open a reasoning section in the prompt')
    add_common_arguments(parser)
    return parser.parse_args()


def show_reasoning(session: ChatSession, reply) -> None:
    '''
    Print the reasoning section of a finished turn, if the model wrote one.

    Args:
        session (ChatSession): The conversation the turn belongs to.
        reply: The `ChatReply` of that turn.
    '''
    if session.enable_thinking and reply.reasoning.strip():
        print(f'\n  [reasoning] {reply.reasoning.strip()}')


def main() -> int:
    '''
    Returns:
        int: Process exit code.
    '''
    args = parse_args()
    tokenizer = load_tokenizer(args)
    model, _ = load_model(args.ckpt, args.device)

    session = ChatSession(
        model,
        tokenizer,
        system=args.system,
        enable_thinking=not args.no_think,
        max_new_tokens=args.max_new_tokens,
        sampler=Sampler(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        ),
    )

    print('[*] /reset starts over, /exit leaves')
    with session:
        while True:
            try:
                line = input('\nuser > ')
            except (EOFError, KeyboardInterrupt):
                print()
                break
            line = line.strip()
            if not line:
                continue
            if line in ('/exit', '/quit'):
                break
            if line == '/reset':
                session.reset()
                print('[*] conversation cleared')
                continue

            started = time.time()
            session.user(line)
            print('model > ', end='', flush=True)
            try:
                reply = session.send(on_chunk=lambda chunk: print(chunk, end='', flush=True))
            except KeyboardInterrupt:
                print('  [interrupted]')
                continue
            elapsed = max(time.time() - started, 1e-6)
            print(flush=True)

            if not reply.streamed:
                print('(empty answer)')
            show_reasoning(session, reply)
            print(f'  [{reply.new_tokens} tokens in {elapsed:.1f}s '
                  f'({reply.new_tokens / elapsed:.1f} tok/s), '
                  f'{reply.prompt_tokens} prompt tokens'
                  f'{", cache reused" if reply.reused_cache else ""}]')
    print('[*] bye')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
