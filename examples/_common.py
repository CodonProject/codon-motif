'''
Shared helpers for the example scripts: resolve the tokenizer / checkpoint paths and
load a model on the right device.

The examples assume only two things survive on disk: this repository (with
`tokenizer.zip` next to `pyproject.toml`) and a checkpoint written by
`motif.pipeline`. Every default below is relative to the current directory, so the
scripts can be copied into another folder and still work.
'''

import argparse
import os
from typing import Optional, Tuple

import torch

# Triton compiles its kernels into ~/.triton/cache on first use. Point it at a folder
# next to the checkout instead, so the examples also work where the home directory is
# read-only: a failed compile falls back to the eager kernels (correct, just slower).
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('TRITON_CACHE_DIR', os.path.join(_HERE, '.tmp', 'triton'))

from motif.model.motif import MotifChordModel
from motif.pipeline import resolve_device
from motif.tokenizer.tokenizer import PackedTokenizer


TOKENIZER_NAME = 'tokenizer.zip'


def find_tokenizer(explicit: Optional[str] = None) -> str:
    '''
    Locate `tokenizer.zip`.

    Args:
        explicit (Optional[str]): Path given on the command line; wins when set.

    Returns:
        str: Path of the tokenizer.

    Raises:
        FileNotFoundError: If nothing was found, with the paths that were tried.
    '''
    tried = []
    candidates = [explicit] if explicit else [TOKENIZER_NAME, os.path.join('..', TOKENIZER_NAME)]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
        tried.append(candidate)
    raise FileNotFoundError(
        f'no {TOKENIZER_NAME} found (tried {tried}); train one with '
        f'motif.tokenizer.tokenizer.create_tokenizer_trainer or pass --tokenizer'
    )


def load_tokenizer(args: argparse.Namespace) -> PackedTokenizer:
    '''
    Build the tokenizer from `--tokenizer` (or the default location).

    Args:
        args (argparse.Namespace): Parsed arguments carrying `tokenizer`.

    Returns:
        PackedTokenizer: The tokenizer, chat template included.
    '''
    return PackedTokenizer(find_tokenizer(getattr(args, 'tokenizer', None)))


def load_model(
    checkpoint: str,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[MotifChordModel, torch.device]:
    '''
    Load a checkpoint and move it to the resolved device.

    `MotifChordModel.from_pretrained` accepts every layout this project writes: a
    full pipeline checkpoint, a bare `state_dict()`, or a safetensors file, with the
    architecture recovered from the tensor shapes when the checkpoint carries no
    config.

    Args:
        checkpoint (str): Checkpoint path.
        device (Optional[str]): Device from the command line; None / 'auto' picks.
        verbose (bool): Print the layout and parameter count.

    Returns:
        Tuple[MotifChordModel, torch.device]: The model in eval mode, and its device.
    '''
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f'checkpoint not found: {checkpoint}')

    model = MotifChordModel.from_pretrained(checkpoint, map_location='cpu', verbose=verbose)
    target = resolve_device(model, device)
    model = model.to(target).eval()
    if verbose:
        params = sum(p.numel() for p in model.parameters())
        print(f'[*] model on {target}, {params / 1e6:.1f} M params')
    return model, target


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    '''
    Add the tokenizer / device arguments every example takes.

    Args:
        parser (argparse.ArgumentParser): Parser to extend.

    Returns:
        argparse.ArgumentParser: The same parser.
    '''
    parser.add_argument('--tokenizer', default=None,
                        help=f'path of {TOKENIZER_NAME} (default: ./{TOKENIZER_NAME})')
    parser.add_argument('--device', default='auto',
                        help="'auto' (default), 'cpu', 'cuda', or 'cuda:1'")
    return parser
