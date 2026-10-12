import os

from motif.model import MotifChordModel

PREFERRED_NAMES = ('model.safetensors', 'model.pt', 'last.pt', 'last.pth',
                   'pytorch_model.bin')
WEIGHT_SUFFIXES = ('.safetensors', '.pt', '.pth', '.bin', '.ckpt')


def resolve_source(source):
    '''
    Resolve a model / path into `(path, description)`.

    Args:
        source (Union[MotifChordModel, str, os.PathLike]): A live model, a weight
            file, or a directory holding one.

    Returns:
        Tuple[Optional[str], str]: The checkpoint path (`None` for a live model)
        and a one-line description of what was found.

    Raises:
        FileNotFoundError: If the path does not exist, or holds no weights.
        TypeError: If `source` is neither a model nor a path.
    '''
    if isinstance(source, MotifChordModel):
        return None, f'a live {type(source).__name__}'
    if not isinstance(source, (str, os.PathLike)):
        raise TypeError(
            f'source must be a MotifChordModel or a path, got {type(source).__name__}.')
    path = os.path.abspath(os.fspath(source))
    if not os.path.exists(path):
        raise FileNotFoundError(f'weights not found: {path!r}')
    if os.path.isfile(path):
        return path, f'file {os.path.basename(path)}'
    chosen = pick_in_directory(path)
    return chosen, f'directory {os.path.basename(path)!r} -> {os.path.basename(chosen)}'


def pick_in_directory(directory):
    '''
    The checkpoint file to read from `directory`, by the documented order.

    Raises:
        FileNotFoundError: If the directory holds no recognisable weights.
    '''
    entries = sorted(os.listdir(directory))
    for name in PREFERRED_NAMES:
        if name in entries:
            return os.path.join(directory, name)
    for name in entries:
        if name.endswith('.index.json'):
            return os.path.join(directory, name)
    candidates = [name for name in entries if name.endswith(WEIGHT_SUFFIXES)]
    if not candidates:
        raise FileNotFoundError(
            f'no weights in {directory!r}: looked for {list(PREFERRED_NAMES)}, an '
            f'*.index.json and any of {list(WEIGHT_SUFFIXES)}, found {entries}.')
    newest = max(candidates, key=lambda name: os.path.getmtime(os.path.join(directory, name)))
    return os.path.join(directory, newest)


def load_model(source, config=None, verbose=False):
    '''
    Load `source` as an eval-mode `MotifChordModel`.

    Args:
        source (Union[MotifChordModel, str, os.PathLike]): Model or weights.
        config (Optional[Union[MotifChordConfig, Dict, str]]): Explicit config,
            passed through to `MotifChordModel.from_pretrained`.
        verbose (bool): Let `from_pretrained` print its layout report.

    Returns:
        MotifChordModel: The model, in eval mode, on the CPU.
    '''
    if isinstance(source, MotifChordModel):
        return source.eval()
    path, _ = resolve_source(source)
    model = MotifChordModel.from_pretrained(path, config=config, verbose=verbose)
    return model.eval()
