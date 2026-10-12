import os

BUNDLE_ENV = 'MOTIF_ONNX_BUNDLE'
TOKENIZER_ENV = 'MOTIF_TOKENIZER'

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))


def default_bundle():
    '''
    The bundle directory named by the environment.

    Returns:
        Optional[str]: `MOTIF_ONNX_BUNDLE`, or None when it is not set.
    '''
    return os.environ.get(BUNDLE_ENV)


def default_tokenizer():
    '''
    Locate the tokenizer archive.

    `MOTIF_TOKENIZER` wins, then `tokenizer.zip` in the working directory, then
    the one at the root of the checkout this module lives in.

    Returns:
        str: Path to the archive. It may not exist; the caller checks.
    '''
    named = os.environ.get(TOKENIZER_ENV)
    if named:
        return named
    local = os.path.join(os.getcwd(), 'tokenizer.zip')
    if os.path.exists(local):
        return local
    return os.path.join(ROOT, 'tokenizer.zip')
