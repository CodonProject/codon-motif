import os
import shutil
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, ROOT)

os.environ.setdefault('TRITON_CACHE_DIR', os.path.join(ROOT, '.tmp', 'triton'))


def _mkdtemp(suffix=None, prefix=None, dir=None):
    '''Portable stand-in for `tempfile.mkdtemp` that the tests can write into.

    `tempfile` is deliberately not used to make the directory. The suite writes tokenizer
    corpora, parquet files and checkpoints into what it asks for, and on Windows a directory
    that `tempfile` made can come back refusing writes (`PermissionError`, or `os error 5`
    from torch/safetensors). `os.makedirs` produces the same directory with the same name,
    so the tests keep their portable layout.
    '''
    base = dir if dir is not None else tempfile.gettempdir()
    for _ in range(tempfile.TMP_MAX):
        path = os.path.join(base, (prefix or '') + next(tempfile._RandomNameSequence()) + (suffix or ''))
        try:
            os.makedirs(path)
        except FileExistsError:
            continue
        return path
    raise FileExistsError('no usable temporary directory name found')


class _TemporaryDirectory:
    '''Portable stand-in for `tempfile.TemporaryDirectory`; same contract, writable directory.'''

    def __init__(self, suffix=None, prefix=None, dir=None, ignore_cleanup_errors=False):
        self.name = _mkdtemp(suffix or '', prefix or '', dir)
        self._ignore_cleanup_errors = ignore_cleanup_errors

    def __enter__(self):
        return self.name

    def __exit__(self, *exc_info):
        self.cleanup()

    def cleanup(self):
        '''Remove the directory, mirroring `tempfile.TemporaryDirectory.cleanup`.'''
        if self.name is not None:
            shutil.rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)
            self.name = None


@pytest.fixture(autouse=True, scope='session')
def _tempdirs_inside_the_repo():
    '''Keep every temporary artefact the tests build inside `<repo>/.tmp`.

    The suite never reads those artefacts back after the run, so putting them next to the
    tests keeps them out of the shared system temp root, where creating or writing inside a
    freshly made directory can be refused (a sandbox that only allows the workspace, or a CI
    runner with a locked-down temp root). The repository ignores `.tmp/`, so nothing leaks
    into a commit. `_mkdtemp` covers the same ground for the callers that ask `tempfile`
    directly.
    '''
    previous_tempdir = tempfile.tempdir
    previous_mkdtemp = tempfile.mkdtemp
    previous_temporary_directory = tempfile.TemporaryDirectory

    root = os.path.join(ROOT, '.tmp')
    os.makedirs(root, exist_ok=True)
    tempfile.tempdir = root
    tempfile.mkdtemp = _mkdtemp
    tempfile.TemporaryDirectory = _TemporaryDirectory
    try:
        yield root
    finally:
        tempfile.tempdir = previous_tempdir
        tempfile.mkdtemp = previous_mkdtemp
        tempfile.TemporaryDirectory = previous_temporary_directory
