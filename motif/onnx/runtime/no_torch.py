import sys


def block_torch():
    '''Forget every loaded torch module and make later imports fail.'''
    for name in [name for name in sys.modules
                 if name == 'torch' or name.startswith('torch.')]:
        del sys.modules[name]
    sys.modules['torch'] = None


def torch_blocked():
    '''
    Returns:
        bool: Whether torch is currently unavailable to this process.
    '''
    return sys.modules.get('torch', 'missing') is None
