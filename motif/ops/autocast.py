import functools

import torch

__all__ = ['no_autocast', 'autocast_safe']


def _device_type(args) -> str:
    '''
    Infer the `device_type` from the arguments (the autocast context requires it).

    Args:
        args: Positional arguments; the first tensor found decides the device type.

    Returns:
        str: The device type of that tensor, or the best available guess when no
        tensor is passed.
    '''
    for value in args:
        if isinstance(value, torch.Tensor):
            return value.device.type
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def no_autocast(fn):
    '''
    Wrap a `Function.forward` / `Function.backward` body in an "autocast disabled"
    region.

    The signature deliberately matches the original (`(ctx, *args)`), so autograd
    keeps calling the wrapper like the method it replaced.

    Args:
        fn: The method to wrap.

    Returns:
        Callable: The wrapped method, marked with `__no_autocast__` so that
        `autocast_safe` does not wrap it twice.
    '''
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        '''Run `fn` with autocast disabled, inferring the device from `args`.'''
        with torch.amp.autocast(device_type=_device_type(args), enabled=False):
            return fn(ctx, *args, **kwargs)
    wrapper.__no_autocast__ = True
    return wrapper


def autocast_safe(cls):
    '''
    Class decorator: disable autocast for the forward / backward of a custom
    `torch.autograd.Function`.

        @autocast_safe
        class _MyFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x): ...
            @staticmethod
            def backward(ctx, dy): ...

    Args:
        cls (type): The `torch.autograd.Function` subclass to patch.

    Returns:
        type: The same class, with `forward` / `backward` wrapped; methods that are
        already wrapped are left untouched.
    '''
    for name in ('forward', 'backward'):
        method = cls.__dict__.get(name)
        if method is None or getattr(method, '__no_autocast__', False):
            continue
        setattr(cls, name, no_autocast(method))
    return cls
