import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import load_file as _st_load_file
    from safetensors.torch import save_file as _st_save_file
    HAS_SAFETENSORS = True
except ImportError:
    _st_load_file = None
    _st_save_file = None
    HAS_SAFETENSORS = False


_DEFAULT_EXCLUDE = ('embed', 'lm_head')


@dataclass
class LoRAConfig:
    '''
    Shape and placement of the low-rank adapters.

    Attributes:
        r (int): Rank of the adapters.
        alpha (float): Scaling numerator; the delta is multiplied by `alpha / r`.
        dropout (float): Dropout applied to the adapter input.
        target_modules (Optional[Tuple[str, ...]]): Leaf module names to wrap; None
            wraps every `nn.Linear` that `exclude` does not rule out.
        exclude (Tuple[str, ...]): Substrings that keep a module out of the 'every
            linear' selection (ignored when `target_modules` is given).
        train_bias (bool): Keep the wrapped layer's bias trainable as well.
    '''

    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: Optional[Tuple[str, ...]] = None
    exclude: Tuple[str, ...] = _DEFAULT_EXCLUDE
    train_bias: bool = False

    def __post_init__(self) -> None:
        '''
        Validate the adapter geometry.

        Raises:
            ValueError: If `r` is not positive, if `alpha` is not positive, or if
                `dropout` is outside `[0, 1)`.
        '''
        if self.r < 1:
            raise ValueError(f'r must be >= 1, got {self.r}')
        if self.alpha <= 0:
            raise ValueError(f'alpha must be > 0, got {self.alpha}')
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f'dropout must be in [0, 1), got {self.dropout}')
        if self.target_modules is not None:
            self.target_modules = tuple(self.target_modules)
        self.exclude = tuple(self.exclude)

    @property
    def scaling(self) -> float:
        '''
        Returns:
            float: The factor the low-rank delta is multiplied by (`alpha / r`).
        '''
        return self.alpha / self.r

    def matches(self, name: str) -> bool:
        '''
        Decide whether a module named `name` receives an adapter.

        Args:
            name (str): Dotted module name inside the model.

        Returns:
            bool: True when the module is a target.
        '''
        if self.target_modules is not None:
            return name.rsplit('.', 1)[-1] in self.target_modules
        return not any(part in name for part in self.exclude)

    def state(self) -> Dict[str, Any]:
        '''
        Returns:
            Dict[str, Any]: A JSON-serializable description of this configuration
            (stored next to an adapter so it can be rebuilt later).
        '''
        return {
            'r': self.r,
            'alpha': self.alpha,
            'dropout': self.dropout,
            'target_modules': list(self.target_modules) if self.target_modules else None,
            'exclude': list(self.exclude),
            'train_bias': self.train_bias,
        }

    @classmethod
    def from_state(cls, state: Optional[Dict[str, Any]]) -> 'LoRAConfig':
        '''
        Rebuild a configuration from `state()`.

        Args:
            state (Optional[Dict[str, Any]]): Output of `state()`; None (or an empty
                dict) yields the defaults.

        Returns:
            LoRAConfig: The reconstructed configuration.
        '''
        if not state:
            return cls()
        known = ('r', 'alpha', 'dropout', 'target_modules', 'exclude', 'train_bias')
        return cls(**{k: v for k, v in state.items() if k in known})


class LoRALinear(nn.Module):
    '''
    An `nn.Linear` plus a trainable rank-`r` delta: `W x + (alpha / r) B A x`.

    The base layer is kept as a submodule (state-dict keys `<name>.base.weight`), so a
    wrapped model still loads a plain checkpoint after `merge_lora(..., unload=True)`.
    '''

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        '''
        Freeze `base` and attach the two low-rank factors.

        Args:
            base (nn.Linear): The layer to adapt.
            config (LoRAConfig): Adapter geometry.
        '''
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)

        self.r = config.r
        self.scaling = float(config.scaling)
        factory = {'device': base.weight.device, 'dtype': base.weight.dtype}
        self.lora_a = nn.Parameter(torch.empty(config.r, base.in_features, **factory))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, config.r, **factory))
        self.reset_parameters()

        if config.train_bias and base.bias is not None:
            base.bias.requires_grad_(True)

        self.lora_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.merged = False

    def reset_parameters(self) -> None:
        '''
        Initialize `lora_a` (kaiming uniform) and `lora_b` (zero), so the delta is
        exactly zero before the first step.
        '''
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    @property
    def in_features(self) -> int:
        '''
        Returns:
            int: Input width of the wrapped layer.
        '''
        return self.base.in_features

    @property
    def out_features(self) -> int:
        '''
        Returns:
            int: Output width of the wrapped layer.
        '''
        return self.base.out_features

    @property
    def weight(self) -> nn.Parameter:
        '''
        Returns:
            nn.Parameter: The frozen base weight.
        '''
        return self.base.weight

    @property
    def bias(self) -> Optional[nn.Parameter]:
        '''
        Returns:
            Optional[nn.Parameter]: The base bias, if there is one.
        '''
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Apply the frozen layer and, while not merged, add the low-rank delta.

        Args:
            x (torch.Tensor): Input activations.

        Returns:
            torch.Tensor: `base(x) + scaling * B(A(dropout(x)))`.
        '''
        out = self.base(x)
        if self.merged or self.scaling == 0.0:
            return out
        delta = F.linear(F.linear(self.lora_dropout(x), self.lora_a), self.lora_b)
        return out + delta * self.scaling

    def delta_weight(self) -> torch.Tensor:
        '''
        Returns:
            torch.Tensor: The delta that `merge()` adds to the base weight, in the
            base weight's dtype and device.
        '''
        delta = (self.lora_b @ self.lora_a) * self.scaling
        return delta.to(device=self.base.weight.device, dtype=self.base.weight.dtype)

    def merge(self) -> 'LoRALinear':
        '''
        Fold the delta into the base weight (idempotent).

        Returns:
            LoRALinear: This module, for chaining.
        '''
        if not self.merged:
            with torch.no_grad():
                self.base.weight.data += self.delta_weight()
            self.merged = True
        return self

    def unmerge(self) -> 'LoRALinear':
        '''
        Undo `merge()` (idempotent).

        Returns:
            LoRALinear: This module, for chaining.
        '''
        if self.merged:
            with torch.no_grad():
                self.base.weight.data -= self.delta_weight()
            self.merged = False
        return self

    def extra_repr(self) -> str:
        '''
        Returns:
            str: A one-line description used by `print(model)`.
        '''
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'r={self.r}, scaling={self.scaling:g}, merged={self.merged}')


def iter_lora_modules(model: nn.Module) -> Iterator[Tuple[str, LoRALinear]]:
    '''
    Walk every wrapped layer of a model.

    Args:
        model (nn.Module): Model to inspect.

    Yields:
        Tuple[str, LoRALinear]: The dotted name and the adapter module.
    '''
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def lora_parameter_count(model: nn.Module) -> Tuple[int, int]:
    '''
    Count the adapter parameters and the trainable ones.

    Args:
        model (nn.Module): Model to inspect.

    Returns:
        Tuple[int, int]: `(adapter parameters, trainable parameters)`.
    '''
    adapter = sum(p.numel() for _, module in iter_lora_modules(model)
                  for p in (module.lora_a, module.lora_b))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return adapter, trainable


def apply_lora(model: nn.Module, config: Optional[LoRAConfig] = None) -> LoRAConfig:
    '''
    Freeze the whole model and wrap every target `nn.Linear` with an adapter.

    Wrapping happens in place; the modules keep their names, so the model's forward is
    untouched and a checkpoint of the base weights still loads into `module.base`.
    Only the `lora_a` / `lora_b` tensors stay trainable::

        apply_lora(model, LoRAConfig(r=8, alpha=16.0, dropout=0.05))
        save_lora(model, 'adapter.safetensors')  # adapter only, a few MB
        load_lora(model, 'adapter.safetensors')  # back into a wrapped model
        merge_lora(model, unload=True)           # fold the adapters into the base weights

    Args:
        model (nn.Module): Model to adapt.
        config (Optional[LoRAConfig]): Adapter geometry; None uses the defaults.

    Returns:
        LoRAConfig: The configuration that was applied.

    Raises:
        ValueError: If no `nn.Linear` matches the configuration (which would silently
            train nothing).
    '''
    config = LoRAConfig() if config is None else config
    if not isinstance(config, LoRAConfig):
        raise TypeError(f'config must be a LoRAConfig, got {type(config).__name__}')

    for param in model.parameters():
        param.requires_grad_(False)

    targets: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            continue
        if isinstance(module, nn.Linear) and name and config.matches(name):
            targets.append((name, module))

    if not targets:
        raise ValueError(
            'no nn.Linear matched the LoRA configuration: set target_modules or relax '
            f'exclude={config.exclude!r}'
        )

    for name, module in targets:
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        parent._modules[child_name] = LoRALinear(module, config)
    model.train(model.training)
    return config


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    '''
    Re-freeze everything except the adapter tensors (and the biases the adapters own).

    Args:
        model (nn.Module): Model whose adapters stay trainable.
    '''
    lora_params = set()
    for _, module in iter_lora_modules(model):
        lora_params.add(id(module.lora_a))
        lora_params.add(id(module.lora_b))
        if module.base.bias is not None and module.base.bias.requires_grad:
            lora_params.add(id(module.base.bias))
    for param in model.parameters():
        param.requires_grad_(id(param) in lora_params)


def trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    '''
    Args:
        model (nn.Module): Model to inspect.

    Returns:
        List[nn.Parameter]: Every parameter that currently requires a gradient.
    '''
    return [p for p in model.parameters() if p.requires_grad]


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    '''
    Extract only the adapter tensors from a model.

    Args:
        model (nn.Module): A wrapped model (or a plain one, which yields nothing).

    Returns:
        Dict[str, torch.Tensor]: Detached CPU copies keyed by the model's own names.
    '''
    return {
        name: tensor.detach().to('cpu').clone()
        for name, tensor in model.state_dict().items()
        if _is_lora_key(name)
    }


def _is_lora_key(name: str) -> bool:
    '''
    Args:
        name (str): A state-dict key.

    Returns:
        bool: True when the key belongs to an adapter tensor.
    '''
    return name.endswith('.lora_a') or name.endswith('.lora_b')


def _strip_data_parallel(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    '''
    Drop a leading `module.` (DDP) from every key.

    Args:
        state (Dict[str, torch.Tensor]): A state dict.

    Returns:
        Dict[str, torch.Tensor]: The same dict, un-prefixed when it was prefixed.
    '''
    if state and all(key.startswith('module.') for key in state):
        return {key[len('module.'):]: value for key, value in state.items()}
    return state


def save_lora(
    model: nn.Module,
    path: str,
    config: Optional[LoRAConfig] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, torch.Tensor]:
    '''
    Write the adapters to `path`.

    A `.safetensors` path is written with `safetensors` (the configuration travels in
    the file's metadata, which has to be strings); any other extension is written as a
    `torch.save` dict of `{'__lora__', 'lora_spec', 'adapter'}`.

    Args:
        model (nn.Module): A wrapped model.
        path (str): Destination file; its parent directory is created.
        config (Optional[LoRAConfig]): Configuration stored next to the adapter.
        metadata (Optional[Dict[str, str]]): Extra metadata merged into the header.

    Returns:
        Dict[str, torch.Tensor]: The adapter tensors that were written.

    Raises:
        RuntimeError: If a `.safetensors` destination is requested while the optional
            dependency is missing.
        ValueError: If the model carries no adapter at all.
    '''
    state = lora_state_dict(model)
    if not state:
        raise ValueError('the model has no LoRA adapter to save; call apply_lora() first')

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    spec = (config or LoRAConfig()).state()

    tmp = str(path) + '.tmp'
    if str(path).endswith('.safetensors'):
        if not HAS_SAFETENSORS:
            raise RuntimeError(
                'safetensors is not installed but a .safetensors path was requested; '
                'use a .pt path instead or pip install safetensors'
            )
        header = {'format': 'motif-lora', 'lora_spec': json.dumps(spec)}
        header.update(metadata or {})
        _st_save_file(state, tmp, metadata=header)
    else:
        torch.save({'__lora__': True, 'lora_spec': spec, 'adapter': state}, tmp)
    os.replace(tmp, str(path))
    return state


def load_lora(
    model: nn.Module,
    source: Union[str, os.PathLike, Dict[str, torch.Tensor]],
    strict: bool = True,
    map_location: Union[str, torch.device] = 'cpu',
) -> LoRAConfig:
    '''
    Load adapters into a model that `apply_lora` already wrapped.

    `source` may be a `.safetensors` file, a `torch.save` file written by `save_lora`,
    a plain adapter state dict, or a full model state dict that happens to contain
    adapter keys.

    Args:
        model (nn.Module): A wrapped model.
        source (Union[str, os.PathLike, Dict[str, torch.Tensor]]): Where the adapter
            comes from.
        strict (bool): Raise when an adapter tensor of the model is missing from
            `source`.
        map_location (Union[str, torch.device]): Device the tensors are loaded onto.

    Returns:
        LoRAConfig: The configuration recorded with the adapter (defaults when the
        file does not carry one).

    Raises:
        RuntimeError: If `source` has to be read from disk but the format is not
            supported, or if `strict` is on and a tensor is missing.
    '''
    spec_state: Optional[Dict[str, Any]] = None
    metadata: Dict[str, str] = {}

    if isinstance(source, (str, os.PathLike)):
        path = str(source)
        if path.endswith('.safetensors'):
            if not HAS_SAFETENSORS:
                raise RuntimeError('safetensors is not installed but a .safetensors file was given')
            state = _st_load_file(path, device=str(map_location))
            try:
                with open(path, 'rb') as handle:
                    header_len = int.from_bytes(handle.read(8), 'little')
                    header = json.loads(handle.read(header_len).decode('utf-8'))
                metadata = header.get('__metadata__', {})
            except Exception:
                metadata = {}
            if 'lora_spec' in metadata:
                spec_state = json.loads(metadata['lora_spec'])
        else:
            payload = torch.load(path, map_location=map_location, weights_only=False)
            if isinstance(payload, dict) and 'adapter' in payload:
                state = payload['adapter']
                spec_state = payload.get('lora_spec')
            elif isinstance(payload, dict):
                state = payload
            else:
                raise RuntimeError(f'unsupported LoRA file: {path!r} holds a {type(payload).__name__}')
    else:
        state = dict(source)

    state = _strip_data_parallel({k: v for k, v in state.items() if _is_lora_key(k)})
    if strict:
        expected = set(lora_state_dict(model))
        missing = sorted(expected - set(state))
        if missing:
            preview = ', '.join(missing[:5])
            raise RuntimeError(
                f'{len(missing)} adapter tensor(s) are missing from the file, starting with '
                f'{preview}; wrap the model with the matching configuration or pass strict=False'
            )
    model.load_state_dict(state, strict=False)
    return LoRAConfig.from_state(spec_state)


def merge_lora(model: nn.Module, unload: bool = False) -> nn.Module:
    '''
    Fold every adapter into its base weight.

    With `unload=True` the wrapped layers are replaced by their plain `nn.Linear`, which
    gives back a model whose state dict is exactly a base checkpoint again (the ordinary
    way to export an adapter as a standalone model).

    Args:
        model (nn.Module): A wrapped model.
        unload (bool): Replace the wrappers instead of only folding them.

    Returns:
        nn.Module: `model`, for chaining.
    '''
    modules = list(iter_lora_modules(model))
    for _, module in modules:
        module.merge()
    if unload:
        for name, module in modules:
            parent_name, _, child_name = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            parent._modules[child_name] = module.base
    return model


def unmerge_lora(model: nn.Module) -> nn.Module:
    '''
    Undo `merge_lora(model, unload=False)`.

    Args:
        model (nn.Module): A wrapped model whose adapters were folded in.

    Returns:
        nn.Module: `model`, for chaining.
    '''
    for _, module in iter_lora_modules(model):
        module.unmerge()
    return model


def lora_summary(model: nn.Module, config: Optional[LoRAConfig] = None) -> str:
    '''
    Build a one-line description of the adapters of a model.

    Args:
        model (nn.Module): A wrapped model.
        config (Optional[LoRAConfig]): Configuration to report; None reads the
            geometry from the first adapter.

    Returns:
        str: A human-readable summary (empty when the model has no adapter).
    '''
    modules = list(iter_lora_modules(model))
    if not modules:
        return ''
    first = modules[0][1]
    adapter, trainable = lora_parameter_count(model)
    r = config.r if config is not None else first.r
    alpha = (config.alpha if config is not None else first.scaling * first.r)
    return (f'LoRA r={r} alpha={alpha:g} layers={len(modules)} '
            f'adapter_params={adapter:,} trainable={trainable:,}')
