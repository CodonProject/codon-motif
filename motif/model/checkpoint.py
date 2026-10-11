import json
import os
import re
import struct
from dataclasses import asdict, dataclass, fields
from enum import Enum, auto
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn

from motif.model.config import MotifChordConfig

try:
    from safetensors import safe_open
    from safetensors.torch import load_file as _safe_load_file
    from safetensors.torch import save_file as _safe_save_file

    HAS_SAFETENSORS = True
except ImportError:
    safe_open = None
    _safe_load_file = None
    _safe_save_file = None
    HAS_SAFETENSORS = False


class PipelinePhase(Enum):
    '''
    Phase of the training pipeline, kept only so that old checkpoints unpickle.

    Pipeline checkpoints store a `TrainState` holding an instance of this enum, and a
    pickle records the module path it was defined under (`motif.pipeline.base`). The
    training pipeline is not part of this package, so the class is recreated here with
    the same name and members and allowlisted for `weights_only` loading: the unpickler
    looks the name up in that allowlist and never has to import the old module. That
    allowlist is `_SAFE_GLOBALS`, which is passed to
    `torch.serialization.safe_globals`.

    Attributes:
        INIT (PipelinePhase): Set up, not training yet.
        SETUP (PipelinePhase): Running `setup()`.
        TRAINING (PipelinePhase): Inside the training loop.
        EVALUATING (PipelinePhase): Running `evaluate()`.
        FINISHED (PipelinePhase): The training loop ended normally.
        INTERRUPTED (PipelinePhase): Stopped by Ctrl+C.
        FAILED (PipelinePhase): Aborted by an exception.
    '''

    INIT = auto()
    SETUP = auto()
    TRAINING = auto()
    EVALUATING = auto()
    FINISHED = auto()
    INTERRUPTED = auto()
    FAILED = auto()


_SAFE_GLOBALS = (
    (PipelinePhase, 'motif.pipeline.base.PipelinePhase'),
)


_WRAPPER_PREFIXES = (
    'module.',
    '_orig_mod.',
    '_forward_module.',
    '_checkpoint_wrapped_module.',
    'model.',
)

_CONFIG_KEYS = ('config', 'model_config', 'motif_config', 'modelconfig', 'config_dict')

_CONFIG_ALIASES = {
    'hidden_size': 'model_dim',
    'd_model': 'model_dim',
    'n_embd': 'model_dim',
    'num_hidden_layers': 'block_n',
    'n_layers': 'block_n',
    'num_layers': 'block_n',
    'num_attention_heads': 'num_heads',
    'n_heads': 'num_heads',
    'num_key_value_heads': 'num_kv_heads',
    'n_kv_heads': 'num_kv_heads',
}

_CONFIG_FIELDS = {f.name for f in fields(MotifChordConfig)}


@dataclass
class LoadedCheckpoint:
    '''
    One checkpoint, resolved down to weights + hyper-parameters.

    Attributes:
        state_dict (Dict[str, torch.Tensor]): The model weights, with wrapper
            prefixes stripped.
        config (Optional[MotifChordConfig]): Config recovered from the file; None
            means the caller has to infer it from `state_dict`.
        source (str): Human-readable description of the detected format, e.g.
            `'safetensors'` or `"pipeline checkpoint ('PretrainPipeline')"`.
        path (str): The file (or directory / index) that was read.
        metadata (Dict[str, str]): Raw safetensors metadata (empty for `.pt`).
    '''

    state_dict: Dict[str, torch.Tensor]
    config: Optional[MotifChordConfig]
    source: str
    path: str
    metadata: Dict[str, str]


def looks_like_safetensors(path: str) -> bool:
    '''
    Decide whether `path` is a safetensors file from its header, not its name.

    The format starts with a little-endian `uint64` header length followed by that
    many bytes of JSON; a plausible length plus a `{` at the expected offset is a
    reliable signature (and a `torch.save` zip never looks like that).

    Args:
        path (str): File to inspect.

    Returns:
        bool: True when the file carries a safetensors header.
    '''
    if path.endswith('.safetensors'):
        return True
    if path.endswith('.index.json'):
        return False
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as handle:
            raw = handle.read(8)
            if len(raw) < 8:
                return False
            header_len = int.from_bytes(raw, 'little')
            if header_len <= 0 or 8 + header_len > size or header_len > 256 * 1024 * 1024:
                return False
            head = handle.read(min(header_len, 64))
        return head.lstrip()[:1] == b'{'
    except OSError:
        return False


def _safetensors_metadata(path: str) -> Dict[str, str]:
    '''
    Read the `__metadata__` block of a safetensors file.

    Args:
        path (str): Safetensors file.

    Returns:
        Dict[str, str]: Metadata entries; empty when unsupported / unreadable.
    '''
    if not HAS_SAFETENSORS:
        return {}
    try:
        with safe_open(path, framework='pt', device='cpu') as handle:
            return dict(handle.metadata() or {})
    except Exception:
        return {}


def _safetensors_shards(path: str) -> Tuple[list, Optional[str]]:
    '''
    Expand a directory / shard index into the list of safetensors files to read.

    Args:
        path (str): A `.safetensors` file, an `*.index.json`, or a directory.

    Returns:
        Tuple[list, Optional[str]]: The shard paths (absolute) and the accompanying
            `config.json` path when one exists.

    Raises:
        FileNotFoundError: If no safetensors file can be found.
    '''
    if path.endswith('.index.json'):
        directory = os.path.dirname(os.path.abspath(path)) or '.'
        with open(path, 'r', encoding='utf-8') as handle:
            index = json.load(handle)
        weight_map = index.get('weight_map') or {}
        shards = sorted({os.path.join(directory, name) for name in weight_map.values()})
        if not shards:
            raise FileNotFoundError(f'{path} lists no weight shards.')
        return shards, _sidecar_config(directory)

    if os.path.isdir(path):
        directory = os.path.abspath(path)
        preferred = os.path.join(directory, 'model.safetensors')
        if os.path.isfile(preferred):
            return [preferred], _sidecar_config(directory)
        shards = sorted(
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if name.endswith('.safetensors')
        )
        if not shards:
            raise FileNotFoundError(f'no .safetensors file under {path!r}.')
        return shards, _sidecar_config(directory)

    return [os.path.abspath(path)], _sidecar_config(os.path.dirname(os.path.abspath(path)))


def _sidecar_config(directory: str) -> Optional[str]:
    '''
    Locate the optional `config.json` sitting next to the weights.

    Args:
        directory (str): Directory holding the weights.

    Returns:
        Optional[str]: Path of a `config.json` next to the weights, when present.
    '''
    candidate = os.path.join(directory, 'config.json')
    return candidate if os.path.isfile(candidate) else None


def config_to_dict(config: MotifChordConfig) -> Dict[str, Any]:
    '''
    Turn a config into plain JSON-compatible data (tuples become lists).

    Args:
        config (MotifChordConfig): The configuration.

    Returns:
        Dict[str, Any]: Every field of the config, ready for `json.dumps`.
    '''
    return {key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(config).items()}


def _drop_tied_weights(sd: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], list]:
    '''
    Drop `lm_head.weight` while it shares storage with `embed.weight`.

    Two distinct storages with equal values are kept, so a genuinely untied model
    never loses its head; only the shared-weight case (the default
    `MotifChordConfig.tied_weight`) is deduplicated, because safetensors refuses
    tensors that alias the same memory.

    Args:
        sd (Dict[str, torch.Tensor]): Weights, canonical names.

    Returns:
        Tuple[Dict[str, torch.Tensor], list]: The weights to write (the input mapping
            when there is nothing to drop) and the names that were dropped.
    '''
    embed = sd.get('embed.weight')
    head = sd.get('lm_head.weight')
    if embed is None or head is None or embed is head:
        return sd, []
    shared = (
        embed.data_ptr() == head.data_ptr()
        and tuple(embed.shape) == tuple(head.shape)
        and embed.device == head.device
    )
    if shared:
        return {key: value for key, value in sd.items() if key != 'lm_head.weight'}, ['lm_head.weight']
    return sd, []


def save_safetensors(
    model_or_state: Any,
    path: str,
    config: Optional[Union[MotifChordConfig, Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    '''
    Write model weights plus their info into one `.safetensors` file.

    This is the writer matching `load_checkpoint`'s safetensors branch: the config is
    stored under the `'config'` metadata key as JSON, so `from_pretrained` recovers
    the hyper-parameters from the file instead of inferring them from tensor shapes.
    Optimizer / schedule / training state is never written - only weights and info.

    Either a model or a checkpoint path may be given; a path is read through
    `load_checkpoint`, which is what turns an old training checkpoint into a
    publishable weight file.

        save_safetensors('D:/MotifChord/pretrained_base.pt',
                         'D:/MotifChord/pretrained_base.safetensors')
        save_safetensors(model, 'model.safetensors', metadata={'license': 'apache-2.0'})

    Args:
        model_or_state (Any): An `nn.Module`, a `state_dict`, or a checkpoint path
            (`.pt` / `.safetensors` / a directory) to read the weights from. A config
            is taken from the path when it carries one.
        path (str): Destination `.safetensors` file; its directory must exist.
        config (Optional[Union[MotifChordConfig, Dict[str, Any]]]): Config to store;
            None uses the one of a model / of the checkpoint that was read.
        metadata (Optional[Dict[str, Any]]): Extra metadata entries; values other
            than strings are stored as JSON.

    Returns:
        Dict[str, Any]: What was written - `path`, `tensors`, `params`, `bytes`,
            `source`, `config` and `metadata` (the last one as actually stored).

    Raises:
        ImportError: If `safetensors` is not installed.
        FileNotFoundError: If `model_or_state` is a path that does not exist.
        ValueError: If no weights / no config could be obtained.
    '''
    if not HAS_SAFETENSORS:
        raise ImportError(
            'writing a safetensors checkpoint needs the `safetensors` package '
            '(pip install safetensors).'
        )

    source = 'state_dict'
    resolved = config if isinstance(config, MotifChordConfig) else None
    if isinstance(model_or_state, str) or hasattr(model_or_state, '__fspath__'):
        loaded = load_checkpoint(os.fspath(model_or_state))
        state_dict = loaded.state_dict
        source = loaded.source
        if resolved is None:
            resolved = loaded.config
    elif isinstance(model_or_state, nn.Module):
        state_dict = model_or_state.state_dict()
        source = type(model_or_state).__name__
        if resolved is None:
            resolved = getattr(model_or_state, 'config', None)
    elif isinstance(model_or_state, dict):
        state_dict = model_or_state
    else:
        raise ValueError(
            f'cannot save {type(model_or_state).__name__}: expected an nn.Module, a '
            f'state_dict or a checkpoint path.'
        )

    if not state_dict:
        raise ValueError('no weights to save.')
    if resolved is None:
        resolved = config_from_state_dict(state_dict)
    elif isinstance(config, dict):
        resolved = config_from_mapping(config)

    weights, dropped = _drop_tied_weights(state_dict)
    out: Dict[str, torch.Tensor] = {}
    for key, value in weights.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f'{key!r} is not a tensor ({type(value).__name__}).')
        out[key] = value.detach().to('cpu').contiguous()

    info: Dict[str, str] = {
        'format': 'pt',
        'producer': 'motif-checkpoint/1',
        'config': json.dumps(config_to_dict(resolved), ensure_ascii=False),
    }
    for key, value in (metadata or {}).items():
        info[str(key)] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'destination directory does not exist: {directory!r}')
    _safe_save_file(out, path, metadata=info)

    return {
        'path': path,
        'source': source,
        'tensors': len(out),
        'params': int(sum(value.numel() for value in out.values())),
        'bytes': os.path.getsize(path),
        'config': config_to_dict(resolved),
        'dropped': dropped,
        'metadata': info,
    }


def config_from_mapping(spec: Dict[str, Any]) -> MotifChordConfig:
    '''
    Build a `MotifChordConfig` from a plain mapping, translating common aliases.

    Unknown keys are ignored, so a HuggingFace-style `config.json` (which carries far
    more fields) does not make the load fail.

    Args:
        spec (Dict[str, Any]): Mapping of config fields / aliases.

    Returns:
        MotifChordConfig: The configuration; fields absent from `spec` keep their
            default.
    '''
    resolved: Dict[str, Any] = {}
    for key, value in spec.items():
        name = _CONFIG_ALIASES.get(key, key)
        if name in _CONFIG_FIELDS:
            resolved[name] = value
    if 'block' in resolved and isinstance(resolved['block'], list):
        resolved['block'] = tuple(resolved['block'])
    return MotifChordConfig(**resolved)


def _config_from_any(value: Any) -> Optional[MotifChordConfig]:
    '''
    Coerce whatever a checkpoint stored as "config" into a `MotifChordConfig`.

    Args:
        value (Any): A config mapping, a JSON string, or None.

    Returns:
        Optional[MotifChordConfig]: The config, or None when `value` carries nothing
            usable.
    '''
    if value is None:
        return None
    if isinstance(value, MotifChordConfig):
        return value
    if isinstance(value, (str, bytes)):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    if isinstance(value, dict):
        return config_from_mapping(value)
    return None


def _find_config(*containers: Any) -> Optional[MotifChordConfig]:
    '''
    Look for a config entry under any of `_CONFIG_KEYS` in each container.

    Args:
        *containers: Mappings (and None) to search, in priority order.

    Returns:
        Optional[MotifChordConfig]: The first config that could be parsed.
    '''
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in _CONFIG_KEYS:
            if key in container:
                config = _config_from_any(container[key])
                if config is not None:
                    return config
    return None


def config_from_state_dict(sd: Dict[str, torch.Tensor]) -> MotifChordConfig:
    '''
    Recover the hyper-parameters from the tensor shapes of a state_dict.

    Everything `MotifChordConfig` needs is determined by the weights:

    * `vocab_size` / `model_dim` from `embed.weight` (or `lm_head.weight`);
    * `block_n` from the highest `blocks.<n>.` index;
    * `block` - the attention recipe - by checking, for every `features.<i>` of block
      0, whether it carries the chord parameters (`theta_alpha`) or the gated
      delta-net ones (`q_conv` / `a_proj`);
    * `num_heads` from the per-head vectors (`theta_alpha` / `A_log` / `a_proj`);
    * `num_kv_heads` from `k_proj` against the head dimension (chord layers only);
    * `head_k_dim` from `q_proj` of the gated delta-net layers - a chord-only
      checkpoint leaves no trace of it (chord layers use `model_dim // num_heads`),
      so the default 64 is kept; nothing reads it in that case;
    * `tied_weight` from whether `embed.weight` and `lm_head.weight` are the same
      tensor.

    Args:
        sd (Dict[str, torch.Tensor]): Model weights, wrapper prefixes already
            stripped.

    Returns:
        MotifChordConfig: The inferred configuration.

    Raises:
        ValueError: If the state_dict has no recognisable embedding / block keys.
    '''
    embed = sd.get('embed.weight')
    if embed is None:
        embed = sd.get('lm_head.weight')
    if embed is None:
        raise ValueError(
            'cannot infer the config: the state_dict has neither '
            "'embed.weight' nor 'lm_head.weight'."
        )
    if embed.dim() != 2:
        raise ValueError(
            f"cannot infer the config: 'embed.weight' has shape {tuple(embed.shape)}, "
            f'expected [vocab_size, model_dim].'
        )
    vocab_size, model_dim = int(embed.shape[0]), int(embed.shape[1])

    block_ids = sorted({
        int(match.group(1))
        for key in sd
        if (match := re.match(r'blocks\.(\d+)\.', key))
    })
    if not block_ids:
        raise ValueError(
            "cannot infer the config: the state_dict has no 'blocks.<n>.' keys."
        )
    block_n = block_ids[-1] + 1

    feature_ids = sorted({
        int(match.group(1))
        for key in sd
        if (match := re.match(r'blocks\.0\.features\.(\d+)\.', key))
    })
    if not feature_ids:
        raise ValueError(
            "cannot infer the config: block 0 has no 'features.<n>.' keys."
        )

    recipe = []
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    head_k_dim: Optional[int] = None

    for index in feature_ids:
        prefix = f'blocks.0.features.{index}.attn.'
        if prefix + 'theta_alpha' in sd:
            recipe.append('chord')
            heads = int(sd[prefix + 'theta_alpha'].shape[0])
            num_heads = heads if num_heads is None else num_heads
            head_dim = model_dim // heads
            kv_weight = sd.get(prefix + 'k_proj.weight')
            if kv_weight is not None and head_dim > 0:
                num_kv_heads = int(kv_weight.shape[0]) // head_dim
        elif prefix + 'a_proj' in sd or prefix + 'q_conv.weight' in sd:
            recipe.append('gdn')
            anchor = sd.get(prefix + 'a_proj.weight', sd.get(prefix + 'A_log'))
            if anchor is not None:
                heads = int(anchor.shape[0])
                num_heads = heads if num_heads is None else num_heads
            q_weight = sd.get(prefix + 'q_proj.weight')
            if q_weight is not None and (num_heads or 0) > 0:
                head_k_dim = int(q_weight.shape[0]) // int(num_heads)
        else:
            raise ValueError(
                f'cannot infer the config: {prefix} carries neither chord '
                f"('theta_alpha') nor gated delta-net ('q_conv' / 'a_proj') weights."
            )

    if num_heads is None:
        raise ValueError('cannot infer the config: no per-head parameter was found.')
    if num_kv_heads is None:
        num_kv_heads = num_heads

    lm_head = sd.get('lm_head.weight')
    if lm_head is None:
        tied = True
    else:
        tied = embed is lm_head or (
            tuple(embed.shape) == tuple(lm_head.shape) and torch.equal(embed, lm_head)
        )

    return MotifChordConfig(
        vocab_size=vocab_size,
        model_dim=model_dim,
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        head_k_dim=int(head_k_dim) if head_k_dim is not None else 64,
        tied_weight=bool(tied),
        block=tuple(recipe),
        block_n=block_n,
    )


def _is_tensor_dict(value: Any) -> bool:
    '''
    Decide whether a value is already a usable (non-empty, all-tensor) state_dict.

    Args:
        value (Any): Candidate mapping.

    Returns:
        bool: True when `value` is a non-empty mapping whose values are all tensors.
    '''
    return (
        isinstance(value, dict)
        and len(value) > 0
        and all(isinstance(item, torch.Tensor) for item in value.values())
    )


def strip_wrapper_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    '''
    Remove `module.` / `_orig_mod.` / `model.` prefixes from every key.

    A prefix is only stripped while *all* keys carry it, so a legitimate submodule
    whose name happens to be `model` is never damaged.

    Args:
        sd (Dict[str, torch.Tensor]): Weights, possibly wrapper-prefixed.

    Returns:
        Dict[str, torch.Tensor]: The same tensors under unprefixed names.
    '''
    if not sd:
        return sd
    current = dict(sd)
    changed = True
    while changed and current:
        changed = False
        for prefix in _WRAPPER_PREFIXES:
            if all(key.startswith(prefix) for key in current):
                current = {key[len(prefix):]: value for key, value in current.items()}
                changed = True
                break
    return current


def _wrap(source: str, sd: Dict[str, torch.Tensor], config: Optional[MotifChordConfig],
          path: str, metadata: Optional[Dict[str, str]] = None) -> LoadedCheckpoint:
    '''
    Assemble a `LoadedCheckpoint` from the raw pieces of one format.

    Args:
        source (str): Format description.
        sd (Dict[str, torch.Tensor]): Raw state_dict.
        config (Optional[MotifChordConfig]): Config found in the file, if any.
        path (str): File that was read.
        metadata (Optional[Dict[str, str]]): Safetensors metadata.

    Returns:
        LoadedCheckpoint: The normalized checkpoint.
    '''
    return LoadedCheckpoint(
        state_dict=strip_wrapper_prefixes(sd),
        config=config,
        source=source,
        path=path,
        metadata=dict(metadata or {}),
    )


def _extract_torch(obj: Any, path: str) -> LoadedCheckpoint:
    '''
    Pull the model weights (and any config) out of a `torch.load` result.

    Args:
        obj (Any): Whatever `torch.load` returned.
        path (str): File that was read (for the source label).

    Returns:
        LoadedCheckpoint: The normalized checkpoint.

    Raises:
        ValueError: If no model weights can be found in `obj`.
    '''
    if isinstance(obj, nn.Module):
        return _wrap('nn.Module', obj.state_dict(), None, path)

    if not isinstance(obj, dict):
        raise ValueError(
            f'{path!r} holds {type(obj).__name__}, expected a state_dict / a '
            f'checkpoint mapping.'
        )

    payload = obj.get('payload')
    if isinstance(payload, dict):
        config = _find_config(obj, payload)
        model = payload.get('model')
        if isinstance(model, nn.Module):
            return _wrap('pipeline checkpoint', model.state_dict(), config, path)
        if _is_tensor_dict(model):
            pipeline_name = obj.get('__pipeline__') or 'pipeline'
            return _wrap(f'pipeline checkpoint ({pipeline_name!r})', model, config, path)

    for key in ('state_dict', 'model_state_dict', 'weights', 'model'):
        candidate = obj.get(key)
        if isinstance(candidate, nn.Module):
            return _wrap(f'{key}=nn.Module', candidate.state_dict(),
                         _find_config(obj), path)
        if _is_tensor_dict(candidate):
            return _wrap(f'state_dict under {key!r}', candidate, _find_config(obj), path)

    tensors = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    if tensors:
        return _wrap('bare state_dict', tensors, _find_config(obj), path)

    raise ValueError(
        f'cannot find model weights in {path!r}: the mapping has keys '
        f'{sorted(obj)[:8]}.'
    )


def _torch_load(path: str, map_location: Any) -> Any:
    '''
    `torch.load` that accepts both a bare state_dict and a pipeline checkpoint.

    `weights_only=True` is tried first (it refuses arbitrary pickles), with the
    pipeline classes of `_SAFE_GLOBALS` allowlisted so an old checkpoint still
    loads without the pipeline package installed; the full unpickler is the last
    resort. `mmap=True` is tried first as well so a multi-GB file is not read twice.

    Args:
        path (str): File to load.
        map_location (Any): Device / mapping passed to `torch.load`.

    Returns:
        Any: The unpickled object.

    Raises:
        RuntimeError: If every attempt fails, with the underlying errors attached.
    '''
    errors = []
    with torch.serialization.safe_globals(_SAFE_GLOBALS):
        for kwargs in (
            {'weights_only': True, 'mmap': True},
            {'weights_only': True},
            {'weights_only': False, 'mmap': True},
            {'weights_only': False},
        ):
            try:
                return torch.load(path, map_location=map_location, **kwargs)
            except Exception as exc:
                errors.append(f'{kwargs}: {type(exc).__name__}: {exc}')
    raise RuntimeError(
        f'failed to load {path!r} with torch.load:\n  ' + '\n  '.join(errors)
    )


def load_checkpoint(path: str, map_location: Any = 'cpu') -> LoadedCheckpoint:
    '''
    Read a checkpoint in any of the three supported layouts.

    The layouts are a bare `state_dict`, a training pipeline checkpoint (a mapping
    whose `payload` holds the model) and safetensors (a single file, a shard index or
    a directory of shards). Wrapper prefixes such as `module.`, `_orig_mod.` and
    `model.` are stripped from every key, so the weights come back under their
    canonical names. When the file carries no config, `LoadedCheckpoint.config` is
    None and the caller has to infer it from the tensor shapes with
    `config_from_state_dict`.

    Args:
        path (str): `.pt` / `.pth` file, `.safetensors` file, safetensors shard
            directory / `*.index.json`, or a directory holding one of those.
        map_location (Any): Where the weights are materialised (`'cpu'` by default).

    Returns:
        LoadedCheckpoint: Weights plus whatever config the file carried.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If the file cannot be interpreted.
        ImportError: If a safetensors file has to be read without `safetensors`
            installed.
    '''
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f'checkpoint not found: {path!r}')

    if os.path.isdir(path) or looks_like_safetensors(path):
        if not HAS_SAFETENSORS:
            raise ImportError(
                'reading a safetensors checkpoint needs the `safetensors` package '
                '(pip install safetensors).'
            )
        shards, sidecar = _safetensors_shards(path)
        sd: Dict[str, torch.Tensor] = {}
        metadata: Dict[str, str] = {}
        for shard in shards:
            sd.update(_safe_load_file(shard, device=str(map_location)))
            metadata.update(_safetensors_metadata(shard))
        config = _find_config(metadata)
        if config is None and sidecar is not None:
            with open(sidecar, 'r', encoding='utf-8') as handle:
                config = _config_from_any(json.load(handle))
        label = 'safetensors' if len(shards) == 1 else f'safetensors ({len(shards)} shards)'
        return _wrap(label, sd, config, path, metadata)

    return _extract_torch(_torch_load(path, map_location), path)
