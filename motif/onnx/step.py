import torch
import torch.nn as nn

from motif.model.cache import KVLayerCache, build_cache


def layers(model):
    for block in model.blocks:
        for layer in block.features:
            yield layer


def layer_plan(model):
    '''`'kv'` twice per chord layer, `'gdn'` four times per delta layer.'''
    kinds = []
    for layer in layers(model):
        if issubclass(layer.cache_type, KVLayerCache):
            kinds += ['kv', 'kv']
        else:
            kinds += ['gdn'] * 4
    return kinds


def new_cache(model):
    '''A wide functional cache: chord keys / values expanded to every head on write.'''
    return build_cache(model, use_attn=True, functional=True, wide_kv=True)


def flat_state(model, cache):
    '''The cache as a flat list of tensors, in `layer_plan` order.'''
    flats = []
    for block_cache in cache.blocks:
        for layer_cache in block_cache.attn:
            if isinstance(layer_cache, KVLayerCache):
                flats += [layer_cache.k, layer_cache.v]
            else:
                flats += [layer_cache.state, *layer_cache.conv_state]
    return flats


def rebuild_cache(model, flats):
    '''Inverse of `flat_state`: caches whose key / value tensors are the given ones.'''
    cache = new_cache(model)
    flats = iter(flats)
    for block_cache in cache.blocks:
        for layer_cache in block_cache.attn:
            if isinstance(layer_cache, KVLayerCache):
                layer_cache.k = next(flats)
                layer_cache.v = next(flats)
                layer_cache.length = layer_cache.k.shape[2]
                layer_cache._owned = True
            else:
                layer_cache.state = next(flats)
                layer_cache.conv_state = (next(flats), next(flats), next(flats))
    return cache


def zero_flats(model, template):
    '''
    The state of a model that has seen nothing, shaped and typed like `template`.

    `template` is any `flat_state` list; the key / value entries shrink to an
    empty sequence and everything else is zeroed, so the dtypes follow whatever
    the model itself produces (a half model keeps its delta state in float32).
    '''
    flats = []
    for kind, tensor in zip(layer_plan(model), template):
        if kind == 'kv':
            flats.append(tensor.new_zeros(1, tensor.shape[1], 0, tensor.shape[3]))
        else:
            flats.append(torch.zeros_like(tensor))
    return flats


def dynamic_specs(model, kv_dim=2):
    '''`torch.export` specs: only the cached sequence length of the chord layers.'''
    return tuple({kv_dim: torch.export.Dim('S')} if kind == 'kv' else None
                 for kind in layer_plan(model))


class ModelStep(nn.Module):
    '''`(input_ids[1, T], position[1], caches) -> (logits, *caches)`.'''

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.kinds = layer_plan(model)

    def forward(self, input_ids, position, caches):
        cache = rebuild_cache(self.model, caches)
        out = self.model(input_ids, cache=cache, position=position)
        return (out.logits, *flat_state(self.model, out.cache))
