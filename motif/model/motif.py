import builtins
import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from motif.block.decoder import DecoderBlock
from motif.model.cache import ModelCache
from motif.model.checkpoint import (
    LoadedCheckpoint,
    config_from_mapping,
    config_from_state_dict,
    load_checkpoint,
)
from motif.model.config import MotifChordConfig
from motif.model.generate import EosTokenId, Generation
from motif.model.sampler import Sampler, make_sampler
from motif.ops import RMSNorm
from motif.utils.units import count_params, format_count


@dataclass
class ModelOutput:
    '''
    Result of one forward pass.

    Attributes:
        hidden_states (torch.Tensor): `[B, T, D]` final normalised hidden states.
        logits (Optional[torch.Tensor]): `[B, T, vocab_size]` scores; None when the
            caller did not ask for them.
        loss (Optional[torch.Tensor]): Cross-entropy over `labels` when labels were
            given, otherwise None.
        cache (Optional[ModelCache]): The cache used and updated by this forward.
        sampler (Optional[Sampler]): The model-level sampler copied onto this
            output, when one is configured.
    '''

    hidden_states: torch.Tensor
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    cache: Optional[ModelCache] = None
    sampler: Optional[Sampler] = None

    @property
    def last_hidden_state(self) -> torch.Tensor:
        '''
        Returns:
            torch.Tensor: `[B, D]` hidden state of the last position.
        '''
        return self.hidden_states[:, -1, :]

    @property
    def last_logits(self) -> Optional[torch.Tensor]:
        '''
        Returns:
            Optional[torch.Tensor]: `[B, vocab_size]` logits of the last position,
            or None when this forward produced no logits.
        '''
        if self.logits is None:
            return None
        return self.logits[:, -1, :]

    def next_token(
        self,
        temperature: float = 1.0,
        sampler: Optional[Sampler] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        '''
        Take the next token at the last position (`[B, 1]`).

        Args:
            temperature (float): Scaling used when no sampler is passed; argmax is
                insensitive to a positive scale, so under greedy decoding this
                argument is only a placeholder.
            sampler (Optional[Sampler]): `motif.model.sampler.Sampler` running
                temperature + top-k / top-p + repetition penalty + grammar
                constraint; passing it replaces the argmax. An explicitly passed
                sampler wins over the one bound through `set_sampler`.
            input_ids (Optional[torch.Tensor]): `[B, T]`, used by the sampler for
                the repetition penalty; None skips that penalty.

        Returns:
            Optional[torch.Tensor]: `[B, 1]` token ids, or None when `logits` is
            None.
        '''
        if self.logits is None:
            return None
        last = self.logits[:, -1, :]
        active = sampler if sampler is not None else self.sampler
        if active is not None:
            return active(last, input_ids)
        return (last / temperature).argmax(dim=-1, keepdim=True)

    def set_sampler(self, sampler: Optional[Sampler] = None, **kwargs) -> 'ModelOutput':
        '''
        Bind a sampler to this output and return the output for chaining; calling it
        with no arguments at all unbinds (back to greedy decoding).

            out.set_sampler(temperature=0.7, top_p=0.9)     # build from parameters
            out.set_sampler(Sampler(temperature=0.7))       # bind a ready instance
            out.set_sampler()                               # unbind

        Once bound, `next_token()` uses it by default; an explicit
        `next_token(sampler=...)` still wins.

        To make a sampler stick across steps use `MotifChordModel.set_sampler` - a
        `ModelOutput` is rebuilt on every forward, so setting it here only affects
        this one result.

        Note: the repetition penalty needs the sampling history, and a `ModelOutput`
        only knows the input of its own forward, so callers that want the penalty
        must pass the sequence themselves via `next_token(input_ids=...)`.

        Args:
            sampler (Optional[Sampler]): Ready sampler instance to bind.
            **kwargs: Constructor arguments for `Sampler`, used to build one.

        Returns:
            ModelOutput: This output, with the sampler attached.
        '''
        self.sampler = make_sampler(sampler, **kwargs)
        return self

    def reset_caches(self):
        '''
        Reset the cache held by this output, if it carries one.

        Returns:
            None: The cache, when present, is cleared in place.
        '''
        if self.cache is not None:
            self.cache.reset()

    def __repr__(self):
        '''
        Returns:
            str: A short debug representation with the tensor shapes and flags.
        '''
        hs_shape = tuple(self.hidden_states.shape)
        logits_shape = tuple(self.logits.shape) if self.logits is not None else None
        return (
            f"ModelOutput(hidden_states={hs_shape}, logits={logits_shape}, "
            f"loss={self.loss is not None}, cache={self.cache!r}, "
            f"sampler={self.sampler!r})"
        )


class MotifChordModel(nn.Module):
    '''
    The MotifChord decoder-only language model.

    Attributes:
        config (MotifChordConfig): Hyper-parameters this model was built from.
        embed (nn.Embedding): Token embedding.
        blocks (nn.ModuleList): `config.block_n` `DecoderBlock`s.
        norm (RMSNorm): Final normalisation before the head.
        lm_head (nn.Linear): Language-model head, built without bias.
        sampler (Optional[Sampler]): Model-level sampler attached to every forward
            output; None means greedy decoding unless a call overrides it.
    '''

    def __init__(self, config: MotifChordConfig):
        '''
        Build the model.

        Args:
            config (MotifChordConfig): Hyper-parameters; when `config.tied_weight`
                is set, the embedding weight and the head weight become the same
                tensor.
        '''
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.model_dim)
        self.blocks = nn.ModuleList([
            DecoderBlock(config) for _ in range(config.block_n)
        ])
        self.norm = RMSNorm(config.model_dim)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)

        if config.tied_weight:
            self.embed.weight = self.lm_head.weight

        self.sampler: Optional[Sampler] = None


    @classmethod
    def _resolve_pretrained_config(
        cls,
        config: Optional[Union[MotifChordConfig, Dict, str]],
        loaded: LoadedCheckpoint,
    ) -> MotifChordConfig:
        '''
        Decide which hyper-parameters a checkpoint is built with.

        Precedence: the explicit `config` argument, then whatever the checkpoint
        itself carried (its `payload`, a config entry, or a sidecar `config.json`),
        then the shapes of the weights, then the default config.

        Args:
            config (Optional[Union[MotifChordConfig, Dict, str]]): The caller's
                config: a `MotifChordConfig`, a mapping of fields, or a JSON path.
            loaded (LoadedCheckpoint): The checkpoint, weights already resolved.

        Returns:
            MotifChordConfig: The config to build the model from.

        Raises:
            TypeError: If `config` is none of the accepted shapes.
            FileNotFoundError: If `config` is a path that does not exist.
        '''
        if config is not None:
            if isinstance(config, MotifChordConfig):
                return config
            if isinstance(config, dict):
                return config_from_mapping(config)
            if isinstance(config, (str, bytes, os.PathLike)):
                with open(os.fspath(config), 'r', encoding='utf-8') as handle:
                    return config_from_mapping(json.load(handle))
            raise TypeError(
                'config must be a MotifChordConfig, a mapping of its fields, or a '
                f'path to a JSON file, got {type(config).__name__}.'
            )
        if loaded.config is not None:
            return loaded.config
        try:
            return config_from_state_dict(loaded.state_dict)
        except ValueError:
            return MotifChordConfig()

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: Optional[Union[MotifChordConfig, Dict, str]] = None,
        map_location: Union[str, torch.device] = 'cpu',
        device: Optional[Union[str, torch.device]] = None,
        strict: bool = False,
        verbose: bool = True,
        **overrides,
    ) -> 'MotifChordModel':
        '''
        Load a model from a checkpoint, auto-detecting the file layout.

        Three layouts are understood:

        * a **bare state_dict** - `torch.save(model.state_dict(), path)`, what
          `motif/pipeline/snapshot.py` writes (`step_000012345.pt`);
        * a **pipeline checkpoint** - what `BasicPipeline.save_checkpoint` writes
          (`last.pt`), whose `payload['model']` is the state_dict;
        * **safetensors** - a `.safetensors` file, a shard directory or an
          `*.index.json`, optionally with a `config.json` or a metadata config.

        None of these has to carry the hyper-parameters, so when no config can be
        found in the file the `MotifChordConfig` is inferred from the tensor shapes,
        using the recipe markers of the blocks (`theta_alpha` -> chord, `q_conv` /
        `a_proj` -> gated delta net) to recover `block` / `block_n`.

        The `config` argument wins over anything found in the file, and `**overrides`
        (`model_dim=...`, `block=...`, ...) is applied on top of the resolved config,
        which is how an inference mistake gets corrected:

            model = MotifChordModel.from_pretrained('D:/last.pt')
            model = MotifChordModel.from_pretrained('ckpt/last.pt', block_n=8,
                                                    block=('gdn', 'chord'))

        Args:
            path (str): Checkpoint file or directory (`.pt`, `.pth`, `.safetensors`,
                `*.index.json`, or a directory holding one of those).
            config (Optional[Union[MotifChordConfig, Dict, str]]): Explicit config,
                a mapping of config fields, or a path to a JSON config file; None
                means "take it from the checkpoint, else infer it from the weights".
            map_location (Union[str, torch.device]): Where the weights are
                materialised while loading. `'cpu'` by default, so a multi-GB
                checkpoint never lands on the GPU by accident.
            device (Optional[Union[str, torch.device]]): Move the finished model
                here; None leaves it on `map_location`.
            strict (bool): Fail when the checkpoint does not match the model exactly.
                False by default - a snapshot of a tied model often stores only
                `embed.weight` - and the mismatch is reported instead.
            verbose (bool): Print the detected format, the tensor count and any
                missing / unexpected keys.
            **overrides: `MotifChordConfig` fields overriding the resolved config.

        Returns:
            MotifChordModel: The model, in `eval()` mode, with its weights loaded.

        Raises:
            FileNotFoundError: If `path` does not exist.
            TypeError: If `config` / an override name is invalid.
            ValueError: If the checkpoint cannot be interpreted.
            RuntimeError: If `strict` is set and the weights do not line up, or if
                a shape does not match the resolved config.
        '''
        loaded = load_checkpoint(path, map_location=map_location)
        state_dict = loaded.state_dict

        resolved = cls._resolve_pretrained_config(config, loaded)
        if overrides:
            known = {field.name for field in dataclasses.fields(MotifChordConfig)}
            unknown = sorted(set(overrides) - known)
            if unknown:
                raise TypeError(
                    f'unknown MotifChordConfig override(s): {unknown}; '
                    f'valid names are {sorted(known)}.'
                )
            resolved = dataclasses.replace(resolved, **overrides)

        model = cls(resolved)

        embed_weight = state_dict.get('embed.weight')
        head_weight = state_dict.get('lm_head.weight')
        shared_note = None
        if resolved.tied_weight:
            if head_weight is None and embed_weight is not None:
                state_dict['lm_head.weight'] = embed_weight
                shared_note = "lm_head.weight taken from embed.weight (tied weights)"
            elif (
                head_weight is not None
                and embed_weight is not None
                and head_weight is not embed_weight
                and not torch.equal(head_weight, embed_weight)
            ):
                shared_note = (
                    'the checkpoint is untied but the config has tied_weight=True: '
                    'lm_head.weight and embed.weight disagree, the last one loaded '
                    'wins.'
                )

        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                f'{os.fspath(path)!r} does not match the model built from '
                f'{resolved!r}: {exc}'
            ) from exc

        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if strict and (missing or unexpected):
            raise RuntimeError(
                f'{os.fspath(path)!r} does not match the model exactly: '
                f'{len(missing)} missing key(s) {missing[:8]}, '
                f'{len(unexpected)} unexpected key(s) {unexpected[:8]}.'
            )

        if device is not None:
            model = model.to(device)
        model.eval()

        if verbose:
            total = model.count_param()
            builtins.print(
                f'[MotifChordModel] {os.fspath(path)} -> {loaded.source}: '
                f'{len(state_dict):,} tensors, {format_count(total)} params'
            )
            builtins.print(f'  config: {resolved}')
            if shared_note:
                builtins.print(f'  note: {shared_note}')
            if missing:
                builtins.print(
                    f'  missing ({len(missing)}): {missing[:8]}'
                    + (' ...' if len(missing) > 8 else '')
                )
            if unexpected:
                builtins.print(
                    f'  unexpected ({len(unexpected)}): {unexpected[:8]}'
                    + (' ...' if len(unexpected) > 8 else '')
                )
        return model

    def count_param(self, print: bool = False) -> int:
        '''
        Parameter count of the model (int, including the `requires_grad=False` ones
        such as `theta_mu`).

        With `print=True` a human-readable summary line (K / M / B / T) is printed
        as well:

            total = model.count_param()
            model.count_param(print=True)      # MotifChordModel params: 242.700 M (242,700,000)

        Note the parameter is really called `print` (named after the caller's habit),
        which is why the body uses `builtins.print`.

        Args:
            print (bool): Also print the summary line.

        Returns:
            int: The total number of parameters.
        '''
        total = count_params(self)
        if print:
            builtins.print(f'{type(self).__name__} params: {format_count(total)} ({total:,})')
        return total

    def cache_types(self) -> List[List[type]]:
        '''
        Returns:
            List[List[type]]: The layer cache type of every decoder inside every
            block.
        '''
        return [
            [decoder.cache_type for decoder in block.features]
            for block in self.blocks
        ]

    def create_cache(self, use_attn: Optional[bool] = None) -> ModelCache:
        '''
        Build the cache of the whole model.

        When `use_attn` is None it follows `not self.training`:
          * outside training mode the attn-res cache of every `DecoderBlockCache`
            and every per-layer attention cache (`KVLayerCache` /
            `GatedDeltaAttentionLayerCache`) are really instantiated, ready for
            decode / chunked prefill;
          * in training mode only empty `DecoderBlockCache`s are created, because the
            layer caches detach internally and enabling them would break the
            gradient flow.

        Args:
            use_attn (Optional[bool]): Force the per-layer attention caches on or
                off; None derives it from the training flag.

        Returns:
            ModelCache: The (possibly empty) cache for this model.
        '''
        if use_attn is None:
            use_attn = not self.training
        types_per_block = self.cache_types() if use_attn else [[] for _ in self.blocks]
        return ModelCache.create_for(types_per_block)

    def set_sampler(self, sampler: Optional[Sampler] = None, **kwargs) -> 'MotifChordModel':
        '''
        Set the model-level sampler; **once is enough**. Every `ModelOutput` returned
        by later forwards carries it, so `out.next_token()` uses it without having to
        be reconfigured at each step (a `ModelOutput` is rebuilt on every forward).

            model.set_sampler(temperature=0.7, top_p=0.9)
            out = model(ids)
            tok = out.next_token()                  # goes through that sampler
            out = model(tok, cache=out.cache)
            tok = out.next_token()                  # still the same one

        Calling it with no arguments unbinds (back to greedy decoding). A one-off
        override uses `out.set_sampler(...)` or `out.next_token(sampler=...)` and
        leaves the model-level setting alone.

        Args:
            sampler (Optional[Sampler]): Ready sampler instance to bind.
            **kwargs: Constructor arguments for `Sampler`, used to build one.

        Returns:
            MotifChordModel: This model, with the sampler attached.
        '''
        self.sampler = make_sampler(sampler, **kwargs)
        return self

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        eos_token_id: EosTokenId = None,
        sampler: Optional[Sampler] = None,
        cache: Optional[ModelCache] = None,
        mask: Optional[torch.Tensor] = None,
        tokenizer=None,
    ) -> Generation:
        '''
        Return a `Generation`: iterating it runs the generation loop and yields one
        `[B, 1]` token per step.

            gen = model.generate(prompt, max_new_tokens=64, eos_token_id=im_end)
            for token in gen:                     # streaming: use tokens as they come
                ...
            nxt = model.generate(gen.sequences, cache=gen.cache)     # continue

            gen = model.generate(prompt, max_new_tokens=64).run()     # or run it all
            print(gen.sequences, gen.generated)

        Args:
            input_ids (torch.Tensor): `[B, T]` prompt.
            max_new_tokens (int): Maximum number of steps to generate.
            eos_token_id (EosTokenId): Stop a row as soon as one of these is
                sampled; the whole batch stops early once every row is done. Each entry
                may be an id or a token name the `tokenizer` resolves. None takes the
                end token of the current sampler's grammar constraint
                (`Sampler.eos_token_id`), then the tokenizer's own eos, and when there
                is none either, generation never stops early.
            sampler (Optional[Sampler]): One-off sampler override; None uses the
                sampler set through `set_sampler`, and greedy decoding when that is
                unset as well.
            cache (Optional[ModelCache]): Cache left by a previous generation, used
                to continue. With it, `input_ids` can be the full sequence - the
                first `cache.seq_len` positions are cut off automatically.
            mask (Optional[torch.Tensor]): Keep-mask over the key axis, forwarded to
                every step (`Generation` documents how a cached prefix is cut off).
            tokenizer: Tokenizer that resolves a name given as `eos_token_id`, and
                whose `token_eos` is used when no eos was spelled out anywhere.

        Returns:
            Generation: The generator, also holding sequences / generated / cache /
            steps / finished.
        '''
        active = sampler if sampler is not None else self.sampler
        if eos_token_id is None and active is not None:
            eos_token_id = active.eos_token_id
        if eos_token_id is None and tokenizer is not None:
            eos_token_id = getattr(tokenizer, 'token_eos', None)
        return Generation(
            self,
            input_ids,
            max_new_tokens,
            eos_token_id,
            sampler,
            cache,
            mask,
            tokenizer,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Optional[ModelCache] = None,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        position: Optional[Union[int, torch.Tensor]] = None,
    ) -> ModelOutput:
        '''
        Run the model over `input_ids`.

        Args:
            input_ids (torch.Tensor): `[B, T]` token ids.
            cache (Optional[ModelCache]): Cache to continue from; one is created
                automatically when None (equivalent to `self.create_cache()`):
                in training mode it carries no per-layer caches, so a single forward
                behaves exactly as without caching, while in eval mode every layer
                cache is instantiated and the returned cache can be used for
                decoding. Passing an existing cache means "continue" (decode a new
                token, or prefill the next chunk) and the input is appended to the
                cached sequence.
            labels (Optional[torch.Tensor]): `[B, T]` targets; positions equal to
                -100 are ignored by the loss.
            mask (Optional[torch.Tensor]): Keep-mask over the key axis, handed to
                every chord attention layer as `attention_mask`. `[B, T]` with `True`
                on the positions that may be attended is the usual shape; an additive
                float mask of the same shape works too (0 where a key is allowed,
                `-inf` where it is not). Both are broadcast over the heads before they
                reach the attention layers. When `cache` already holds positions the
                mask has to cover them as well, i.e. `cached + T` columns. The
                recurrent layers keep their own state and do not read the mask, so a
                padded batch is only fully isolated once it reaches the attention
                layers that do.
            position (Optional[int | torch.Tensor]): Absolute position of the first
                token, handed to the attention layers that track positions. Defaults
                to the position the cache reports. Pass a scalar tensor to decode
                without baking the position into the graph.

        Returns:
            ModelOutput: Hidden states, logits, the optional loss, the cache and the
            model-level sampler.

        Raises:
            ValueError: If `mask` is not 2-dimensional or does not match the key
                width (`cached + T`) of `input_ids`.
        '''
        cached = cache.seq_len if cache is not None else 0
        if mask is not None:
            width = cached + input_ids.shape[1]
            if mask.dim() != 2:
                raise ValueError(f'mask must be [B, T], got {tuple(mask.shape)}')
            if mask.shape != (input_ids.shape[0], width):
                raise ValueError(
                    f'mask {tuple(mask.shape)} does not cover the '
                    f'{width} keys of a {cached}-position cache plus '
                    f'{input_ids.shape[1]} inputs'
                )
            mask = mask[:, None, None, :]

        x = self.embed(input_ids)

        if cache is None:
            cache = self.create_cache()

        for i, block in enumerate(self.blocks):
            out = block(x, cache[i], mask, 0 if position is None else position)
            x = out.hidden_states

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return ModelOutput(
            hidden_states=x,
            logits=logits,
            loss=loss,
            cache=cache,
            sampler=self.sampler,
        )
