import copy
from typing import Optional, Protocol

import torch
import torch.nn.functional as F


class Constraint(Protocol):
    '''
    Protocol implemented by `motif.model.grammar.JSONConstraint` (and by any other
    constrained-decoding backend).

    Attributes:
        eos_token_id (int): Token the sampler has to emit once the constrained
            output is complete.
    '''

    eos_token_id: int

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        '''
        Drop every grammar-forbidden token from `logits`.

        Args:
            logits (torch.Tensor): `[batch_size, vocab_size]` scores.

        Returns:
            torch.Tensor: The same scores with every grammar-forbidden token set to
                -inf.
        '''
        ...

    def apply_forcing(self, tokens: torch.Tensor) -> torch.Tensor:
        '''
        Emit the forced end token on the rows whose constrained value is complete.

        Args:
            tokens (torch.Tensor): `[batch_size, 1]` sampled tokens.

        Returns:
            torch.Tensor: The tokens, with the forced end token written on the rows
                whose constrained value is already complete.
        '''
        ...

    def advance(self, tokens: torch.Tensor) -> None:
        '''
        Advance the per-row grammar state with the tokens just sampled.

        Args:
            tokens (torch.Tensor): `[batch_size, 1]` tokens just sampled.

        Returns:
            None: The per-row grammar state is advanced in place.
        '''
        ...


class Sampler:
    '''
    Sampler: repetition penalty + temperature + top-k / top-p truncation, plus an
    optional grammar constraint.

    Attributes:
        temperature (float): Sampling temperature.
        top_k (Optional[int]): top-k truncation.
        top_p (Optional[float]): nucleus truncation.
        repetition_penalty (float): Repetition penalty factor; 1.0 disables it.
        constraint (Optional[Constraint]): Grammar constraint; None means free
            sampling.
    '''

    def __init__(
        self,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.15,
        constraint: Optional[Constraint] = None,
    ) -> None:
        '''
        Store the sampling configuration.

        Args:
            temperature (float): Sampling temperature.
            top_k (Optional[int]): Keep only the `top_k` most likely tokens.
            top_p (Optional[float]): Keep the smallest set of tokens whose
                cumulative probability reaches `top_p`.
            repetition_penalty (float): Penalty applied to tokens already present in
                the sequence; 1.0 disables it.
            constraint (Optional[Constraint]): Grammar constraint to apply.
        '''
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.constraint = constraint

    @property
    def eos_token_id(self) -> Optional[int]:
        '''
        The end token the current constraint requires (None when there is no
        constraint, or when it did not configure one).

        `MotifChordModel.generate` uses it as the default `eos_token_id`, so a bound
        grammar constraint does not have to be passed in again by hand.

        Returns:
            Optional[int]: The constraint's forced end token, if any.
        '''
        return getattr(self.constraint, 'eos_token_id', None)

    def __repr__(self) -> str:
        '''
        Render the sampling settings as a short debug string.

        Returns:
            str: A short debug representation of the sampling settings.
        '''
        return (
            f'Sampler(temperature={self.temperature}, top_k={self.top_k}, '
            f'top_p={self.top_p}, repetition_penalty={self.repetition_penalty}, '
            f'constraint={type(self.constraint).__name__ if self.constraint else None})'
        )

    def with_constraint(
        self,
        constraint: Optional[Constraint],
        eos_token_id: Optional[int] = None,
    ) -> 'Sampler':
        '''
        Return a copy of this sampler bound to a grammar constraint (self is left
        untouched).

        Args:
            constraint (Optional[Constraint]): Constraint object; None unbinds.
            eos_token_id (Optional[int]): Override the forced end token of the
                constraint.

        Returns:
            Sampler: The new sampler carrying the constraint.
        '''
        bound = copy.copy(self)
        bound.constraint = constraint
        if constraint is not None and eos_token_id is not None:
            constraint.eos_token_id = int(eos_token_id)
        return bound

    @torch.no_grad()
    def __call__(self, logits: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        Sample one token per batch row.

        The order of operations is: repetition penalty -> temperature -> grammar
        constraint -> top-k -> top-p -> multinomial, and the constraint is then asked
        to force its end token and advanced.

        Args:
            logits (torch.Tensor): `[batch_size, vocab_size]` unnormalised scores.
            input_ids (Optional[torch.Tensor]): `[batch_size, seq_len]` token ids
                used by the repetition penalty only; None skips that step.

        Returns:
            torch.Tensor: `[batch_size, 1]` sampled token ids.
        '''
        if self.repetition_penalty != 1.0 and input_ids is not None:
            logits = logits.clone()
            for i in range(logits.shape[0]):
                seen = torch.unique(input_ids[i])
                vals = logits[i, seen]
                logits[i, seen] = torch.where(
                    vals > 0,
                    vals / self.repetition_penalty,
                    vals * self.repetition_penalty,
                )

        if self.temperature != 1.0:
            logits = logits / max(self.temperature, 1e-5)

        if self.constraint is not None:
            logits = self.constraint.mask_logits(logits)

        if self.top_k is not None and self.top_k > 0:
            top_k = min(self.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        if self.top_p is not None and 0.0 < self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if self.constraint is not None:
            next_token = self.constraint.apply_forcing(next_token)
            self.constraint.advance(next_token)

        return next_token


def make_sampler(sampler: Optional[Sampler] = None, **kwargs) -> Optional[Sampler]:
    '''
    Normalise the two calling styles "bind a ready instance" and "build from
    parameters"; shared by `set_sampler`.

        make_sampler()                            # -> None (unbind, back to greedy)
        make_sampler(Sampler(temperature=0.7))    # -> that instance
        make_sampler(temperature=0.7, top_p=0.9)  # -> Sampler(...)

    Args:
        sampler (Optional[Sampler]): A ready sampler instance.
        **kwargs: Constructor arguments for `Sampler`.

    Returns:
        Optional[Sampler]: The sampler to use, or None for greedy decoding.

    Raises:
        ValueError: If a ready instance and constructor arguments are given
            together.
    '''
    if sampler is not None and kwargs:
        raise ValueError('sampler and Sampler(...) constructor arguments are mutually exclusive')
    return Sampler(**kwargs) if kwargs else sampler
