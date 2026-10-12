import numpy as np


def softmax(logits):
    '''A stable softmax; an all-masked row degenerates to a uniform one.'''
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    total = weights.sum()
    if not np.isfinite(total) or total == 0.0:
        return np.full(logits.shape, 1.0 / logits.size)
    return weights / total


class NumpySampler:
    '''Temperature, top-k, top-p and a repetition penalty over token ids.'''

    def __init__(self, temperature=0.7, top_k=None, top_p=None,
                 repetition_penalty=1.15, seed=None):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.rng = np.random.default_rng(seed)

    def __repr__(self):
        return (f'NumpySampler(temperature={self.temperature}, top_k={self.top_k}, '
                f'top_p={self.top_p}, repetition_penalty={self.repetition_penalty})')

    def probabilities(self, logits, seen=None):
        '''The final token distribution, in the order of the torch sampler.'''
        logits = np.array(logits, dtype=np.float64, copy=True)

        if self.repetition_penalty != 1.0 and seen is not None and len(seen):
            unique = np.unique(np.asarray(seen))
            values = logits[unique]
            logits[unique] = np.where(values > 0,
                                      values / self.repetition_penalty,
                                      values * self.repetition_penalty)

        if self.temperature != 1.0:
            logits = logits / max(self.temperature, 1e-5)

        if self.top_k is not None and self.top_k > 0:
            keep = min(int(self.top_k), logits.shape[-1])
            threshold = np.partition(logits, -keep)[-keep]
            logits = np.where(logits < threshold, -np.inf, logits)

        if self.top_p is not None and 0.0 < self.top_p < 1.0:
            order = np.argsort(-logits)
            cumulative = np.cumsum(softmax(logits[order]))
            drop = cumulative > self.top_p
            drop[1:] = drop[:-1].copy()
            drop[0] = False
            removed = np.zeros(logits.shape, dtype=bool)
            removed[order] = drop
            logits = np.where(removed, -np.inf, logits)

        return softmax(logits)

    def __call__(self, logits, seen=None):
        probs = self.probabilities(logits, seen)
        return int(self.rng.choice(probs.shape[-1], p=probs))
