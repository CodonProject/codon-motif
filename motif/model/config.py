from typing import Optional, Literal, Tuple
from dataclasses import dataclass


@dataclass
class MotifChordConfig:
    '''
    Hyper-parameters of `motif.model.motif.MotifChordModel`.

    A plain data holder: it carries the vocabulary, the width of the residual
    stream, the attention head layout and the per-block attention recipe. It has
    no behaviour, so it can be built, copied and pickled by the pipelines.

    Attributes:
        vocab_size (int): Size of the tokenizer vocabulary; the bundled tokenizer
            trains `5 * 2**13 = 40960` tokens.
        model_dim (int): Width of the residual stream and of every sub-layer.
        num_heads (int): Number of attention query heads.
        num_kv_heads (Optional[int]): Number of key / value heads for
            grouped-query attention; None means one kv head per query head.
        head_k_dim (Optional[int]): Per-head key dimension used by the gated
            delta-net attention layers.
        tied_weight (bool): Tie the token embedding matrix to the output head
            instead of allocating a separate `lm_head` weight.
        block (Tuple[Literal['chord', 'gdn'], ...]): Attention type of each
            `Decoder` inside one `DecoderBlock`; 'chord' selects `ChordAttention`
            and 'gdn' selects `GatedDeltaAttention`.
        block_n (int): Number of `DecoderBlock`s stacked in the model.
    '''

    vocab_size: int = 5*(2**13)
    model_dim:  int = 768
    num_heads:  int = 12
    num_kv_heads: Optional[int] = 3
    head_k_dim:   Optional[int] = 64
    tied_weight: bool = True

    block: Tuple[Literal['chord', 'gdn']] = ('gdn', 'gdn', 'gdn', 'chord')
    block_n: int = 4
