import torch
import torch.nn as nn

from motif.ops import SwiGLU as SwiGLUOp


class SwiGLU(nn.Module):
    '''
    Gated feed-forward block of a decoder layer, `down_proj(SwiGLUOp(gate_up_proj(x)))`.

    The gated width is fixed to two thirds of the input width rounded up to a
    multiple of 128, which is the layout the fused `motif.ops.SwiGLU` kernel expects.

    Attributes:
        gate_up_proj (nn.Linear): Fused gate and up projection, `[D, 2 * hidden]`.
        down_proj (nn.Linear): Down projection back to the model width.
        act (SwiGLUOp): The fused activation, stateless but held once so that a
            forward pass does not rebuild it on every call.
        hidden_features (int): Gated width, i.e. half of `gate_up_proj`'s output.
    '''

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        bias: bool = False,
    ):
        '''
        Build the projections, allocating the default gated width when asked to.

        Args:
            in_features (int): Dimension of the input features.
            hidden_features (Optional[int]): Gated width; None computes
                `ceil(8 * in_features / 3)` rounded up to a multiple of 128.
            out_features (Optional[int]): Output dimension; None mirrors
                `in_features`.
            bias (bool): Whether the projections carry a bias. Defaults to False.
        '''
        super().__init__()

        if hidden_features is None:
            h = int(in_features * 8 / 3)
            hidden_features = (h + 127) // 128 * 128

        out_features = out_features or in_features
        self.hidden_features = hidden_features

        self.gate_up_proj = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.down_proj = nn.Linear(hidden_features, out_features, bias=bias)
        self.act = SwiGLUOp(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Apply the gated feed-forward block.

        Args:
            x (torch.Tensor): `[..., D]` input features.

        Returns:
            torch.Tensor: `[..., D]` output features.
        '''
        return self.down_proj(self.act(self.gate_up_proj(x)))
