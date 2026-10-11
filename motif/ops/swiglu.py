import torch
import torch.nn as nn
import torch.nn.functional as F

from .autocast import autocast_safe

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_H': 128}, num_warps=4),
            triton.Config({'BLOCK_H': 256}, num_warps=4),
            triton.Config({'BLOCK_H': 256}, num_warps=8),
            triton.Config({'BLOCK_H': 512}, num_warps=8),
            triton.Config({'BLOCK_H': 1024}, num_warps=8),
        ],
        key=['H'],
    )
    @triton.jit
    def _swiglu_fwd_kernel(
        x_ptr, y_ptr,
        H,
        BLOCK_H: tl.constexpr,
    ):
        '''
        Forward pass, one program per (row, column block).

        Args:
            x_ptr: Input rows of `2 * H` columns, values first and gate second.
            y_ptr: Output rows of `H` columns.
            H: Number of value (and gate) columns.
            BLOCK_H (tl.constexpr): Power-of-two block size over the columns.
        '''
        row = tl.program_id(0)
        col_blk = tl.program_id(1)
        cols = col_blk * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = cols < H

        x_row = x_ptr + row * 2 * H
        y_row = y_ptr + row * H

        out  = tl.load(x_row + cols,     mask=mask, other=0.0)
        gate = tl.load(x_row + H + cols, mask=mask, other=0.0)

        out_f  = out.to(tl.float32)
        gate_f = gate.to(tl.float32)

        sig = tl.sigmoid(gate_f)
        y = out_f * (gate_f * sig)

        tl.store(y_row + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_H': 128}, num_warps=4),
            triton.Config({'BLOCK_H': 256}, num_warps=4),
            triton.Config({'BLOCK_H': 256}, num_warps=8),
            triton.Config({'BLOCK_H': 512}, num_warps=8),
            triton.Config({'BLOCK_H': 1024}, num_warps=8),
        ],
        key=['H'],
    )
    @triton.jit
    def _swiglu_bwd_kernel(
        dy_ptr, x_ptr, dx_ptr,
        H,
        BLOCK_H: tl.constexpr,
    ):
        '''
        Backward pass, one program per (row, column block).

        With `silu_gate = gate * sigmoid(gate)` the gradients are
        `d_out = dy * silu_gate` and
        `d_gate = dy * out * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))`.

        Args:
            dy_ptr: Upstream gradient rows of `H` columns.
            x_ptr: Saved input rows of `2 * H` columns.
            dx_ptr: Input-gradient output of `2 * H` columns.
            H: Number of value (and gate) columns.
            BLOCK_H (tl.constexpr): Power-of-two block size over the columns.
        '''
        row = tl.program_id(0)
        col_blk = tl.program_id(1)
        cols = col_blk * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = cols < H

        x_row  = x_ptr + row * 2 * H
        dy_row = dy_ptr + row * H
        dx_row = dx_ptr + row * 2 * H

        out  = tl.load(x_row + cols,     mask=mask, other=0.0)
        gate = tl.load(x_row + H + cols, mask=mask, other=0.0)
        dy   = tl.load(dy_row + cols,    mask=mask, other=0.0)

        out_f  = out.to(tl.float32)
        gate_f = gate.to(tl.float32)
        dy_f   = dy.to(tl.float32)

        sig = tl.sigmoid(gate_f)
        silu_gate = gate_f * sig
        silu_grad = sig * (1.0 + gate_f * (1.0 - sig))

        d_out  = dy_f * silu_gate
        d_gate = dy_f * out_f * silu_grad

        tl.store(dx_row + cols,     d_out.to(dx_ptr.dtype.element_ty),  mask=mask)
        tl.store(dx_row + H + cols, d_gate.to(dx_ptr.dtype.element_ty), mask=mask)


if HAS_TRITON:

    class _SwiGLUFunction(torch.autograd.Function):
        '''
        Fused Triton SwiGLU with a custom backward.

        The kernel works on the flattened `[M, 2 * H]` view of the input and returns
        `[M, H]`, which is reshaped back to the caller's layout.
        '''

        @staticmethod
        def forward(ctx, x: torch.Tensor) -> torch.Tensor:
            '''
            Apply `values * SiLU(gate)` and save the input for the backward pass.

            Args:
                ctx: Autograd context; stores the 2-D input, the value width `H` and
                    the original shape.
                x (torch.Tensor): Input whose last dimension is `2 * H`.

            Returns:
                torch.Tensor: The activated tensor, last dimension `H`.
            '''
            orig_shape = x.shape
            H = orig_shape[-1] // 2

            x2d = x.reshape(-1, 2 * H).contiguous()
            M = x2d.shape[0]
            y = torch.empty((M, H), dtype=x.dtype, device=x.device)

            grid = lambda meta: (M, triton.cdiv(H, meta['BLOCK_H']))
            _swiglu_fwd_kernel[grid](x2d, y, H)

            ctx.save_for_backward(x2d)
            ctx.H = H
            ctx.orig_shape = orig_shape
            return y.reshape(*orig_shape[:-1], H)

        @staticmethod
        def backward(ctx, dy: torch.Tensor):
            '''
            Backpropagate through the activation into both halves of the input.

            Args:
                ctx: Autograd context holding `x2d`, `H` and `orig_shape`.
                dy (torch.Tensor): Upstream gradient of shape `[..., H]`.

            Returns:
                torch.Tensor: Gradient with the original `2 * H` last dimension.
            '''
            (x2d,) = ctx.saved_tensors
            H = ctx.H
            orig_shape = ctx.orig_shape

            dy2d = dy.reshape(-1, H).contiguous()
            M = dy2d.shape[0]
            dx = torch.empty((M, 2 * H), dtype=x2d.dtype, device=x2d.device)

            grid = lambda meta: (M, triton.cdiv(H, meta['BLOCK_H']))
            _swiglu_bwd_kernel[grid](dy2d, x2d, dx, H)

            return dx.reshape(orig_shape)


class SwiGLU(nn.Module):
    '''
    SwiGLU activation module: splits the last dimension into values and gate.

    `forward` returns `values * SiLU(gate)`, i.e. half the channels it receives. The
    Triton kernel is used when the tensor is large enough and `dim == -1`; otherwise
    the split is done with `torch.split`.

    Attributes:
        dim (int): Dimension that is split; -1 (the default) enables the Triton path.
        use_triton (bool): Whether the Triton path may be taken.
        MIN_OUT_ELEMS_FOR_TRITON (int): Minimum number of output elements required to
            consider the Triton kernel.
    '''

    MIN_OUT_ELEMS_FOR_TRITON = 1 << 22

    def __init__(self, dim: int = -1, use_triton: bool = True):
        '''
        Build the module.

        Args:
            dim (int): Dimension to split into values and gate.
            use_triton (bool): Allow the Triton kernel; it is still only used when
                Triton is installed.
        '''
        super().__init__()
        self.dim = dim
        self.use_triton = bool(use_triton and HAS_TRITON)

    def _should_use_triton(self, x: torch.Tensor) -> bool:
        '''
        Decide whether `x` should go through the fused Triton kernel.

        The kernel needs the split dimension to be the last one, a CUDA tensor of
        fp16 / bf16 / fp32, an even last dimension and at least
        `MIN_OUT_ELEMS_FOR_TRITON` output elements.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            bool: True when the fused kernel can and should be used.
        '''
        if not self.use_triton or self.dim != -1:
            return False
        if not x.is_cuda:
            return False
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        if x.size(-1) % 2 != 0:
            return False
        out_elems = x.numel() // 2
        return out_elems >= self.MIN_OUT_ELEMS_FOR_TRITON

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Apply the SwiGLU activation.

        Args:
            x (torch.Tensor): Input whose split dimension has even size.

        Returns:
            torch.Tensor: `values * SiLU(gate)`, with half the channels of `x`.
        '''
        if self._should_use_triton(x):
            return _SwiGLUFunction.apply(x)

        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        return out * F.silu(gate)
