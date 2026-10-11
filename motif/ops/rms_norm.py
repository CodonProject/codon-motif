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

    @triton.jit
    def _rms_norm_fwd_kernel(
        x_ptr, w_ptr, y_ptr, rstd_ptr,
        N, eps,
        stride_xm, stride_ym,
        HAS_WEIGHT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        '''
        Forward pass, one program per row.

        Computes `x * rsqrt(mean(x^2) + eps)`, stores the reciprocal standard
        deviation for the backward pass and applies the affine weight when it is
        present.

        Args:
            x_ptr: Input rows, `[M, N]`, row stride `stride_xm`.
            w_ptr: Optional weight of length `N`; ignored when HAS_WEIGHT is False.
            y_ptr: Output rows, `[M, N]`, row stride `stride_ym`.
            rstd_ptr: `[M]` fp32 output holding `1 / sqrt(mean(x^2) + eps)`.
            N: Number of columns.
            eps: Numerical floor added before the square root.
            stride_xm: Row stride of the input.
            stride_ym: Row stride of the output.
            HAS_WEIGHT (tl.constexpr): Whether the affine weight is applied.
            BLOCK_N (tl.constexpr): Power-of-two block size over the columns.
        '''
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0)
        x_f = x.to(tl.float32)

        mean_sq = tl.sum(x_f * x_f, axis=0) / N
        rstd = 1.0 / tl.sqrt(mean_sq + eps)
        tl.store(rstd_ptr + row, rstd)

        y = x_f * rstd
        if HAS_WEIGHT:
            w = tl.load(w_ptr + cols, mask=mask, other=0.0)
            y = y * w.to(tl.float32)

        tl.store(y_ptr + row * stride_ym + cols,
                 y.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rms_norm_bwd_kernel(
        dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, dw_ptr,
        N,
        HAS_WEIGHT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        '''
        Backward pass, one program per row.

        With `x_hat = x * rstd` and `g = dy * w` the input gradient is
        `dx = rstd * (g - mean(g * x_hat) * x_hat)`. The weight gradient is
        accumulated across rows with atomics.

        Args:
            dy_ptr: Upstream gradient rows, `[M, N]`.
            x_ptr: Saved input rows, `[M, N]`.
            w_ptr: Optional weight of length `N`.
            rstd_ptr: `[M]` fp32 reciprocal standard deviation from the forward.
            dx_ptr: Input-gradient output rows, `[M, N]`.
            dw_ptr: Atomic accumulator of length `N` for the weight gradient
                (unused when HAS_WEIGHT is False).
            N: Number of columns.
            HAS_WEIGHT (tl.constexpr): Whether the affine weight is present.
            BLOCK_N (tl.constexpr): Power-of-two block size over the columns.
        '''
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row)

        x_hat = x * rstd

        if HAS_WEIGHT:
            w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            g = dy * w
        else:
            g = dy

        mean_g_x = tl.sum(g * x_hat, axis=0) / N
        dx = rstd * (g - mean_g_x * x_hat)

        tl.store(dx_ptr + row * N + cols,
                 dx.to(dx_ptr.dtype.element_ty), mask=mask)

        if HAS_WEIGHT:
            tl.atomic_add(dw_ptr + cols, dy * x_hat, mask=mask)

    @triton.jit
    def _gated_rms_norm_fwd_kernel(
        x_ptr, w_ptr, y_ptr, rstd_ptr,
        N, eps,
        HAS_WEIGHT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        '''
        Forward pass of the gated variant, one program per row.

        Each row holds `[values, gate]` concatenated along the last dimension: the
        first `N` columns are normalised and the remaining `N` columns act as the
        gate, so the output is `rms_norm(values) * SiLU(gate)`.

        Args:
            x_ptr: Input rows of `2 * N` columns, values first and gate second.
            w_ptr: Optional weight of length `N`.
            y_ptr: Output rows of `N` columns.
            rstd_ptr: `[M]` fp32 reciprocal standard deviation for the backward.
            N: Number of value columns (and of gate columns).
            eps: Numerical floor added before the square root.
            HAS_WEIGHT (tl.constexpr): Whether the affine weight is applied.
            BLOCK_N (tl.constexpr): Power-of-two block size over the columns.
        '''
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        x_row = x_ptr + row * 2 * N
        y_row = y_ptr + row * N

        x_in = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(x_row + N + cols, mask=mask, other=0.0).to(tl.float32)

        mean_sq = tl.sum(x_in * x_in, axis=0) / N
        rstd = 1.0 / tl.sqrt(mean_sq + eps)
        tl.store(rstd_ptr + row, rstd)

        x_norm = x_in * rstd
        if HAS_WEIGHT:
            w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            x_norm = x_norm * w

        sig = tl.sigmoid(gate)
        y = x_norm * (gate * sig)

        tl.store(y_row + cols, y.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _gated_rms_norm_bwd_kernel(
        dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, dw_ptr,
        N,
        HAS_WEIGHT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        '''
        Backward pass of the gated variant, one program per row, writing gradients
        into both halves of the row.

        With `x_norm = x_hat * w` and `silu_gate = gate * sigmoid(gate)` the value
        gradient is `dy * silu_gate` and the gate gradient is
        `dy * x_norm * silu_grad` with
        `silu_grad = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))`. The weight
        gradient is accumulated with atomics.

        Args:
            dy_ptr: Upstream gradient rows, `[M, N]`.
            x_ptr: Saved input rows of `2 * N` columns.
            w_ptr: Optional weight of length `N`.
            rstd_ptr: `[M]` fp32 reciprocal standard deviation from the forward.
            dx_ptr: Input-gradient output of `2 * N` columns.
            dw_ptr: Atomic accumulator of length `N` for the weight gradient.
            N: Number of value columns (and of gate columns).
            HAS_WEIGHT (tl.constexpr): Whether the affine weight is present.
            BLOCK_N (tl.constexpr): Power-of-two block size over the columns.
        '''
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        x_row  = x_ptr  + row * 2 * N
        dx_row = dx_ptr + row * 2 * N

        x    = tl.load(x_row + cols,     mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(x_row + N + cols, mask=mask, other=0.0).to(tl.float32)
        dy   = tl.load(dy_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row)

        x_hat = x * rstd

        if HAS_WEIGHT:
            w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            x_norm = x_hat * w
        else:
            x_norm = x_hat

        sig = tl.sigmoid(gate)
        silu_gate = gate * sig
        silu_grad = sig * (1.0 + gate * (1.0 - sig))

        d_x_norm = dy * silu_gate
        d_gate   = dy * x_norm * silu_grad

        if HAS_WEIGHT:
            g = d_x_norm * w
        else:
            g = d_x_norm

        mean_g_x = tl.sum(g * x_hat, axis=0) / N
        d_x = rstd * (g - mean_g_x * x_hat)

        tl.store(dx_row + cols,     d_x.to(dx_ptr.dtype.element_ty),    mask=mask)
        tl.store(dx_row + N + cols, d_gate.to(dx_ptr.dtype.element_ty), mask=mask)

        if HAS_WEIGHT:
            tl.atomic_add(dw_ptr + cols, d_x_norm * x_hat, mask=mask)


if HAS_TRITON:

    @autocast_safe
    class _RMSNormFunction(torch.autograd.Function):
        '''
        Fused Triton RMSNorm with a custom backward.

        The kernel works on the flattened `[M, N]` view of the input, where `N` is
        the last dimension.
        '''

        @staticmethod
        def forward(ctx, x, weight, eps):
            '''
            Normalise `x` over its last dimension and save what the backward needs.

            Args:
                ctx: Autograd context; stores the 2-D input, the weight (or an empty
                    placeholder) and the reciprocal standard deviation.
                x: Input tensor of any shape; only the last dimension matters.
                weight: Optional per-channel weight of length `x.shape[-1]`.
                eps: Numerical floor added before the square root.

            Returns:
                torch.Tensor: The normalised tensor, shaped like `x`.
            '''
            orig_shape = x.shape
            N = orig_shape[-1]
            x2d = x.reshape(-1, N).contiguous()
            M = x2d.shape[0]

            y = torch.empty_like(x2d)
            rstd = torch.empty(M, dtype=torch.float32, device=x.device)

            BLOCK_N = triton.next_power_of_2(N)
            _rms_norm_fwd_kernel[(M,)](
                x2d, weight if weight is not None else x2d,
                y, rstd, N, eps,
                x2d.stride(0), y.stride(0),
                HAS_WEIGHT=weight is not None,
                BLOCK_N=BLOCK_N,
            )

            w_save = weight if weight is not None else x2d.new_empty(0)
            ctx.save_for_backward(x2d, w_save, rstd)
            ctx.orig_shape = orig_shape
            ctx.has_weight = weight is not None
            return y.reshape(orig_shape)

        @staticmethod
        def backward(ctx, dy):
            '''
            Compute the input and the weight gradient.

            Args:
                ctx: Autograd context holding `x2d`, the saved weight and `rstd`.
                dy: Upstream gradient, shaped like the forward output.

            Returns:
                tuple: `(dx, dw, None)`; `dw` is None when the weight does not
                require a gradient.
            '''
            x2d, w_save, rstd = ctx.saved_tensors
            orig_shape = ctx.orig_shape
            N = orig_shape[-1]
            M = x2d.shape[0]
            dy2d = dy.reshape(-1, N).contiguous()

            dx = torch.empty_like(x2d)

            if ctx.has_weight:
                dw_acc = torch.zeros_like(w_save, dtype=torch.float32)
            else:
                dw_acc = x2d.new_empty(0)

            BLOCK_N = triton.next_power_of_2(N)
            _rms_norm_bwd_kernel[(M,)](
                dy2d, x2d, w_save, rstd, dx,
                dw_acc if ctx.has_weight else dx,
                N,
                HAS_WEIGHT=ctx.has_weight,
                BLOCK_N=BLOCK_N,
            )

            dx = dx.reshape(orig_shape)
            if ctx.has_weight and ctx.needs_input_grad[1]:
                dw = dw_acc.to(w_save.dtype)
            else:
                dw = None

            return dx, dw, None


    @autocast_safe
    class _GatedRMSNormFunction(torch.autograd.Function):
        '''
        Fused Triton gated RMSNorm with a custom backward.

        The last dimension of the input is `2 * N`: the first `N` channels are the
        values to normalise, the last `N` channels are the gate.
        '''

        @staticmethod
        def forward(ctx, x, weight, eps):
            '''
            Normalise the value half of `x`, gate it and save what the backward
            needs.

            Args:
                ctx: Autograd context; stores the 2-D input, the weight (or an empty
                    placeholder), `rstd` and the value width `N`.
                x: Input tensor whose last dimension is `2 * N`.
                weight: Optional per-channel weight of length `N`.
                eps: Numerical floor added before the square root.

            Returns:
                torch.Tensor: The gated output, with the last dimension reduced to
                `N`.
            '''
            orig_shape = x.shape
            N = orig_shape[-1] // 2
            x2d = x.reshape(-1, 2 * N).contiguous()
            M = x2d.shape[0]

            y = torch.empty(M, N, dtype=x.dtype, device=x.device)
            rstd = torch.empty(M, dtype=torch.float32, device=x.device)

            BLOCK_N = triton.next_power_of_2(N)
            _gated_rms_norm_fwd_kernel[(M,)](
                x2d, weight if weight is not None else x2d,
                y, rstd, N, eps,
                HAS_WEIGHT=weight is not None,
                BLOCK_N=BLOCK_N,
            )

            w_save = weight if weight is not None else x2d.new_empty(0)
            ctx.save_for_backward(x2d, w_save, rstd)
            ctx.orig_shape = orig_shape
            ctx.N = N
            ctx.has_weight = weight is not None
            return y.reshape(*orig_shape[:-1], N)

        @staticmethod
        def backward(ctx, dy):
            '''
            Compute the gradients of the value half, the gate half and the weight.

            Args:
                ctx: Autograd context holding `x2d`, the saved weight, `rstd` and
                    `N`.
                dy: Upstream gradient of shape `[..., N]`.

            Returns:
                tuple: `(dx, dw, None)` where `dx` has the original `2 * N` last
                dimension; `dw` is None when the weight does not require a gradient.
            '''
            x2d, w_save, rstd = ctx.saved_tensors
            orig_shape = ctx.orig_shape
            N = ctx.N
            M = x2d.shape[0]
            dy2d = dy.reshape(-1, N).contiguous()

            dx = torch.empty_like(x2d)

            if ctx.has_weight:
                dw_acc = torch.zeros_like(w_save, dtype=torch.float32)
            else:
                dw_acc = x2d.new_empty(0)

            BLOCK_N = triton.next_power_of_2(N)
            _gated_rms_norm_bwd_kernel[(M,)](
                dy2d, x2d, w_save, rstd, dx,
                dw_acc if ctx.has_weight else dx,
                N,
                HAS_WEIGHT=ctx.has_weight,
                BLOCK_N=BLOCK_N,
            )

            dx = dx.reshape(orig_shape)
            if ctx.has_weight and ctx.needs_input_grad[1]:
                dw = dw_acc.to(w_save.dtype)
            else:
                dw = None

            return dx, dw, None


class RMSNorm(nn.Module):
    '''
    Root-mean-square layer normalisation over the last dimension.

    `y = x / sqrt(mean(x^2) + eps) * weight`, where the weight is optional. The
    module picks the fastest available implementation at call time: the Triton
    kernel, `F.rms_norm`, or the eager formula.

    Attributes:
        dim (int): Size of the normalised (last) dimension.
        eps (float): Numerical floor added before the square root.
        elementwise_affine (bool): Whether a learnable weight is used.
        weight (Optional[nn.Parameter]): The weight, or None when disabled.
        use_triton (bool): Whether the Triton path may be taken.
        MIN_ELEMS_FOR_TRITON (int): Element count below which the Triton kernel is
            not worth launching.
    '''

    MIN_ELEMS_FOR_TRITON = 1 << 18

    def __init__(self, dim, eps=1e-6,
                 elementwise_affine=True, use_triton=True):
        '''
        Build the module.

        Args:
            dim (int): Size of the normalised dimension.
            eps (float): Numerical floor added before the square root.
            elementwise_affine (bool): Whether to create a learnable weight.
            use_triton (bool): Allow the Triton kernel; it is still only used when
                Triton is installed.
        '''
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)
        self.use_triton = bool(use_triton and HAS_TRITON)

    def forward(self, x):
        '''
        Normalise `x` over its last dimension.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: The normalised tensor, same shape and dtype as `x`.
        '''
        if (self.use_triton and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and x.numel() >= self.MIN_ELEMS_FOR_TRITON):
            return _RMSNormFunction.apply(x, self.weight, self.eps)

        if (x.is_cuda and self.weight is not None
            and self.weight.numel() == x.shape[-1]
            and hasattr(F, 'rms_norm')):
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)

        x_f = x.float()
        var = x_f.pow(2).mean(-1, keepdim=True)
        y = (x_f * torch.rsqrt(var + self.eps)).to(x.dtype)
        if self.weight is not None:
            y = y * self.weight
        return y


class GatedRMSNorm(nn.Module):
    '''
    RMSNorm with a SiLU gate, applied to the value half of the input.

    The last dimension of the input is `2 * dim`: the first `dim` channels are the
    values and the last `dim` channels are the gate, so the output is
    `rms_norm(values) * SiLU(gate)` with `dim` channels.

    Attributes:
        dim (int): Number of value (and gate) channels.
        eps (float): Numerical floor added before the square root.
        elementwise_affine (bool): Whether a learnable weight is used.
        weight (Optional[nn.Parameter]): The weight, or None when disabled.
        use_triton (bool): Whether the Triton path may be taken.
        MIN_ELEMS_FOR_TRITON (int): Element count below which the Triton kernel is
            not worth launching.
    '''

    MIN_ELEMS_FOR_TRITON = 1 << 18

    def __init__(self, dim, eps=1e-6, elementwise_affine=True, use_triton=True):
        '''
        Build the module.

        Args:
            dim (int): Number of value (and gate) channels.
            eps (float): Numerical floor added before the square root.
            elementwise_affine (bool): Whether to create a learnable weight.
            use_triton (bool): Allow the Triton kernel; it is still only used when
                Triton is installed.
        '''
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)
        self.use_triton = bool(use_triton and HAS_TRITON)

    def forward(self, x):
        '''
        Normalise the value half of `x` and gate it.

        Args:
            x (torch.Tensor): Input whose last dimension is `2 * dim`; the first
                `dim` channels are the values, the last `dim` channels the gate.

        Returns:
            torch.Tensor: `rms_norm(values) * SiLU(gate)`, last dimension `dim`.
        '''
        if (self.use_triton and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and x.size(-1) == 2 * self.dim
            and x.numel() >= self.MIN_ELEMS_FOR_TRITON):
            return _GatedRMSNormFunction.apply(x, self.weight, self.eps)

        x_in, gate = x.split(self.dim, dim=-1)
        x_f = x_in.float()
        var = x_f.pow(2).mean(-1, keepdim=True)
        y = (x_f * torch.rsqrt(var + self.eps)).to(x.dtype)
        if self.weight is not None:
            y = y * self.weight
        return y * F.silu(gate)
