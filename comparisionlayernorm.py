import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import logging

# Dummy logger setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Dummy STE Functions ---
class floor_ste(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class round_ste(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# --- Dummy QuantAct ---
class QuantAct(nn.Module):
    def __init__(self, bit, quant_mode=True):
        super().__init__()
        self.bit = bit
        self.quant_mode = quant_mode
    def forward(self, x, scale=None):
        return x

# --- Your IntLayerNorm Implementation ---
class IntLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps, output_bit=8, quant_mode=False, force_dequant="none"):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(normalized_shape))  # Use ones for fair comparison
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

        self.quant_mode = quant_mode
        if force_dequant in ["nonlinear", "layernorm"]:
            logger.info("Force dequantize layernorm")
            self.quant_mode = False

        self.register_buffer("shift", torch.zeros(1))
        self.output_bit = output_bit
        self.max_bit = 32
        self.dim_sqrt = None
        self.activation = QuantAct(self.output_bit, quant_mode=self.quant_mode)

    def set_shift(self, y_int):
        with torch.no_grad():
            y_sq_int = y_int**2
            var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
            shift = (torch.log2(torch.sqrt(var_int / 2**self.max_bit)).ceil()).max()
            self.shift = torch.max(self.shift, shift)

    def overflow_fallback(self, y_int):
        self.set_shift(y_int)
        y_int_shifted = floor_ste.apply(y_int / 2**self.shift)
        y_sq_int = y_int_shifted**2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
        return var_int

    def forward(self, x, scaling_factor=None):
        if not self.quant_mode:
            mean = x.mean(axis=2, keepdim=True)
            y = x - mean
            var = torch.mean(y**2, axis=2, keepdim=True)
            x = y / torch.sqrt(self.eps + var)
            x = x * self.weight + self.bias
            return x, None

        if self.dim_sqrt is None:
            n = torch.tensor(x.shape[2], dtype=torch.float)
            self.dim_sqrt = torch.sqrt(n).to(x.device)

        x_int = x / scaling_factor
        mean_int = round_ste.apply(x_int.mean(axis=2, keepdim=True))
        y_int = x_int - mean_int
        y_int_shifted = floor_ste.apply(y_int / 2**self.shift)
        y_sq_int = y_int_shifted**2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

        if self.training:
            if var_int.max() >= 2**self.max_bit:
                var_int = self.overflow_fallback(y_int)

        std_int = floor_ste.apply(torch.sqrt(var_int)) * 2**self.shift
        factor = floor_ste.apply(2**31 / std_int)
        y_int = floor_ste.apply(y_int * factor / 2)
        scaling_factor = self.dim_sqrt / 2**30

        bias = self.bias.data.detach() / (self.weight.data.detach())
        bias_int = floor_ste.apply(bias / scaling_factor)

        y_int = y_int + bias_int
        scaling_factor = scaling_factor * self.weight
        x = y_int * scaling_factor

        return x, scaling_factor

# --- Main Plotting Code ---
B, H, D = 1, 1, 128  # Batch, Head, Dimension
dummy_input = torch.randn(B, H, D)

# PyTorch LayerNorm
ln = nn.LayerNorm(D, eps=1e-5)
ln_out = ln(dummy_input)

# IntLayerNorm
int_ln = IntLayerNorm(normalized_shape=D, eps=1e-5, quant_mode=True)
int_ln.train(False)  # Set to eval mode to avoid fallback
scaling_factor = torch.tensor(0.1)  # Try values like 0.05, 0.1, etc.
int_ln_out, _ = int_ln(dummy_input.clone(), scaling_factor)

# Plotting
plt.figure(figsize=(10, 4))
plt.plot(ln_out[0][0].detach().numpy(), label='LayerNorm (float)', linewidth=2)
plt.plot(int_ln_out[0][0].detach().numpy(), label='IntLayerNorm (quant)', linestyle='--')
plt.title('Comparison: Standard LayerNorm vs IntLayerNorm')
plt.xlabel('Feature Index')
plt.ylabel('Normalized Output')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
