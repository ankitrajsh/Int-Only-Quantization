import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import logging

# Dummy logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Floor with straight-through estimator (dummy for plotting)
class floor_ste(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# IntGELU class as provided
class IntGELU(nn.Module):
    def __init__(self, quant_mode=True, force_dequant="none"):
        super().__init__()
        self.quant_mode = quant_mode
        if force_dequant in ["nonlinear", "gelu"]:
            logger.info("Force dequantize gelu")
            self.quant_mode = False

        if not self.quant_mode:
            self.activation_fn = nn.GELU()

        self.k = 1.4142
        self.const = 14  # dummy integer constant
        self.coeff = [-0.2888, -1.769, 1]
        self.coeff[2] /= self.coeff[0]

    def int_erf(self, x_int, scaling_factor):
        b_int = torch.floor(self.coeff[1] / scaling_factor)
        c_int = torch.floor(self.coeff[2] / scaling_factor**2)
        sign = torch.sign(x_int)

        abs_int = torch.min(torch.abs(x_int), -b_int)
        y_int = sign * ((abs_int + b_int) ** 2 + c_int)
        scaling_factor = scaling_factor**2 * self.coeff[0]

        y_int = floor_ste.apply(y_int / 2**self.const)
        scaling_factor = scaling_factor * 2**self.const
        return y_int, scaling_factor

    def forward(self, x, scaling_factor=None):
        if not self.quant_mode:
            return self.activation_fn(x), None

        x_int = x / scaling_factor
        sigmoid_int, sigmoid_scaling_factor = self.int_erf(x_int, scaling_factor / self.k)

        shift_int = 1.0 // sigmoid_scaling_factor
        x_int = x_int * (sigmoid_int + shift_int)
        scaling_factor = scaling_factor * sigmoid_scaling_factor / 2

        return x_int * scaling_factor, scaling_factor

# Define input range
x = torch.linspace(-5, 5, steps=500)

# Use normal GELU
gelu = nn.GELU()
y_gelu = gelu(x)

# Use IntGELU
scaling_factor = torch.tensor(0.1)  # try different values like 0.1, 0.2, etc.
int_gelu = IntGELU(quant_mode=True)
y_int_gelu, _ = int_gelu(x, scaling_factor=scaling_factor)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x.numpy(), y_gelu.detach().numpy(), label="GELU (Float32)", linewidth=2)
plt.plot(x.numpy(), y_int_gelu.detach().numpy(), label=f"IntGELU (Quantized, scale={scaling_factor})", linestyle='--')
plt.title("Comparison of GELU vs IntGELU")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()
