import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Assuming floor_ste is some custom function needed by your int_erf method.
# If you don't have the definition, I'll define a placeholder here:
class floor_ste(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.floor(input)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# Paste your IntGELU class here or import it if you saved it elsewhere.
class IntGELU(nn.Module):
    def __init__(self, quant_mode=True, force_dequant="none"):
        super().__init__()
        self.quant_mode = quant_mode

        if force_dequant in ["nonlinear", "gelu"]:
            print("Force dequantize gelu")
            self.quant_mode = False

        if not self.quant_mode:
            self.activation_fn = nn.GELU()

        self.k = 1.4142
        self.const = 14  # dummy integer constant
        self.coeff = [-0.2888, -1.769, 1]  # a(x+b)**2 + c
        self.coeff[2] /= self.coeff[0]

    def int_erf(self, x_int, scaling_factor):
        b_int = torch.floor(self.coeff[1] / scaling_factor)
        c_int = torch.floor(self.coeff[2] / scaling_factor**2)
        sign = torch.sign(x_int)

        abs_int = torch.min(torch.abs(x_int), -b_int)
        y_int = sign * ((abs_int + b_int) ** 2 + c_int)
        scaling_factor = scaling_factor**2 * self.coeff[0]

        # avoid overflow
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


# Now, let's plot

def plot_intgelu():
    model = IntGELU(quant_mode=True)

    # Define input range: typically -5 to +5 for activation functions
    x = torch.linspace(-5, 5, 500)

    # Choose an arbitrary scaling factor (positive, small)
    scaling_factor = 0.1

    # Run forward pass
    with torch.no_grad():
        y, _ = model.forward(x, scaling_factor)

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(x.numpy(), y.numpy(), label="IntGELU Quantized")
    
    # Also plot standard GELU for comparison
    gelu = nn.GELU()
    plt.plot(x.numpy(), gelu(x).numpy(), label="Standard GELU", linestyle="--")
    
    plt.title("IntGELU Activation Function")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_intgelu()









#########   
class IntGELU(nn.Module):
    """
    Quantized version of `torch.nn.GELU`. Adds quantization-specific arguments on top of `torch.nn.GELU`.

    Args:
        quant_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the layer is quantized.
        force_dequant (`str`, *optional*, defaults to `"none"`):
            Force dequantize the layer if either "gelu" or "nonlinear" is given.
    """

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
        self.coeff = [-0.2888, -1.769, 1]  # a(x+b)**2 + c
        self.coeff[2] /= self.coeff[0]

    def int_erf(self, x_int, scaling_factor):
        b_int = torch.floor(self.coeff[1] / scaling_factor)
        c_int = torch.floor(self.coeff[2] / scaling_factor**2)
        sign = torch.sign(x_int)

        abs_int = torch.min(torch.abs(x_int), -b_int)
        y_int = sign * ((abs_int + b_int) ** 2 + c_int)
        scaling_factor = scaling_factor**2 * self.coeff[0]

        # avoid overflow
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


