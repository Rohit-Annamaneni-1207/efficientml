import torch

from quant.algorithms import AffineQuantizer

x = torch.randn(100)

quantizer = AffineQuantizer()

q = quantizer.quantize(x)
x_hat = quantizer.dequantize(q)

print("Original")
print(x[:5])

print("Quantized")
print(q[:5])

print("Reconstructed")
print(x_hat[:5])

print("MSE:", torch.mean((x - x_hat) ** 2))