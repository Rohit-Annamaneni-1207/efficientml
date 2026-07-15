import torch
import torch.nn as nn

from quant.observers import MinMaxObserver

class FakeQuantizer(nn.Module):

    def __init__(
        self,
        observer=None,
        num_bits: int = 8,
    ):
        super().__init__()

        self.num_bits = num_bits

        self.qmin = -(2 ** (num_bits - 1))
        self.qmax = (2 ** (num_bits - 1)) - 1

        if observer is None:
            observer = MinMaxObserver(num_bits)

        self.observer = observer

    def forward(self, x: torch.Tensor):

        self.observer.observe(x)

        scale, zero_point = self.observer.calculate_qparams()
        scale = scale.detach()
        zero_point = zero_point.detach()

        scale = torch.clamp(
                scale,
                min=1e-8,
            )

        q = torch.round(x / scale) + zero_point

        q = torch.clamp(
            q,
            self.qmin,
            self.qmax,
        )

        x_hat = (q - zero_point) * scale

        # Straight Through Estimator
        return x + (x_hat - x).detach()