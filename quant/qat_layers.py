import torch
import torch.nn as nn
import torch.nn.functional as F

from quant.fake_quant import FakeQuantizer


class QATLinear(nn.Module):

    def __init__(
        self,
        linear: nn.Linear,
        num_bits: int = 8,
    ):
        super().__init__()

        self.linear = linear
        
        self.weight_fake_quant = FakeQuantizer(
            num_bits=num_bits
        )

        self.activation_fake_quant = FakeQuantizer(
            num_bits=num_bits
        )

    def forward(self, x):

        x = self.activation_fake_quant(x)

        weight = self.weight_fake_quant(
            self.linear.weight
        )

        return F.linear(
            x,
            weight,
            self.linear.bias,
        )