import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear(nn.Module):

    def __init__(
        self,
        q_weight,
        bias,
        scale,
        zero_point,
    ):
        super().__init__()

        self.register_buffer(
            "q_weight",
            q_weight,
        )

        self.register_buffer(
            "scale",
            scale,
        )

        self.register_buffer(
            "zero_point",
            zero_point,
        )

        self.bias = bias

    def forward(self, x):

        weight = (
            self.q_weight.float()
            - self.zero_point
        ) * self.scale

        return F.linear(
            x,
            weight,
            self.bias,
        )