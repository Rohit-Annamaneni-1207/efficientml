import torch
import torch.nn as nn

class QuantizedLinear(nn.Module):

    def __init__(
        self,
        q_weight,
        bias,
        scale,
        zero_point,
    ):

        self.q_weight = q_weight
        self.bias = bias
        self.scale = scale
        self.zero_point = zero_point
        