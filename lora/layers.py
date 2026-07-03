import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()

        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False

        in_features = linear.in_features
        out_features = linear.out_features

        # Down projection: d_in -> r
        self.A = nn.Linear(in_features, rank, bias=False)

        # Up projection: r -> d_out
        self.B = nn.Linear(rank, out_features, bias=False)

        # LoRA initialization
        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        # W + r*BA transformation
        return self.linear(x) + self.scaling*self.B(self.A(x))