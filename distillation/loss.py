import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 2.0,
    ):
        super().__init__()

        self.alpha = alpha
        self.temperature = temperature

        self.ce_loss = nn.CrossEntropyLoss()

        self.kl_loss = nn.KLDivLoss(
            reduction="batchmean",
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ):

        # Standard supervised loss
        ce = self.ce_loss(
            student_logits,
            labels,
        )

        # Temperature-scaled distributions
        student_log_probs = F.log_softmax(
            student_logits / self.temperature,
            dim=-1,
        )

        teacher_probs = F.softmax(
            teacher_logits / self.temperature,
            dim=-1,
        )

        kl = self.kl_loss(
            student_log_probs,
            teacher_probs,
        )

        # Hinton et al. scaling
        kl = kl * (self.temperature ** 2)

        loss = (
            self.alpha * ce
            + (1 - self.alpha) * kl
        )

        return loss