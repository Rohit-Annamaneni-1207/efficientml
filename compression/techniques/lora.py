from compression.base import CompressionTechnique
from lora.inject import inject_lora
from lora.merge import merge_lora
from lora.utils import mark_only_lora_trainable


import torch


import torch


def _reset_trainer_optimizer(trainer):

    params = [
        p
        for p in trainer.model.parameters()
        if p.requires_grad
    ]

    lr = trainer.optimizer.param_groups[0]["lr"]

    trainer.optimizer = torch.optim.AdamW(
        params,
        lr=lr,
    )


class LoRA(CompressionTechnique):

    def __init__(
        self,
        rank: int,
        alpha: float,
        target_modules,
    ):
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules

    def apply(self, model, context):
        model = inject_lora(
            model,
            rank=self.rank,
            alpha=self.alpha,
            target_modules=self.target_modules,
        )
        model = mark_only_lora_trainable(model)
        model.to(context.device)

        trainer = context.trainer
        trainer.model = model
        trainer.train_loader = context.train_loader
        trainer.val_loader = context.val_loader
        trainer.device = context.device

        _reset_trainer_optimizer(trainer)

        trainer.fit(context.epochs)

        model = merge_lora(model)
        trainer.model = model

        return model

    def __repr__(self):
        return (
            "LoRA("
            f"rank={self.rank}, "
            f"alpha={self.alpha}, "
            f"target_modules={self.target_modules}"
            ")"
        )
