from compression.base import CompressionTechnique
from distillation.trainer import DistillationTrainer


class Distillation(CompressionTechnique):

    def __init__(
        self,
        student_model,
        optimizer,
        loss_fn,
    ):
        self.student_model = student_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def apply(self, model, context):
        teacher = model.to(context.device)
        teacher.eval()

        for param in teacher.parameters():
            param.requires_grad = False

        student = self.student_model.to(context.device)

        trainer = DistillationTrainer(
            teacher=teacher,
            student=student,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            device=context.device,
            train_loader=context.train_loader,
            val_loader=context.val_loader,
        )

        trainer.fit(context.epochs)

        if context.trainer is not None:
            context.trainer.model = student
            context.trainer.train_loader = context.train_loader
            context.trainer.val_loader = context.val_loader
            context.trainer.device = context.device

        return student

    def __repr__(self):
        return (
            "Distillation("
            f"student_model={self.student_model.__class__.__name__}"
            ")"
        )
