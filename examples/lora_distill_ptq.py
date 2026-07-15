import torch
import torch.nn as nn

from compression import (
    CompressionPipeline,
    Distillation,
    LoRA,
    PTQ,
    PipelineContext,
)
from core.trainer import Trainer
from data.sst2 import get_dataloaders, get_tokenizer, prepare_sst2
from distillation.loss import DistillationLoss
from models.bert import get_bert


def main():
    teacher_name = "textattack/bert-base-uncased-SST-2"
    student_name = "distilbert-base-uncased"
    batch_size = 8
    epochs = 3
    learning_rate = 2e-5

    device = "cpu"

    tokenizer = get_tokenizer(student_name)
    dataset = prepare_sst2(tokenizer)

    train_loader, val_loader = get_dataloaders(
        dataset,
        batch_size=batch_size,
        tokenizer=tokenizer,
    )

    teacher = get_bert(teacher_name)
    teacher.to(device)

    trainer = Trainer(
        model=teacher,
        optimizer=torch.optim.AdamW(
            teacher.parameters(),
            lr=learning_rate,
        ),
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    student = get_bert(
        student_name,
        num_labels=2,
    )

    distillation = Distillation(
        student_model=student,
        optimizer=torch.optim.AdamW(
            student.parameters(),
            lr=learning_rate,
        ),
        loss_fn=DistillationLoss(
            alpha=0.5,
            temperature=2.0,
        ),
    )

    context = PipelineContext(
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        calibration_loader=val_loader,
        device=device,
        epochs=epochs,
    )

    pipeline = CompressionPipeline(teacher)
    pipeline.add(
        LoRA(
            rank=8,
            alpha=16,
            target_modules=["query", "value"],
        )
    )
    pipeline.add(distillation)
    pipeline.add(PTQ())

    model = pipeline.apply(context)
    print(model)


if __name__ == "__main__":
    main()
