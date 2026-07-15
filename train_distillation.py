import torch

from data.sst2 import (
    get_tokenizer,
    prepare_sst2,
    get_dataloaders,
)

from models.bert import get_bert

from distillation.loss import DistillationLoss
from distillation.trainer import DistillationTrainer

from eval.benchmark import benchmark
from eval.report import print_report


def main():

    teacher_name = "textattack/bert-base-uncased-SST-2"
    student_name = "distilbert-base-uncased"

    batch_size = 8
    epochs = 3
    learning_rate = 2e-5

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    tokenizer = get_tokenizer(student_name)

    dataset = prepare_sst2(tokenizer)

    train_loader, val_loader = get_dataloaders(
        dataset,
        batch_size=batch_size,
        tokenizer=tokenizer,
    )

    print("Loading teacher...")

    teacher = get_bert(teacher_name)

    teacher.to(device)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    print("Loading student...")

    student = get_bert(
        student_name,
        num_labels=2,
    )

    student.to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=learning_rate,
    )

    loss_fn = DistillationLoss(
        alpha=0.5,
        temperature=2.0,
    )

    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    trainer.fit(epochs)

    torch.save(
        student.state_dict(),
        "checkpoints/distilled_distilbert.pth",
    )

    print("\nRunning benchmark...\n")

    results = benchmark(
        student,
        val_loader,
        device,
    )

    print_report(results)


if __name__ == "__main__":
    main()