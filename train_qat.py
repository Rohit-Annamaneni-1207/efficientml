import torch
import torch.nn as nn

from data.sst2 import (
    get_tokenizer,
    prepare_sst2,
    get_dataloaders,
)

from models.bert import get_bert

from quant.inject_qat import inject_qat

from core.trainer import Trainer

from eval.benchmark import benchmark
from eval.report import print_report


def main():

    model_name = "distilbert-base-uncased"

    batch_size = 16
    epochs = 3
    learning_rate = 2e-5

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    tokenizer = get_tokenizer(model_name)

    dataset = prepare_sst2(tokenizer)

    train_loader, val_loader = get_dataloaders(
        dataset,
        batch_size=batch_size,
        tokenizer=tokenizer,
    )

    model = get_bert(model_name)

    model = inject_qat(
        model,
        num_bits=8,
    )

    model.to(device)

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(trainable)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    trainer.fit(epochs)

    torch.save(
        model.state_dict(),
        "checkpoints/qat_distilbert.pth",
    )

    print("\nRunning benchmark...\n")

    results = benchmark(
        model,
        val_loader,
        device,
    )

    print_report(results)


if __name__ == "__main__":
    main()