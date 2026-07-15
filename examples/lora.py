import torch
import torch.nn as nn

from compression import CompressionPipeline, LoRA, PipelineContext
from core.trainer import Trainer
from data.sst2 import get_dataloaders, get_tokenizer, prepare_sst2
from models.bert import get_bert


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
    model.to(device)

    trainer = Trainer(
        model=model,
        optimizer=torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
        ),
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    context = PipelineContext(
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        calibration_loader=None,
        device=device,
        epochs=epochs,
    )

    pipeline = CompressionPipeline(model)
    pipeline.add(
        LoRA(
            rank=8,
            alpha=16,
            target_modules=["q_lin", "v_lin"],
        )
    )

    model = pipeline.apply(context)
    print(model)


if __name__ == "__main__":
    main()
