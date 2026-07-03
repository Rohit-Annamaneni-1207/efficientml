import torch
import torch.nn as nn
import json

from data.sst2 import get_tokenizer, prepare_sst2, get_dataloaders
from models.bert import *
from lora.inject import *
from lora.utils import *
from lora.merge import *
from core.trainer import Trainer

def main():
    # model_name = "bert-base-uncased"
    model_name = "distilbert-base-uncased"
    batch_size = 32
    rank = 8
    alpha = 16
    epochs = 3
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # target_modules = ["query", "value"]
    target_modules = ["q_lin", "v_lin"]

    tokenizer = get_tokenizer(model_name=model_name)
    sst2_dataset = prepare_sst2(tokenizer)
    train_loader, val_loader = get_dataloaders(sst2_dataset, batch_size=batch_size, tokenizer=tokenizer)

    bert_model = get_bert(model_name=model_name)
    LoRA_model = inject_lora(bert_model, rank, alpha, target_modules)

    LoRA_model = mark_only_lora_trainable(LoRA_model).to(device)

    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, LoRA_model.parameters()), lr=2e-4)

    loss = nn.CrossEntropyLoss()

    trainer = Trainer(LoRA_model, optimizer, loss, device, train_loader, val_loader)

    history = trainer.fit(epochs)
    print("Training complete. History:", history)

    trainable = sum(p.numel() for p in LoRA_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in LoRA_model.parameters())

    print(trainable, total)
    LoRA_model = merge_lora(LoRA_model)

    torch.save(LoRA_model.state_dict(), "lora_bert_sst2.pt")

    with open(f"lora_{model_name}_sst2.json", "w") as f:
        json.dump(history, f)


if __name__ == "__main__":
    main()


