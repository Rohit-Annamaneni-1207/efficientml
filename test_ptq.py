import torch

from quant.ptq import PTQ
from models.bert import get_bert
from data.sst2 import (
    get_tokenizer,
    prepare_sst2,
    get_dataloaders,
)


def print_quantized_modules(model):

    print("\n========== Quantized Modules ==========\n")

    for name, module in model.named_modules():

        if "quantized" in str(type(module)).lower():
            print(f"{name:<70} {type(module)}")


def get_model_size(model):

    torch.save(model.state_dict(), "temp.pth")

    size_mb = (
        __import__("os").path.getsize("temp.pth")
        / (1024 * 1024)
    )

    __import__("os").remove("temp.pth")

    return size_mb


def main():

    tokenizer = get_tokenizer("distilbert-base-uncased")

    dataset = prepare_sst2(tokenizer)

    train_loader, val_loader = get_dataloaders(
        dataset,
        batch_size=16,
        tokenizer=tokenizer,
    )

    model = get_bert("distilbert-base-uncased")

    print(f"FP32 Size : {get_model_size(model):.2f} MB")

    ptq = PTQ(backend="qnnpack")

    print("\nPreparing model...")
    prepared_model = ptq.prepare(model)

    print("Calibrating model...")
    prepared_model = ptq.calibrate(
        prepared_model,
        train_loader,
        device="cpu",
    )

    print("Converting model...")
    quantized_model = ptq.convert(prepared_model)

    print(f"INT8 Size : {get_model_size(quantized_model):.2f} MB")

    print_quantized_modules(quantized_model)


if __name__ == "__main__":
    main()