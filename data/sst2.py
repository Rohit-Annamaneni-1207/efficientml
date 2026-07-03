from transformers import AutoTokenizer
from datasets import load_dataset

def get_tokenizer(model_name: str):
    """
    Get the tokenizer for the specified model.

    Args:
        model_name (str): The name of the model.

    Returns:
        tokenizer: The tokenizer corresponding to the specified model.
    """

    return AutoTokenizer.from_pretrained(model_name)

def load_sst2():
    dataset = load_dataset("glue", "sst2")
    return dataset

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True
    )

def prepare_sst2(tokenizer):
    dataset = load_sst2()

    # Tokenize dataset in batches
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True
    )

    # Rename label column to match trainer expectations
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    # Remove unused columns
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])

    # Convert to PyTorch tensors
    tokenized_dataset.set_format("torch")

    return tokenized_dataset

from torch.utils.data import DataLoader


def get_dataloaders(tokenized_dataset, batch_size):
    train_loader = DataLoader(
        tokenized_dataset["train"],
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader