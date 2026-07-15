import time

import torch
from tqdm import tqdm

from eval.metrics import accuracy
from eval.utils import get_model_size


def benchmark(model, dataloader, device):

    model.to(device)
    model.eval()

    total_correct = 0
    total_samples = 0

    total_time = 0.0

    with torch.no_grad():

        for batch in tqdm(dataloader, desc="Benchmark", leave=False):

            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            labels = batch.pop("labels")

            start = time.perf_counter()
            outputs = model(**batch)
            end = time.perf_counter()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            total_time += end - start

    results = {
        "accuracy": accuracy(total_correct, total_samples),
        "total_time": total_time,
        "avg_latency": total_time / total_samples,
        "throughput": total_samples / total_time,
        "model_size_mb": get_model_size(model),
    }

    return results