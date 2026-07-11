import torch

from .latency import Timer
from .report import build_report


def benchmark(model, dataloader, device):
    model.eval()

    preds = []
    labels = []

    with Timer() as timer:
        with torch.no_grad():
            for batch in dataloader:

                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                }

                y = batch.pop("labels")

                outputs = model(**batch)

                pred = torch.argmax(outputs.logits, dim=-1)

                preds.extend(pred.cpu().tolist())
                labels.extend(y.cpu().tolist())

    batch_size = dataloader.batch_size

    return build_report(
        model=model,
        labels=labels,
        preds=preds,
        total_time=timer.elapsed,
        batch_size=batch_size,
        device=device,
    )