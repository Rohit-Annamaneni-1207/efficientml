from .metrics import classification_metrics
from .latency import latency_metrics
from .model_stats import parameter_stats, model_size


def build_report(
    model,
    labels,
    preds,
    total_time,
    batch_size,
    device,
):
    report = {}

    report.update(classification_metrics(labels, preds))
    report.update(latency_metrics(total_time, len(labels)))
    report.update(parameter_stats(model))

    report["model_size_mb"] = model_size(model)
    report["device"] = str(device)
    report["batch_size"] = batch_size
    report["num_samples"] = len(labels)

    return report