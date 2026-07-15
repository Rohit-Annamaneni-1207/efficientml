from __future__ import annotations

import io
import queue
import tempfile
import threading
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import torch
import torch.nn as nn

from compression import CompressionPipeline, Distillation, LoRA, PTQ, PipelineContext, QAT
from core.trainer import Trainer
from eval.benchmark import benchmark
from eval.report import print_report
from ui.components import (
    TECHNIQUE_DESCRIPTIONS,
    initial_comparison_rows,
    render_execution_summary,
    render_metrics_dashboard,
    render_pipeline_preview,
    render_progress,
)


class QueueLogStream(io.TextIOBase):

    def __init__(self, event_queue: queue.Queue):
        self.event_queue = event_queue

    def write(self, text: str) -> int:
        if not text:
            return 0

        cleaned = text.replace("\r", "\n")
        self.event_queue.put(("log", cleaned))
        return len(text)

    def flush(self) -> None:
        return None


class AdaptiveTrainer:

    def __init__(
        self,
        model,
        learning_rate: float,
        device: str,
        train_loader,
        val_loader,
    ):
        self._learning_rate = learning_rate
        self._loss_fn = nn.CrossEntropyLoss()
        self._device = device
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._trainer = Trainer(
            model=model,
            optimizer=self._build_optimizer(model),
            loss_fn=self._loss_fn,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
        )

    def _build_optimizer(self, model):
        params = [
            param
            for param in model.parameters()
            if param.requires_grad
        ]

        if not params:
            params = list(model.parameters())

        return torch.optim.AdamW(
            params,
            lr=self._learning_rate,
        )

    @property
    def model(self):
        return self._trainer.model

    @model.setter
    def model(self, value):
        self._trainer.model = value
        self._trainer.optimizer = self._build_optimizer(value)

    @property
    def optimizer(self):
        return self._trainer.optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._trainer.optimizer = value

    @property
    def loss_fn(self):
        return self._trainer.loss_fn

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value
        self._trainer.device = value

    @property
    def train_loader(self):
        return self._train_loader

    @train_loader.setter
    def train_loader(self, value):
        self._train_loader = value
        self._trainer.train_loader = value

    @property
    def val_loader(self):
        return self._val_loader

    @val_loader.setter
    def val_loader(self, value):
        self._val_loader = value
        self._trainer.val_loader = value

    def fit(self, epochs: int):
        return self._trainer.fit(epochs)


def _load_dataset_backend():
    from data.sst2 import (
        get_dataloaders,
        get_tokenizer,
        prepare_sst2,
    )

    return get_dataloaders, get_tokenizer, prepare_sst2


def _load_model_backend():
    from models.bert import get_bert

    return get_bert


def _load_distillation_backend():
    from distillation.loss import DistillationLoss

    return DistillationLoss


def technique_visibility(technique: str):
    return (
        gr.update(
            value=f'<div class="technique-note">{TECHNIQUE_DESCRIPTIONS[technique]}</div>'
        ),
        gr.update(visible=technique == "LoRA"),
        gr.update(visible=technique == "Distillation"),
        gr.update(visible=technique == "QAT"),
        gr.update(visible=technique == "PTQ"),
    )


def add_technique(technique: str, pipeline: list[str] | None):
    pipeline = list(pipeline or [])

    if technique not in pipeline:
        pipeline.append(technique)

    return (
        pipeline,
        render_pipeline_preview(pipeline),
        gr.update(choices=pipeline, value=technique),
    )


def move_technique_up(selected: str | None, pipeline: list[str] | None):
    pipeline = list(pipeline or [])

    if selected in pipeline:
        index = pipeline.index(selected)
        if index > 0:
            pipeline[index - 1], pipeline[index] = pipeline[index], pipeline[index - 1]

    return (
        pipeline,
        render_pipeline_preview(pipeline),
        gr.update(choices=pipeline, value=selected),
    )


def move_technique_down(selected: str | None, pipeline: list[str] | None):
    pipeline = list(pipeline or [])

    if selected in pipeline:
        index = pipeline.index(selected)
        if index < len(pipeline) - 1:
            pipeline[index + 1], pipeline[index] = pipeline[index], pipeline[index + 1]

    return (
        pipeline,
        render_pipeline_preview(pipeline),
        gr.update(choices=pipeline, value=selected),
    )


def remove_technique(selected: str | None, pipeline: list[str] | None):
    pipeline = list(pipeline or [])

    if selected in pipeline:
        pipeline.remove(selected)

    next_value = pipeline[0] if pipeline else None

    return (
        pipeline,
        render_pipeline_preview(pipeline),
        gr.update(choices=pipeline, value=next_value),
    )


def clear_pipeline():
    return (
        [],
        render_pipeline_preview([]),
        gr.update(choices=[], value=None),
    )


def _parse_target_modules(raw_value: str) -> list[str]:
    return [
        item.strip()
        for item in raw_value.replace("\n", ",").split(",")
        if item.strip()
    ]


def _format_percent(value: float) -> str:
    return f"{value:.2f}%"


def _format_accuracy(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_size(value: float) -> str:
    return f"{value:.2f} MB"


def _format_latency(value: float) -> str:
    return f"{value * 1000:.3f} ms/sample"


def _format_throughput(value: float) -> str:
    return f"{value:.2f} samples/s"


def _format_seconds(value: float) -> str:
    return f"{value:.2f} s"


def _format_change(compressed: float, original: float, scale: float = 1.0) -> str:
    delta = (compressed - original) * scale
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f}"


def _build_metrics(
    original_results: dict[str, float],
    compressed_results: dict[str, float],
    pipeline_time: float,
) -> dict[str, str]:
    original_size = original_results["model_size_mb"]
    compressed_size = compressed_results["model_size_mb"]
    compression_pct = (
        (1 - (compressed_size / original_size)) * 100
        if original_size
        else 0.0
    )
    accuracy_change = (
        (compressed_results["accuracy"] - original_results["accuracy"]) * 100
    )
    size_reduction = original_size - compressed_size
    speedup = (
        original_results["avg_latency"] / compressed_results["avg_latency"]
        if compressed_results["avg_latency"]
        else 0.0
    )

    return {
        "Compression %": _format_percent(compression_pct),
        "Accuracy": _format_accuracy(compressed_results["accuracy"]),
        "Accuracy Change %": f"{accuracy_change:+.2f}%",
        "Model Size": _format_size(compressed_size),
        "Model Size Reduction": _format_size(size_reduction),
        "Latency": _format_latency(compressed_results["avg_latency"]),
        "Throughput": _format_throughput(compressed_results["throughput"]),
        "Speedup": f"{speedup:.2f}x",
        "Total Pipeline Time": _format_seconds(pipeline_time),
    }


def _build_comparison_table(
    original_results: dict[str, float],
    compressed_results: dict[str, float],
) -> pd.DataFrame:
    rows = [
        [
            "Accuracy",
            _format_accuracy(original_results["accuracy"]),
            _format_accuracy(compressed_results["accuracy"]),
            f"{(compressed_results['accuracy'] - original_results['accuracy']) * 100:+.2f} pts",
        ],
        [
            "Model Size (MB)",
            f"{original_results['model_size_mb']:.2f}",
            f"{compressed_results['model_size_mb']:.2f}",
            f"{compressed_results['model_size_mb'] - original_results['model_size_mb']:+.2f}",
        ],
        [
            "Avg Latency (ms/sample)",
            f"{original_results['avg_latency'] * 1000:.3f}",
            f"{compressed_results['avg_latency'] * 1000:.3f}",
            f"{(compressed_results['avg_latency'] - original_results['avg_latency']) * 1000:+.3f}",
        ],
        [
            "Throughput (samples/s)",
            f"{original_results['throughput']:.2f}",
            f"{compressed_results['throughput']:.2f}",
            f"{compressed_results['throughput'] - original_results['throughput']:+.2f}",
        ],
        [
            "Total Time (s)",
            f"{original_results['total_time']:.2f}",
            f"{compressed_results['total_time']:.2f}",
            f"{compressed_results['total_time'] - original_results['total_time']:+.2f}",
        ],
    ]

    return pd.DataFrame(
        rows,
        columns=["Metric", "Original", "Compressed", "Change"],
    )


def _capture_report(
    title: str,
    results: dict[str, float],
) -> str:
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        print(title)
        print_report(results)

    return buffer.getvalue().strip()


def _save_model_artifact(model, base_model: str, pipeline: list[str]) -> str:
    safe_model_name = base_model.replace("/", "_")
    safe_pipeline = "_".join(step.lower() for step in pipeline)
    suffix = f"{safe_model_name}_{safe_pipeline or 'compressed'}.pth"

    output_path = Path(tempfile.gettempdir()) / suffix
    torch.save(model.state_dict(), output_path)
    return str(output_path)


def _build_pipeline(
    model,
    learning_rate: float,
    pipeline_order: list[str],
    lora_rank: int,
    lora_alpha: float,
    lora_targets: str,
    student_model_name: str,
    distill_alpha: float,
    distill_temperature: float,
    qat_bits: int,
    ptq_backend: str,
):
    get_bert = _load_model_backend()
    DistillationLoss = _load_distillation_backend()
    pipeline = CompressionPipeline(model)

    for technique in pipeline_order:
        if technique == "LoRA":
            pipeline.add(
                LoRA(
                    rank=int(lora_rank),
                    alpha=float(lora_alpha),
                    target_modules=_parse_target_modules(lora_targets),
                )
            )
            continue

        if technique == "Distillation":
            student_model = get_bert(
                student_model_name,
                num_labels=2,
            )
            optimizer = torch.optim.AdamW(
                student_model.parameters(),
                lr=learning_rate,
            )
            loss_fn = DistillationLoss(
                alpha=float(distill_alpha),
                temperature=float(distill_temperature),
            )
            pipeline.add(
                Distillation(
                    student_model=student_model,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                )
            )
            continue

        if technique == "QAT":
            pipeline.add(
                QAT(
                    num_bits=int(qat_bits),
                )
            )
            continue

        if technique == "PTQ":
            pipeline.add(
                PTQ(
                    # backend=ptq_backend,
                )
            )

    return pipeline



def _run_pipeline_job(
    emit,
    base_model_name: str,
    dataset_name: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: str,
    pipeline_order: list[str],
    lora_rank: int,
    lora_alpha: float,
    lora_targets: str,
    student_model_name: str,
    distill_alpha: float,
    distill_temperature: float,
    qat_bits: int,
    ptq_backend: str,
) -> dict[str, Any]:
    if dataset_name != "glue/sst2":
        raise ValueError("Only GLUE / SST-2 is supported by the current backend.")

    if not pipeline_order:
        raise ValueError("Add at least one technique before running the pipeline.")

    get_dataloaders, get_tokenizer, prepare_sst2 = _load_dataset_backend()
    get_bert = _load_model_backend()

    emit("Loading tokenizer and dataset", 0.08)
    tokenizer = get_tokenizer(base_model_name)
    dataset = prepare_sst2(tokenizer)
    train_loader, val_loader = get_dataloaders(
        dataset,
        batch_size=int(batch_size),
        tokenizer=tokenizer,
    )

    emit("Loading base model", 0.18)
    model = get_bert(base_model_name)
    model.to(device)

    emit("Benchmarking original model", 0.32)
    original_results = benchmark(
        model,
        val_loader,
        device,
    )

    emit("Preparing pipeline context", 0.46)
    trainer = AdaptiveTrainer(
        model=model,
        learning_rate=float(learning_rate),
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    context = PipelineContext(
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        calibration_loader=val_loader,
        device=device,
        epochs=int(epochs),
    )

    emit("Building compression pipeline", 0.58)
    pipeline = _build_pipeline(
        model=model,
        learning_rate=float(learning_rate),
        pipeline_order=pipeline_order,
        lora_rank=int(lora_rank),
        lora_alpha=float(lora_alpha),
        lora_targets=lora_targets,
        student_model_name=student_model_name,
        distill_alpha=float(distill_alpha),
        distill_temperature=float(distill_temperature),
        qat_bits=int(qat_bits),
        ptq_backend=ptq_backend,
    )

    emit("Executing compression pipeline", 0.7)
    pipeline_start = time.perf_counter()
    compressed_model = pipeline.apply(context)
    pipeline_time = time.perf_counter() - pipeline_start

    emit("Benchmarking compressed model", 0.88)
    benchmark_device = str(
    next(compressed_model.parameters()).device
)

    compressed_results = benchmark(
        compressed_model,
        val_loader,
        benchmark_device,
    )

    if compressed_results["model_size_mb"]:
        compressed_results["compression_ratio"] = (
            original_results["model_size_mb"]
            / compressed_results["model_size_mb"]
        )

    emit("Packaging results", 0.96)
    metrics = _build_metrics(
        original_results=original_results,
        compressed_results=compressed_results,
        pipeline_time=pipeline_time,
    )
    comparison_table = _build_comparison_table(
        original_results=original_results,
        compressed_results=compressed_results,
    )
    report = "\n\n".join(
        [
            _capture_report("Original Model", original_results),
            _capture_report("Compressed Model", compressed_results),
        ]
    )
    download_path = _save_model_artifact(
        compressed_model,
        base_model=base_model_name,
        pipeline=pipeline_order,
    )

    emit("Pipeline complete", 1.0)

    return {
        "pipeline_text": " → ".join(pipeline_order),
        "metrics": metrics,
        "comparison_table": comparison_table,
        "report": report,
        "download_path": download_path,
        "original_results": original_results,
        "compressed_results": compressed_results,
    }


def _result_payload(
    progress: float,
    status: str,
    tone: str,
    pipeline_text: str,
    metrics: dict[str, str] | None,
    comparison_table,
    logs: str,
    report: str,
    download_path: str | None,
):
    return (
        render_progress(progress, status, tone=tone),
        render_execution_summary(pipeline_text),
        render_metrics_dashboard(metrics),
        comparison_table,
        logs,
        report,
        download_path,
    )


def run_pipeline(
    base_model_name: str,
    dataset_name: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: str,
    pipeline_order: list[str] | None,
    lora_rank: int,
    lora_alpha: float,
    lora_targets: str,
    student_model_name: str,
    distill_alpha: float,
    distill_temperature: float,
    qat_bits: int,
    ptq_backend: str,
):
    pipeline_order = list(pipeline_order or [])
    event_queue: queue.Queue = queue.Queue()
    log_stream = QueueLogStream(event_queue)
    result_holder: dict[str, Any] = {}
    error_holder: dict[str, str] = {}

    current_progress = 0.0
    current_status = "Queued"
    current_tone = "neutral"
    current_logs = "Initializing pipeline job..."
    current_report = "Benchmark report will appear here."
    current_metrics = None
    current_comparison = pd.DataFrame(
        initial_comparison_rows(),
        columns=["Metric", "Original", "Compressed", "Change"],
    )
    current_pipeline_text = " → ".join(pipeline_order) if pipeline_order else "No pipeline configured."
    current_download = None

    def emit(status: str, progress: float):
        event_queue.put(("status", status, progress))
        print(f"[{int(progress * 100):02d}%] {status}")

    def worker():
        try:
            with redirect_stdout(log_stream), redirect_stderr(log_stream):
                result_holder["value"] = _run_pipeline_job(
                    emit=emit,
                    base_model_name=base_model_name,
                    dataset_name=dataset_name,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    device=device,
                    pipeline_order=pipeline_order,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_targets=lora_targets,
                    student_model_name=student_model_name,
                    distill_alpha=distill_alpha,
                    distill_temperature=distill_temperature,
                    qat_bits=qat_bits,
                    ptq_backend=ptq_backend,
                )
        except Exception:
            error_holder["traceback"] = traceback.format_exc()

    thread = threading.Thread(
        target=worker,
        daemon=True,
    )
    thread.start()

    yield _result_payload(
        progress=current_progress,
        status=current_status,
        tone=current_tone,
        pipeline_text=current_pipeline_text,
        metrics=current_metrics,
        comparison_table=current_comparison,
        logs=current_logs,
        report=current_report,
        download_path=current_download,
    )

    while thread.is_alive() or not event_queue.empty():
        updated = False

        while not event_queue.empty():
            event = event_queue.get_nowait()

            if event[0] == "status":
                _, current_status, current_progress = event
                current_tone = "neutral"
                updated = True
                continue

            if event[0] == "log":
                current_logs = (current_logs + event[1])[-24000:]
                updated = True

        if updated:
            yield _result_payload(
                progress=current_progress,
                status=current_status,
                tone=current_tone,
                pipeline_text=current_pipeline_text,
                metrics=current_metrics,
                comparison_table=current_comparison,
                logs=current_logs,
                report=current_report,
                download_path=current_download,
            )

        time.sleep(0.15)

    if "traceback" in error_holder:
        current_tone = "error"
        current_status = "Pipeline failed"
        current_progress = 1.0
        current_logs = (
            current_logs
            + "\n\n"
            + error_holder["traceback"]
        )[-24000:]
        current_report = "Pipeline execution failed before benchmark results were finalized."

        yield _result_payload(
            progress=current_progress,
            status=current_status,
            tone=current_tone,
            pipeline_text=current_pipeline_text,
            metrics=current_metrics,
            comparison_table=current_comparison,
            logs=current_logs,
            report=current_report,
            download_path=current_download,
        )
        return

    result = result_holder["value"]
    current_tone = "success"
    current_status = "Completed"
    current_progress = 1.0
    current_pipeline_text = result["pipeline_text"]
    current_metrics = result["metrics"]
    current_comparison = result["comparison_table"]
    current_report = result["report"]
    current_download = result["download_path"]

    yield _result_payload(
        progress=current_progress,
        status=current_status,
        tone=current_tone,
        pipeline_text=current_pipeline_text,
        metrics=current_metrics,
        comparison_table=current_comparison,
        logs=current_logs,
        report=current_report,
        download_path=current_download,
    )
