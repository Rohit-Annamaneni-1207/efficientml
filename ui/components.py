from __future__ import annotations

from typing import Any

import gradio as gr
import torch

APP_THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="slate",
    neutral_hue="slate",
)


MODEL_CHOICES = [
    "textattack/bert-base-uncased-SST-2",
    "distilbert-base-uncased",
]

DATASET_CHOICES = [
    ("GLUE / SST-2", "glue/sst2"),
]

TECHNIQUE_CHOICES = [
    "LoRA",
    "Distillation",
    "QAT",
    "PTQ",
]

TECHNIQUE_DESCRIPTIONS = {
    "LoRA": "Low-rank adaptation for parameter-efficient finetuning.",
    "Distillation": "Train a compact student from the current model as teacher.",
    "QAT": "Inject fake quantization and finetune before conversion.",
    "PTQ": "Post-training quantization with calibration and conversion.",
}

APP_CSS = """
:root {
  --bg-0: #07141c;
  --bg-1: #0f2430;
  --panel: rgba(10, 28, 36, 0.78);
  --panel-strong: rgba(7, 19, 27, 0.92);
  --panel-border: rgba(118, 191, 179, 0.16);
  --text-0: #ecf5f3;
  --text-1: #9eb7b3;
  --accent: #6fe1c6;
  --accent-2: #f2c167;
  --danger: #ff8c78;
  --shadow: 0 24px 80px rgba(0, 0, 0, 0.26);
}

.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(111, 225, 198, 0.16), transparent 28%),
    radial-gradient(circle at top right, rgba(242, 193, 103, 0.16), transparent 22%),
    linear-gradient(180deg, var(--bg-0) 0%, #09111a 100%);
  color: var(--text-0);
  font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
}

.app-shell {
  max-width: 1440px;
  margin: 0 auto;
}

.hero {
  background: linear-gradient(135deg, rgba(9, 30, 38, 0.94), rgba(7, 18, 26, 0.88));
  border: 1px solid rgba(111, 225, 198, 0.14);
  border-radius: 28px;
  box-shadow: var(--shadow);
  overflow: hidden;
  padding: 28px 32px;
  position: relative;
}

.hero::after {
  background: linear-gradient(90deg, transparent, rgba(111, 225, 198, 0.08), transparent);
  content: "";
  height: 1px;
  left: 24px;
  position: absolute;
  right: 24px;
  top: 74px;
}

.hero h1 {
  font-size: 2.4rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  margin: 0;
}

.hero p {
  color: var(--text-1);
  font-size: 1rem;
  margin: 12px 0 0;
  max-width: 820px;
}

.hero-grid {
  display: grid;
  gap: 18px;
  grid-template-columns: 1.7fr 1fr;
}

.mini-stat-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin-top: 18px;
}

.mini-stat {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 18px;
  padding: 14px 16px;
}

.mini-stat-label {
  color: var(--text-1);
  font-size: 0.74rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.mini-stat-value {
  font-size: 1.24rem;
  font-weight: 650;
  margin-top: 8px;
}

.control-panel,
.results-panel {
  background: var(--panel);
  border: 1px solid var(--panel-border);
  border-radius: 24px;
  box-shadow: var(--shadow);
  padding: 18px;
}

.section-title {
  font-size: 0.92rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  margin: 0 0 10px;
  text-transform: uppercase;
}

.section-copy {
  color: var(--text-1);
  font-size: 0.95rem;
  margin: 0 0 18px;
}

.pipeline-preview {
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.02));
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 20px;
  min-height: 92px;
  padding: 18px;
}

.pipeline-empty {
  color: var(--text-1);
  font-size: 0.95rem;
}

.pipeline-steps {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.pipeline-step {
  align-items: center;
  background: rgba(111, 225, 198, 0.08);
  border: 1px solid rgba(111, 225, 198, 0.18);
  border-radius: 999px;
  display: inline-flex;
  gap: 10px;
  padding: 10px 14px;
}

.pipeline-index {
  align-items: center;
  background: rgba(7, 19, 27, 0.9);
  border-radius: 50%;
  display: inline-flex;
  font-size: 0.78rem;
  height: 24px;
  justify-content: center;
  width: 24px;
}

.pipeline-arrow {
  color: var(--accent-2);
  font-size: 1rem;
  padding: 0 2px;
}

.technique-note {
  color: var(--text-1);
  font-size: 0.92rem;
  margin-top: 6px;
}

.metric-grid {
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.metric-card {
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.045), rgba(255, 255, 255, 0.02));
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 20px;
  min-height: 112px;
  padding: 16px;
}

.metric-label {
  color: var(--text-1);
  font-size: 0.8rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.metric-value {
  font-size: 1.42rem;
  font-weight: 700;
  line-height: 1.2;
  margin-top: 12px;
}

.metric-subtext {
  color: var(--text-1);
  font-size: 0.86rem;
  margin-top: 10px;
}

.status-wrap {
  display: grid;
  gap: 12px;
}

.status-row {
  align-items: center;
  display: flex;
  gap: 12px;
  justify-content: space-between;
}

.status-label {
  color: var(--text-1);
  font-size: 0.84rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.status-text {
  font-size: 1rem;
  font-weight: 650;
}

.progress-shell {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 999px;
  height: 10px;
  overflow: hidden;
}

.progress-bar {
  background: linear-gradient(90deg, var(--accent), #9ee2ff);
  border-radius: 999px;
  height: 100%;
  transition: width 0.2s ease-out;
}

.status-badge {
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 700;
  padding: 6px 10px;
}

.status-neutral {
  background: rgba(158, 183, 179, 0.14);
  color: #d3e5e1;
}

.status-success {
  background: rgba(111, 225, 198, 0.14);
  color: #93f0da;
}

.status-error {
  background: rgba(255, 140, 120, 0.14);
  color: #ffb1a1;
}

.summary-card {
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.045), rgba(255, 255, 255, 0.02));
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 20px;
  padding: 16px 18px;
}

.summary-title {
  color: var(--text-1);
  font-size: 0.82rem;
  letter-spacing: 0.06em;
  margin: 0 0 10px;
  text-transform: uppercase;
}

.summary-value {
  font-size: 1.1rem;
  font-weight: 650;
  line-height: 1.4;
}

@media (max-width: 980px) {
  .hero-grid,
  .metric-grid {
    grid-template-columns: 1fr;
  }

  .mini-stat-grid {
    grid-template-columns: 1fr;
  }
}
"""


def available_devices() -> list[str]:
    devices = ["cpu"]

    if torch.cuda.is_available():
        devices.insert(0, "cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and "mps" not in devices
    ):
        devices.insert(0, "mps")

    return devices


def default_device() -> str:
    return available_devices()[0]


def initial_comparison_rows() -> list[list[str]]:
    return [
        ["Accuracy", "-", "-", "-"],
        ["Model Size (MB)", "-", "-", "-"],
        ["Avg Latency (ms/sample)", "-", "-", "-"],
        ["Throughput (samples/s)", "-", "-", "-"],
        ["Total Time (s)", "-", "-", "-"],
    ]


def render_pipeline_preview(techniques: list[str]) -> str:
    if not techniques:
        return """
        <div class="pipeline-preview">
          <div class="pipeline-empty">
            Add compression steps to start building your execution graph.
          </div>
        </div>
        """

    parts = ['<div class="pipeline-preview"><div class="pipeline-steps">']

    for index, technique in enumerate(techniques, start=1):
        parts.append(
            f"""
            <div class="pipeline-step">
              <span class="pipeline-index">{index}</span>
              <span>{technique}</span>
            </div>
            """
        )

        if index < len(techniques):
            parts.append('<span class="pipeline-arrow">→</span>')

    parts.append("</div></div>")
    return "".join(parts)


def render_metrics_dashboard(metrics: dict[str, str] | None = None) -> str:
    placeholders = {
        "Compression %": "-",
        "Accuracy": "-",
        "Accuracy Change %": "-",
        "Model Size": "-",
        "Model Size Reduction": "-",
        "Latency": "-",
        "Throughput": "-",
        "Speedup": "-",
        "Total Pipeline Time": "-",
    }

    metrics = metrics or placeholders

    cards = []

    for label, value in metrics.items():
        cards.append(
            f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
            </div>
            """
        )

    return f'<div class="metric-grid">{"".join(cards)}</div>'


def render_progress(progress: float, status: str, tone: str = "neutral") -> str:
    progress_pct = max(0, min(100, int(progress * 100)))

    return f"""
    <div class="summary-card status-wrap">
      <div class="status-row">
        <div>
          <div class="status-label">Pipeline Status</div>
          <div class="status-text">{status}</div>
        </div>
        <div class="status-badge status-{tone}">{progress_pct}%</div>
      </div>
      <div class="progress-shell">
        <div class="progress-bar" style="width: {progress_pct}%;"></div>
      </div>
    </div>
    """


def render_execution_summary(pipeline_text: str) -> str:
    return f"""
    <div class="summary-card">
      <div class="summary-title">Executed Pipeline</div>
      <div class="summary-value">{pipeline_text}</div>
    </div>
    """


def create_layout() -> tuple[gr.Blocks, dict[str, Any]]:
    with gr.Blocks(
        title="EfficientML Studio",
    ) as demo:
        pipeline_state = gr.State([])

        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
                <section class="hero">
                  <div class="hero-grid">
                    <div>
                      <h1>EfficientML Studio</h1>
                      <p>
                        Compose LoRA, distillation, QAT, and PTQ into a single
                        compression pipeline, then benchmark the result against
                        the original model from one Gradio workspace.
                      </p>
                    </div>
                    <div class="mini-stat-grid">
                      <div class="mini-stat">
                        <div class="mini-stat-label">Pipeline Core</div>
                        <div class="mini-stat-value">CompressionPipeline</div>
                      </div>
                      <div class="mini-stat">
                        <div class="mini-stat-label">Execution</div>
                        <div class="mini-stat-value">Live Streaming</div>
                      </div>
                      <div class="mini-stat">
                        <div class="mini-stat-label">Evaluation</div>
                        <div class="mini-stat-value">Benchmark Reuse</div>
                      </div>
                    </div>
                  </div>
                </section>
                """
            )

            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Column(elem_classes=["control-panel"]):
                        gr.HTML(
                            """
                            <div class="section-title">Experiment Setup</div>
                            <div class="section-copy">
                              Configure the base run, then add techniques and
                              reorder them into the exact pipeline you want.
                            </div>
                            """
                        )

                        base_model = gr.Dropdown(
                            choices=MODEL_CHOICES,
                            value=MODEL_CHOICES[0],
                            label="Base model",
                            allow_custom_value=True,
                        )
                        dataset = gr.Dropdown(
                            choices=DATASET_CHOICES,
                            value="glue/sst2",
                            label="Dataset",
                        )

                        with gr.Row():
                            epochs = gr.Slider(
                                minimum=1,
                                maximum=10,
                                step=1,
                                value=3,
                                label="Epochs",
                            )
                            learning_rate = gr.Number(
                                value=2e-5,
                                label="Learning rate",
                            )

                        with gr.Row():
                            batch_size = gr.Slider(
                                minimum=4,
                                maximum=64,
                                step=4,
                                value=8,
                                label="Batch size",
                            )
                            device = gr.Dropdown(
                                choices=available_devices(),
                                value=default_device(),
                                label="Device",
                            )

                        gr.HTML(
                            """
                            <div class="section-title" style="margin-top: 12px;">
                              Technique Builder
                            </div>
                            <div class="section-copy">
                              Pick a technique, tune its parameters, then add it
                              into the ordered pipeline.
                            </div>
                            """
                        )

                        technique_picker = gr.Dropdown(
                            choices=TECHNIQUE_CHOICES,
                            value="LoRA",
                            label="Selected technique",
                        )
                        technique_description = gr.HTML(
                            f'<div class="technique-note">{TECHNIQUE_DESCRIPTIONS["LoRA"]}</div>'
                        )

                        with gr.Group(visible=True) as lora_group:
                            with gr.Row():
                                lora_rank = gr.Slider(
                                    minimum=1,
                                    maximum=32,
                                    step=1,
                                    value=8,
                                    label="LoRA rank",
                                )
                                lora_alpha = gr.Slider(
                                    minimum=1,
                                    maximum=64,
                                    step=1,
                                    value=16,
                                    label="LoRA alpha",
                                )
                            lora_targets = gr.Textbox(
                                value="query,value,q_lin,v_lin",
                                label="Target modules",
                                lines=2,
                                info="Comma-separated module substrings.",
                            )

                        with gr.Group(visible=False) as distillation_group:
                            student_model = gr.Dropdown(
                                choices=MODEL_CHOICES,
                                value="distilbert-base-uncased",
                                label="Student model",
                                allow_custom_value=True,
                            )
                            with gr.Row():
                                distill_alpha = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.05,
                                    value=0.5,
                                    label="Distillation alpha",
                                )
                                distill_temperature = gr.Slider(
                                    minimum=1.0,
                                    maximum=8.0,
                                    step=0.5,
                                    value=2.0,
                                    label="Temperature",
                                )

                        with gr.Group(visible=False) as qat_group:
                            qat_bits = gr.Dropdown(
                                choices=[4, 8],
                                value=8,
                                label="QAT bit width",
                            )

                        with gr.Group(visible=False) as ptq_group:
                            ptq_backend = gr.Dropdown(
                                choices=["qnnpack", "fbgemm"],
                                value="qnnpack",
                                label="PTQ backend",
                                info="PTQ pipelines execute on cpu in this UI.",
                            )

                        add_button = gr.Button(
                            "Add Technique To Pipeline",
                            variant="secondary",
                        )

                        gr.HTML(
                            """
                            <div class="section-title" style="margin-top: 20px;">
                              Pipeline Composer
                            </div>
                            <div class="section-copy">
                              Select a step and move it until the execution order
                              reflects the compression strategy you want.
                            </div>
                            """
                        )

                        pipeline_preview = gr.HTML(
                            render_pipeline_preview([]),
                        )
                        selected_step = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="Selected pipeline step",
                        )

                        with gr.Row():
                            move_up_button = gr.Button("Move Up")
                            move_down_button = gr.Button("Move Down")
                            remove_button = gr.Button("Remove")
                            clear_button = gr.Button("Clear")

                        run_button = gr.Button(
                            "Run Compression Pipeline",
                            variant="primary",
                        )

                with gr.Column(scale=7):
                    with gr.Column(elem_classes=["results-panel"]):
                        gr.HTML(
                            """
                            <div class="section-title">Results Dashboard</div>
                            <div class="section-copy">
                              Original and compressed metrics are benchmarked side
                              by side after the pipeline finishes.
                            </div>
                            """
                        )

                        progress_panel = gr.HTML(
                            render_progress(0.0, "Idle"),
                        )
                        execution_summary = gr.HTML(
                            render_execution_summary("No pipeline executed yet."),
                        )
                        metrics_dashboard = gr.HTML(
                            render_metrics_dashboard(),
                        )
                        comparison_table = gr.Dataframe(
                            headers=["Metric", "Original", "Compressed", "Change"],
                            value=initial_comparison_rows(),
                            interactive=False,
                            wrap=True,
                            label="Original vs Compressed",
                        )
                        logs_box = gr.Textbox(
                            label="Live logs",
                            lines=18,
                            max_lines=24,
                            autoscroll=True,
                            value="Run the pipeline to stream training and benchmark logs here.",
                        )
                        benchmark_report = gr.Textbox(
                            label="Benchmark report",
                            lines=14,
                            max_lines=18,
                            value="Benchmark report will appear here.",
                            interactive=False,
                        )
                        model_file = gr.File(
                            label="Download compressed model",
                            interactive=False,
                        )

    components = {
        "pipeline_state": pipeline_state,
        "base_model": base_model,
        "dataset": dataset,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "device": device,
        "technique_picker": technique_picker,
        "technique_description": technique_description,
        "lora_group": lora_group,
        "distillation_group": distillation_group,
        "qat_group": qat_group,
        "ptq_group": ptq_group,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_targets": lora_targets,
        "student_model": student_model,
        "distill_alpha": distill_alpha,
        "distill_temperature": distill_temperature,
        "qat_bits": qat_bits,
        "ptq_backend": ptq_backend,
        "add_button": add_button,
        "pipeline_preview": pipeline_preview,
        "selected_step": selected_step,
        "move_up_button": move_up_button,
        "move_down_button": move_down_button,
        "remove_button": remove_button,
        "clear_button": clear_button,
        "run_button": run_button,
        "progress_panel": progress_panel,
        "execution_summary": execution_summary,
        "metrics_dashboard": metrics_dashboard,
        "comparison_table": comparison_table,
        "logs_box": logs_box,
        "benchmark_report": benchmark_report,
        "model_file": model_file,
    }

    return demo, components
