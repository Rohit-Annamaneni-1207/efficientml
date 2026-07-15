# EfficientML

EfficientML is a Gradio-based model-compression and benchmarking workspace for SST-2 sentiment classification. It lets you compose an ordered pipeline of techniques such as LoRA, knowledge distillation, post-training quantization, and quantization-aware training, run that pipeline on a language model, and compare the compressed result against the original model with live metrics and reports.

The main idea is simple: pick a base model, choose a dataset, configure one or more compression techniques, arrange them in the order you want, then run the pipeline and inspect the benchmark results. The repository is centered around the interactive UI, but it also includes smaller example scripts for common combinations.

## What This Project Does

- Builds a sequential compression pipeline from reusable techniques.
- Applies LoRA, distillation, PTQ, and QAT to SST-2 classification models.
- Benchmarks the original and compressed model on accuracy, latency, throughput, and model size.
- Streams logs, progress, and a benchmark report through a Gradio UI.
- Saves a compressed model artifact that can be downloaded after the run finishes.

## How It Works

1. Select a base model and dataset.
2. Configure technique-specific settings such as LoRA rank, distillation temperature, or quantization options.
3. Add techniques to the pipeline and reorder them if needed.
4. Run the pipeline to train, compress, and evaluate the model.
5. Review the comparison table, metrics dashboard, streamed logs, and benchmark report.

The project is classification-focused today and is built around SST-2. The evaluation path measures logits-based accuracy and timing statistics, so the current workflow is best understood as a practical compression and benchmarking sandbox rather than a general-purpose training framework.

## Installation

Create a Python environment, then install the dependencies used by the project in `requirements.txt`.

```bash
python3 -m venv venv
pip install -r requirements.txt
```

## Run The Examples

The `examples/` folder contains small entrypoints that show the pipeline in action:

```bash
python examples/lora.py
python examples/lora_ptq.py
python examples/lora_distill_ptq.py
```

These scripts demonstrate three common workflows:

- LoRA-only compression.
- LoRA followed by PTQ.
- LoRA, distillation, and PTQ chained together.

## Launch The UI

The interactive workspace lives in `ui/app.py`:

```bash
python ui/app.py
```

The UI lets you configure the base run, add techniques into an ordered pipeline, move or remove steps, and then execute the full compression-and-benchmark flow from one screen.

## Repository Layout

- `compression/` contains the pipeline abstraction and technique exports.
- `compression/techniques/` implements LoRA, distillation, PTQ, and QAT.
- `core/` provides shared training and configuration utilities.
- `data/` prepares SST-2 data and tokenization.
- `distillation/` holds the distillation loss and trainer logic.
- `eval/` computes benchmark metrics and formats reports.
- `lora/` and `quant/` contain lower-level injection and quantization helpers.
- `models/` provides model constructors.
- `ui/` implements the Gradio application.
- `examples/` contains the runnable end-to-end scripts.
- `demo/` stores screenshots and result artifacts for the README demo.

## Outputs

After a successful run, the UI produces:

- A live progress indicator.
- Streamed training and benchmark logs.
- An executed-pipeline summary.
- A metrics dashboard.
- An original-versus-compressed comparison table.
- A benchmark report with accuracy, model size, latency, throughput, and compression ratio.
- A downloadable compressed model artifact.

## Demo

The screenshots below show the interactive UI and a completed benchmark run.

### UI Setup And Pipeline Builder

![EfficientML setup screen](demo/ui/Screenshot%202026-07-15%20at%206.59.05%E2%80%AFPM.png)

![EfficientML technique configuration](demo/ui/Screenshot%202026-07-15%20at%206.59.18%E2%80%AFPM.png)

![EfficientML pipeline composer](demo/ui/Screenshot%202026-07-15%20at%206.59.30%E2%80%AFPM.png)

![EfficientML results panel](demo/ui/Screenshot%202026-07-15%20at%207.00.49%E2%80%AFPM.png)

### Completed Run

![EfficientML benchmark results](demo/results/Screenshot%202026-07-15%20at%207.40.50%E2%80%AFPM.png)

## Notes

- The current workflow is centered on SST-2 text classification.
- The UI expects a non-empty pipeline before execution.
- PTQ and QAT controls are available in the interface, but the project is still a focused research-and-experimentation workspace rather than a fully generalized compression platform.
