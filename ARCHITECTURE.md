# EfficientML Architecture

## Purpose

EfficientML is a modular model compression framework built around a pipeline architecture.

The central idea is that every compression technique behaves as an independent transformation:

```

Model → Technique → Model

```

Rather than implementing compression algorithms as standalone scripts, every technique exposes a common interface so they can be composed into arbitrary pipelines.

Current techniques include

- LoRA
- Knowledge Distillation
- Post Training Quantization (PTQ)

Future techniques such as pruning, QAT and low-rank decomposition can be added without modifying the pipeline itself.

---

## High-Level Architecture

```

                  UI
                   │
                   ▼
          Pipeline Builder
                   │
                   ▼
         Compression Pipeline
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
      LoRA   Distillation    PTQ
        │          │          │
        └──────────┼──────────┘
                   ▼
          Compressed Model
                   │
          Benchmark / Export

```

The pipeline owns the model and sequentially applies every selected compression technique.

# Directory Structure

```

efficientml/

│

├── compression/
│ ├── base.py
│ ├── pipeline.py
│ ├── context.py
│ └── techniques/
│ ├── lora.py
│ ├── distillation.py
│ └── ptq.py
│
├── lora/
│ ├── inject.py
│ ├── merge.py
│ ├── layers.py
│ └── utils.py
│
├── distillation/
│ └── trainer.py
│
├── quant/
│ └── ptq.py
│
├── core/
│ └── trainer.py
│
├── eval/
│ └── benchmark.py
│
├── ui/
│ └── callbacks.py
│
└── models/

```

Responsibilities

compression/

Contains the framework itself.

Defines the pipeline, shared interfaces and compression techniques.

---

lora/

Contains only LoRA-specific implementation.

No knowledge of pipelines or UI exists here.

---

quant/

Contains quantization implementations.

Independent from the compression framework.

---

core/

Contains the generic Trainer.

Compression techniques reuse this trainer instead of implementing their own training loops.

---

eval/

Responsible only for evaluation.

Never modifies models.

---

ui/

Collects user input, builds pipelines and launches execution.

Contains no compression logic.

# CompressionTechnique

Every algorithm inherits from

CompressionTechnique

which defines

```python
apply(model, context)
```

Arguments

model

Current model produced by the previous pipeline stage.

context

Shared pipeline state containing

- trainer
- train loader
- validation loader
- calibration loader
- optimizer
- epochs
- device

Return

A modified model.

No technique communicates directly with another technique.

All communication occurs through the returned model.

# Compression Pipeline

Pipeline execution is intentionally simple.

```python
model = base_model

for technique in techniques:
    model = technique.apply(model, context)

return model
```

Each stage receives the output of the previous stage.

Example

```

Original Model
      │
      ▼
 LoRA Injection
      │
      ▼
 LoRA Fine-tuning
      │
      ▼
 Distillation
      │
      ▼
 Dynamic PTQ
      │
      ▼
 Benchmark

```

This design makes every compression algorithm composable.

# PipelineContext

Instead of passing many arguments into every technique, a shared context object is used.

Typical members

```python
context.model
context.device
context.train_loader
context.val_loader
context.calibration_loader
context.trainer
context.optimizer
context.epochs
```

Advantages

- Single source of truth
- Easier extension
- Cleaner interfaces
- New techniques automatically gain access to required resources

Different techniques consume different subsets of the context.

Example

PTQ

- calibration_loader
- device

LoRA

- trainer
- train_loader
- epochs

Distillation

- trainer
- device
- optimizer

# Model Ownership

The pipeline owns the model.

Compression techniques never permanently store models internally.

Instead

```

Pipeline
│
owns
│
▼
Current Model

```

Each technique

receives

```

Model

```

returns

```

Modified Model

```

This prevents synchronization problems and keeps the pipeline stateless.

# Trainer Synchronization

Several compression techniques create a new model instance.

Examples

- LoRA injects custom layers
- Distillation returns the student network

The Trainer originally references the old model.

Therefore each technique updates

```python
trainer.model = model
```

after modifying the architecture.

The trainer itself is never recreated.

Only its model reference changes.

This ensures

- optimizer
- UI callbacks
- benchmarking
- checkpointing

continue operating on the latest model.

# LoRA Execution Flow

```

Base Model
     │
     ▼
Inject LoRA Layers
     │
     ▼
Freeze Base Weights
     │
     ▼
Enable LoRA Parameters
     │
     ▼
Reset Optimizer
     │
     ▼
Fine-tune
     │
     ▼
Merge LoRA
     │
     ▼
Standard Transformer

```

The merged model contains no LoRA layers.

Inference therefore incurs no additional runtime cost.

# Distillation Execution Flow

```

Teacher
│
│ Frozen
│
▼

Student

│

▼

Distillation Trainer

│

▼

Updated Student

```

Only the student is returned to the pipeline.

The teacher exists only during training.

# Static PTQ Execution Flow

Static Post Training Quantization follows a three-stage workflow.

```

Floating Point Model
        │
        ▼
 Deep Copy Model
        │
        ▼
Assign qconfigs
        │
        ▼
prepare()
        │
        ▼
Insert Observers
        │
        ▼
Calibration
        │
        ▼
Collect Activation Statistics
        │
        ▼
convert()
        │
        ▼
Replace Float Modules
        │
        ▼
INT8 Model

```

## Step 1 — Prepare

The original model is first copied to avoid modifying the source model.

The quantization backend (e.g. QNNPACK) is selected and every quantizable module is assigned a qconfig.

Special handling is required for certain modules.

- Linear layers receive the default qconfig.
- Embedding layers receive `float_qparams_weight_only_qconfig`.
- LayerNorm is excluded from quantization.
- Dropout is excluded since it is inactive during inference.

Calling `prepare()` traverses the model and inserts observer modules.

These observers record activation statistics during calibration but do not yet modify the model weights.

---

## Step 2 — Calibration

The prepared model is executed on representative calibration data.

No gradients are computed.

The purpose is **not training** but collecting activation distributions.

Observers record information such as

- minimum values
- maximum values
- histograms

These statistics are later used to determine quantization parameters including

- scale
- zero point

The calibration dataset should closely resemble the expected inference data.

---

## Step 3 — Convert

After calibration, `convert()` replaces floating-point modules with quantized implementations.

Examples include

- nn.Linear → QuantizedLinear
- nn.Embedding → QuantizedEmbedding

Observer modules are removed.

The recorded calibration statistics are converted into quantization parameters and embedded into the model.

The final model is ready for INT8 inference.

---

## Implementation Notes

Although the implementation correctly completed all three stages,

```
prepare()
↓

calibrate()

↓

convert()
```

runtime inference failed on Apple Silicon because PyTorch does not provide complete backend support for transformer static quantization.

The implementation itself was architecturally correct, but backend limitations prevented successful deployment.

For this reason, the project later adopted Dynamic PTQ as the production implementation while retaining the Static PTQ implementation for educational purposes.

# PTQ Execution Flow

Current implementation uses Dynamic Post Training Quantization.

```

Floating Point Model

│

▼

Copy Model

│

▼

Move to CPU

│

▼

Dynamic Quantization

│

▼

INT8 Linear Layers

│

▼

Benchmark

```

Only Linear layers are quantized.

Embedding layers, LayerNorm, Softmax, GELU and other transformer operations remain floating point.

This minimizes implementation complexity while preserving accuracy.


# Quantization Aware Training (QAT) Execution Flow

Unlike PTQ, QAT performs quantization during training.

```

Floating Point Model
        │
        ▼
Deep Copy Model
        │
        ▼
Assign qconfigs
        │
        ▼
prepare_qat()
        │
        ▼
Insert Fake Quantization Modules
        │
        ▼
Fine-tuning
        │
        ▼
convert()
        │
        ▼
INT8 Model

```

## Step 1 — Prepare

The model is switched into training mode.

A backend qconfig is assigned to every quantizable module.

`prepare_qat()` inserts

- FakeQuantize modules
- Observers

instead of immediately replacing modules.

The model remains entirely floating point.

---

## Step 2 — QAT Training

Training proceeds normally.

However,

every forward pass contains simulated quantization.

Weights are

```
float

↓

fake quantize

↓

dequantize

↓

forward
```

Activations follow the same process.

Gradients are still computed in floating point.

This allows the optimizer to learn weights that are naturally robust to quantization error.

---

## Step 3 — Convert

After training,

`convert()`

replaces fake quantization modules with actual quantized operators.

The exported model performs real INT8 inference.

---

## Advantages

Compared to PTQ,

QAT usually

- preserves more accuracy
- adapts to quantization noise
- produces better deployment models


# Benchmark Pipeline

Every compressed model is evaluated using the same benchmark.

Metrics include

- Accuracy
- Model Size
- Latency
- Throughput
- Total Evaluation Time

Benchmarking occurs after every compression pipeline completes.

This allows direct comparison between

Original Model

↓

Compressed Model

using identical evaluation data.

# Design Principles

The architecture follows several guiding principles.

## Separation of Concerns

Algorithms

should not contain UI logic.

UI

should not contain compression logic.

Benchmarking

should not modify models.

---

## Composability

Compression techniques should be stackable.

Any valid ordering should execute without requiring pipeline changes.

---

## Extensibility

Adding a new compression algorithm should require

1. Implement CompressionTechnique

2. Register the technique

No pipeline modifications should be necessary.

---

## Minimal Shared State

Shared information is stored only inside PipelineContext.

Everything else is passed explicitly.

---

## Stateless Techniques

Compression techniques should not permanently store models.

They operate only on the model received by apply() and return the transformed result.

This keeps the framework deterministic and easy to debug.