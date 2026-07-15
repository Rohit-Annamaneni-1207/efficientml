# EfficientML - Engineering Notes

This document captures the important architectural decisions, implementation details, design tradeoffs, and lessons learned while building EfficientML.

Its purpose is to document why the code is written the way it is rather than simply explaining what each file contains.

---

# Project Goal

EfficientML is a modular model compression framework for PyTorch models.

The objective is to provide multiple compression techniques behind a common API so that users can experiment with different approaches individually or compose them into compression pipelines.

The framework currently supports:

- LoRA
- Knowledge Distillation
- Post Training Quantization (PTQ)

The long-term goal is to support additional techniques such as:

- Quantization Aware Training
- Structured pruning
- Unstructured pruning
- Low-rank decomposition
- Mixed precision
- Sparsity-aware inference

without requiring changes to the overall pipeline.

---

# Core Philosophy

Every compression algorithm should behave like a plug-in.

Each technique receives the current model and the training context, then returns a modified model without needing knowledge of previous or future compression stages.

This allows arbitrary pipelines such as:

Teacher -> Distillation -> LoRA Fine-tuning -> PTQ -> Benchmark

or:

Model -> LoRA -> Pruning -> QAT -> Export

The pipeline itself contains no compression-specific logic.

# Compression Pipeline Design

Compression techniques inherit from CompressionTechnique, which exposes apply(model, context).

The interface intentionally remains minimal.

The pipeline owns the model.

Each technique transforms the model and returns it.

Example:

```text
model = base_model

for technique in techniques:
    model = technique.apply(model, context)
```

This makes every technique composable.

The pipeline never needs special cases for LoRA, Distillation, PTQ, or future algorithms.

# Pipeline Context

A PipelineContext object is passed to every stage.

It contains:

- train_loader
- validation_loader
- calibration_loader
- trainer
- optimizer
- device
- epochs

This avoids constructors with large parameter lists while allowing future techniques to reuse the same information.

Example:

- PTQ only needs calibration_loader and device.
- LoRA needs trainer, train_loader, and optimizer.
- Distillation needs trainer, train_loader, and epochs.

The context acts as shared state across the pipeline.

# Trainer Synchronization

One important design decision is that compression techniques update the trainer rather than creating new training infrastructure.

Example:

```text
trainer.model = model
```

instead of:

```text
trainer = Trainer(...)
```

Reason:

The UI, benchmarking, checkpoint saving, and progress callbacks already reference the original trainer.

Replacing the trainer would require updating multiple components.

Updating only trainer.model keeps every downstream component synchronized automatically.

This became particularly important for LoRA and Distillation because both produce new model instances.

Whenever a technique replaces the model, it must also update trainer.model to avoid training stale parameters.

# LoRA Implementation

LoRA replaces selected Linear layers with LoRALayer.

Original:

```text
Linear(x)
```

becomes:

```text
Linear(x) + BAx
```

where:

- A: input -> low rank
- B: low rank -> output

Only A and B are trainable.

The original Linear weights remain frozen.

Advantages:

- Very small number of trainable parameters
- Minimal memory usage
- Fast fine-tuning

After training, merge_lora() computes W' = W + BA and replaces the LoRA module with a standard Linear layer.

This removes all inference overhead.

The exported model no longer depends on LoRA modules.

# Optimizer Recreation After LoRA

Injecting LoRA changes which parameters require gradients.

The original optimizer still references the old parameter list.

Therefore the optimizer must be recreated after LoRA injection.

The implementation recreates the optimizer using type(trainer.optimizer) and copies only compatible defaults.

One important issue encountered was AdamW containing internal fields such as decoupled_weight_decay that are not accepted by the constructor.

Instead of forwarding optimizer.defaults directly, unsupported arguments are filtered before reconstruction.

Otherwise PyTorch raises TypeError for an unexpected keyword argument.

# Knowledge Distillation

Distillation trains a smaller student model using predictions from a larger teacher.

Teacher:

- frozen
- evaluation mode
- gradients disabled

Student:

- trainable

Only the student is returned by the compression stage.

The teacher exists solely during training.

This keeps deployment lightweight while transferring knowledge from the larger model.

# Post Training Quantization

Initially static PTQ was implemented.

Pipeline:

```text
prepare()
->
calibration
->
convert()
```

Calibration used the validation loader.

Embedding layers required float_qparams_weight_only_qconfig.

LayerNorm and Dropout were excluded from quantization.

## Static PTQ Problems

Although conversion succeeded, forward inference failed.

Errors included quantized::layer_norm and later quantized::linear.

These errors were traced to Apple Silicon backend limitations rather than incorrect implementation.

The quantized operators exist primarily for QuantizedCPU backends.

PyTorch currently provides incomplete support for static transformer quantization on macOS.

## Dynamic PTQ

The project switched to dynamic quantization.

Implementation:

```python
quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)
```

Advantages:

- No calibration
- Simpler implementation
- Significant model size reduction

Observed results:

- Accuracy: 92.43% -> 91.86%
- Model size: 417 MB -> 173 MB

However, latency increased substantially on Apple Silicon.

The slowdown is caused by backend limitations rather than the quantization algorithm itself.

Dynamic PTQ is heavily optimized for Intel CPUs (FBGEMM) but not Apple ARM processors.

# Important Lessons

Building model compression involves more than implementing algorithms.

Practical engineering challenges included:

- optimizer reconstruction
- trainer synchronization
- backend compatibility
- calibration datasets
- module-specific qconfigs
- deployment constraints

One major lesson is that a mathematically correct compression algorithm does not guarantee faster inference.

Runtime performance depends heavily on:

- hardware
- kernel implementations
- backend support
- operator coverage

Compression should therefore always be evaluated using:

- accuracy
- model size
- latency
- throughput

rather than assuming improvements.

# SOME MATHEMATICAL FORMULATIONS AND BACKGROUND

# LoRA Mathematical Foundation

## Motivation

Fine-tuning every parameter of a modern language model is expensive.

For a Linear layer:

```text
y = Wx
```

where W ∈ R^(d_out x d_in) contains millions of trainable parameters.

Research has shown that during fine-tuning, the weight update ΔW is approximately low-rank.

Instead of learning W', we decompose:

```text
ΔW = BA
```

where:

- A ∈ R^(r x d_in)
- B ∈ R^(d_out x r)
- r << d

Therefore:

```text
W' = W + BA
```

The number of trainable parameters becomes r(d_in + d_out) instead of d_in x d_out.

For example, 4096 x 4096 requires 16.7 million parameters.

Using r = 8 requires only 65,536 parameters.

This is more than 250x fewer trainable parameters.

## Forward Pass

Original:

```text
y = Wx
```

LoRA:

```text
y = Wx + BAx
```

The original Linear layer remains frozen.

Only A and B receive gradients.

During inference, merge_lora() computes W <- W + BA so no additional computation remains.

# LoRA Initialization

LoRA intentionally initializes A and B differently.

- A: random initialization (Kaiming Uniform)
- B: all zeros

Reason:

Initially, BA = 0.

Therefore W' = W.

The injected LoRA layer behaves identically to the original model.

Training therefore starts from the pretrained model rather than a perturbed model.

After the first optimizer step, B becomes non-zero.

Gradients then flow through both A and B.

If both matrices were initialized randomly, the model output would immediately change, causing a large accuracy drop before training even begins.

If both matrices were initialized to zero, the gradients would remain symmetric, preventing effective learning.

The asymmetric initialization solves both problems.

# LoRA Scaling

The LoRA update is ΔW = BA.

In practice, the update is scaled as:

```text
ΔW = α/r · BA
```

where α is the scaling factor.

Reasons:

1. The update magnitude remains approximately independent of the chosen rank.
2. Changing rank does not require retuning learning rates.
3. The optimizer behaves consistently for different LoRA configurations.

Typical values:

- rank = 8
- alpha = 16

This gives scale = 2.

Increasing alpha increases the contribution of the LoRA update.

Decreasing alpha keeps the pretrained weights dominant.

# Post Training Quantization

## Goal

Reduce model size and inference cost without retraining.

A floating point tensor W is converted into int8 values.

Instead of storing 32 bits per weight, only 8 bits are stored.

Memory usage decreases by approximately 4x.

## Linear Quantization

Given floating-point weights W, compute scale and zero point such that:

```text
q = round(W / scale) + zero_point
```

where q is an int8 value.

The inverse transformation is:

```text
W ≈ scale(q - zero_point)
```

Quantization therefore approximates the original tensor rather than storing it exactly.

## Calibration

Static PTQ also quantizes activations.

To determine suitable activation ranges, the model is run on representative data.

Observers record minimum, maximum, or histogram statistics.

These statistics determine scale and zero_point for activation tensors.

Without calibration, activations may saturate, leading to severe accuracy degradation.

# Static vs Dynamic PTQ

Static PTQ:

- Quantizes weights and activations
- Requires prepare(), calibration, and convert()
- Faster inference
- Better deployment
- Requires calibration
- Backend support is limited

Dynamic PTQ:

- Quantizes only weights
- Activations remain floating point until inference
- Each Linear layer quantizes activations dynamically
- No calibration required
- Very easy to apply
- Minimal accuracy loss
- Smaller speedup
- Runtime quantization overhead

# Embedding Quantization

Embedding layers perform table lookups rather than matrix multiplication.

The standard Linear qconfig assumes symmetric int8 quantization.

Embedding weights require float_qparams_weight_only_qconfig, which uses floating-point zero points.

Without this qconfig, embedding quantization fails or produces poor accuracy.

Therefore every Embedding module is assigned a custom qconfig before calling prepare().

# LayerNorm Exclusion

LayerNorm computes (x - μ) / σ, which is highly sensitive to numerical precision.

Transformer models execute LayerNorm repeatedly.

PyTorch's support for quantized LayerNorm is backend dependent.

During development, quantizing LayerNorm produced runtime failures on Apple Silicon.

Therefore LayerNorm.qconfig = None, allowing LayerNorm to remain floating point while Linear layers were quantized.

This mixed-precision approach is common in transformer quantization.

# Knowledge Distillation

Knowledge distillation trains a student model using the outputs of a larger teacher.

Instead of learning only from ground truth labels, the student also learns from teacher logits.

Teacher probabilities: P_t

Student probabilities: P_s

The total loss is:

```text
L = α L_CE + (1 - α) T² KL(P_t || P_s)
```

where T is the temperature.

Higher temperatures soften the probability distribution, revealing similarities between classes.

The student therefore learns richer information than hard labels alone.
