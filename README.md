# AGCLR

The code base is the official implementation of Why Limit the Residual Stream to Layers and Not Tokens? Persistent Memory for Continuous Latent Reasoning.

AGCLR (Adaptive Gated Continuous Latent Reasoning) extends CoCoNuT-style continuous latent reasoning with a Gated Concept Stream: a persistent memory carried across reasoning passes via read, write, and forget gates. It addresses the concept bottleneck observed in vanilla CoCoNuT, where performance degrades as more latent reasoning steps are added.

## Getting Started

This codebase runs as a single notebook (`hotpotQA_agclr.ipynb`), executed cell by cell rather than via a CLI entry point. Each major stage (environment setup, data, model, training, evaluation, ablations) is its own cell or group of cells.

Setup environment:

```bash
pip install transformers==4.46.2 datasets==3.1.0 tqdm==4.67.0 torch
```

Dependency versions above match the ones pinned for the paper's experiments. A single GPU (A100/GH200-class, ~40GB+) is sufficient; the notebook does not use multi-GPU/torchrun.

## Data

HotpotQA is loaded directly via the `datasets` library (`hotpot_qa`) and subsampled for efficient iteration. Each example is preprocessed into a question, multi-hop supporting-fact context, and answer, then tokenized with GPT-2's tokenizer plus three added special tokens:

```python
SPECIAL_TOKENS = {
    "start_latent": "<|start-latent|>",
    "end_latent":   "<|end-latent|>",
    "latent":       "<|latent|>"
}
```

## Configuration

Each training run is configured with a plain Python dict wrapped in a small `HotpotQAConfig` class (no yaml). There are three configs per dataset, one per stage of the pipeline: CoT baseline, vanilla CoCoNuT, and AGCLR.

**General settings**

- `project`: wandb project name
- `name`: run name (e.g. `hotpotqa-cot-baseline`, `hotpotqa-agclr-enhanced`)
- `dataset_name`: HuggingFace dataset identifier (`hotpot_qa`)
- `save_path`: checkpoint directory for the run

**Method flags**

- `cot`: train the plain Chain-of-Thought baseline
- `coconut`: enable CoCoNuT-style latent reasoning (multi-pass forward with latent token substitution)
- `agclr`: enable the Gated Concept Stream on top of CoCoNuT (AGCLR builds on `coconut: True`)

**Training settings**

- `model_id`: Huggingface model id used as initialization, e.g. `openai-community/gpt2`
- `batch_size_training`: per-GPU batch size (32 in the paper's runs)
- `gradient_accumulation_steps`: gradient accumulation steps (4, giving an effective batch size of 64)
- `num_epochs`: total epochs for the run (15 for CoT, CoCoNuT, and AGCLR)
- `lr`: learning rate (1e-4)
- `weight_decay`: weight decay (0.01)
- `warmup_steps`: linear warmup steps (100)
- `seed`: random seed (42)
- `bf16`: whether to use bf16 training
- `max_length`: max tokenized sequence length (1024, to fit HotpotQA's multi-hop context)
- `train_samples` / `val_samples` / `eval_samples`: subsampled dataset sizes used for faster iteration (15,000 / 3,000 / 500)

**CoCoNuT / AGCLR curriculum settings**

- `c_thought`: number of continuous latent thoughts per reasoning step (2)
- `epochs_per_stage`: epochs spent at each curriculum stage before advancing (3)
- `final_stage_epochs`: epochs spent at the final, fully-latent stage (6)
- `max_latent_stage`: highest curriculum stage reachable (3, i.e. stages 0–3)
- `reset_optimizer`: whether the optimizer is reset when advancing to a new curriculum stage (True)

The curriculum maps epoch ranges to latent-token counts: stage 0 (epochs 1–3, 0 latent tokens) → stage 1 (epochs 4–6, 2 latent tokens) → stage 2 (epochs 7–9, 4 latent tokens) → stage 3 (epochs 10–15, 6 latent tokens). Stage 3 is where the concept bottleneck appears in vanilla CoCoNuT, and where AGCLR's gating is intended to help.

**Gated Concept Stream settings (AGCLR only)**

- `gate_init_read`: initial bias for the read gate before sigmoid (0.0 → σ ≈ 0.43; pass-1 reads zeros for free, later passes read real hop-1 facts)
- `gate_init_forget`: initial bias for the forget gate before sigmoid (−1.0 → σ ≈ 0.27; conservative, keeps 73% of the hidden state early on)
- `gate_init_write`: initial bias for the write gate before sigmoid (−1.5 → σ ≈ 0.18; very conservative, avoids flooding the residual stream before gates are trained)

Gate weights are zero-initialized (gates start input-independent, bias-only) and the gates learn input-dependence during training. The concept stream itself is reset to zero at the start of every forward pass and carried across passes within that call.

## Model Architecture

`GatedConceptStream` implements the core memory mechanism:

```
ĥ_t  = LayerNorm(h_t)
r_t  = σ(W_r · ĥ_t)
f_t  = σ(W_f · ĥ_t)
w_t  = σ(W_w · ĥ_t)
h'_t = (1 - f_t) ⊙ h_t  +  r_t ⊙ c_{t-1}
c_t  = LayerNorm(c_{t-1} + w_t ⊙ h'_t)
```

`AGCLR` wraps a base causal LM (GPT-2 or Llama-family) the same way vanilla CoCoNuT does — locating latent token positions, running iterative multi-pass forward calls with KV-cache reuse, and substituting latent token embeddings with hidden states from the previous pass — but routes each pass's hidden state through `GatedConceptStream` first, so the concept stream persists across all reasoning passes instead of being discarded. On GPT-2 (124M params), the gates add 1,774,848 parameters (1.41% of the total).

## Training

There's no single launch command; you run the relevant cells for each stage in order:

1. **CoT baseline** — trains a standard Chain-of-Thought GPT-2 on HotpotQA (used both as a baseline and as the initialization checkpoint for CoCoNuT/AGCLR).
2. **Vanilla CoCoNuT** — initializes from the CoT checkpoint, then runs the 4-stage latent curriculum to demonstrate the concept bottleneck.
3. **AGCLR** — same curriculum and hyperparameters as vanilla CoCoNuT, with the Gated Concept Stream enabled, to demonstrate the fix.

Checkpoints are saved per epoch (e.g. `checkpoint_epoch_15.pt`) and include the model state dict, epoch, curriculum stage, train/val loss, accuracy, and the three gate values at save time.

## Evaluation

Evaluation runs greedy generation on a held-out sample of the validation set (typically 500 examples) and reports Exact Match (EM) and F1. Results from a controlled comparison at matched settings (500 samples, GPT-2, stage 3 / 6 latent tokens):

| Model | EM | F1 |
|---|---|---|
| CoT baseline | 11.0% | 15.5% |
| Vanilla CoCoNuT | 10.2% | 15.6% |
| **AGCLR (ours)** | **13.2%** | **19.0%** |

## Ablations

Two kinds of ablations are implemented directly on the trained AGCLR model:

**Gate knockouts** — disable a single gate by freezing its bias to −10 (σ ≈ 0, effectively off) and setting `requires_grad = False` on that gate's weight and bias before training, then training and evaluating as usual:

- AGCLR w/o read gate
- AGCLR w/o write gate
- AGCLR w/o forget gate

**Write-freeze ablation** — on an already-trained AGCLR model, the write gate is frozen (forced to 0) after a chosen pass `k`, so the concept stream can still be read from but no longer written to for passes `k+1` onward. This isolates whether AGCLR's gains come from persistent storage of early-pass information versus the extra gate parameters themselves. In our 500-sample run, freezing the write gate after pass 1 or pass 2 retained 100% of full AGCLR's EM/F1, supporting the persistent-storage explanation over a parameter-count explanation.

## Citation

If you use this code base in your research, please cite our paper with the following BibTex entry:

```bibtex
@article{farhan2026agclr,
  title={Why Limit the Residual Stream to Layers and Not Tokens? Persistent Memory for Continuous Latent Reasoning},
  author={Farhan, Mujtaba and Chaudhary, Maheep},
  journal={EIML Workshop, ICML},
  year={2026}
}
```

## License

This code is released under the MIT license (see LICENSE).
