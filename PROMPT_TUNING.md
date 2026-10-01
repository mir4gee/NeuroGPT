# Class-Conditional Soft Prompt-Tuning for NeuroGPT

Two implementations live in this repo side by side:

- **The actual, published approach** — `scripts/finetune.sh` / `src/train_gpt.py`, completely unmodified vendor code. Reproduced at `experiments/run_actual_baseline.sh` for convenience, using the repo's own default downstream recipe (`--ft-only-encoder=True`).
- **Our novelty** — class-conditional soft prompt-tuning, implemented in `src/prompt_tuning/` and run via `experiments/run_novelty_prompt_tuning.sh`.

Both scripts use the same BCI-IV-2a fold and the same core hyperparameters, so a single pair of runs gives a direct comparison.

## The novelty

> Apply soft/continuous prompt-tuning — learn a small set of virtual prompt tokens prepended to the frozen EEG sequence, one learned set per mental-state condition, steering the frozen model's internal representations.

A naive reading of "one learned set per mental-state condition" would pick the prompt using the *true label* — that leaks the classification answer and isn't real prompt-tuning. Instead:

- Learn a **fixed menu** of soft-prompt tokens: one contiguous block per condition, shape `(num_conditions, num_tokens_per_condition, embed_dim)`.
- On **every** forward pass — training and inference alike — prepend the **entire flattened menu** to `inputs_embeds`, and extend `attention_mask` with that many leading 1s. The same menu is shown for every example; nothing about the true label is injected.
- **Freeze the whole backbone** (encoder, embedder, decoder's GPT2 transformer) and train only: the soft-prompt menu + the existing `pooler_layer` + `decoding_head`. The frozen self-attention layers must learn to route/attend to the relevant condition's block based on the actual EEG content that follows — this is the "steering the frozen model's internal representations" mechanism, generalized from single-prompt NLP tuning to a multi-condition menu for a discriminative task.
- `--num-prompt-tokens-per-condition 0` degenerates exactly to a frozen-backbone linear-probe baseline — a free, apples-to-apples ablation using the same code path.

## Files added (no vendored files touched)

- `src/prompt_tuning/soft_prompt.py` — `ConditionSoftPrompt` (the menu + prepend logic), `PromptTunedModel` (subclasses `src/model.py::Model`, overrides `forward()` to route through the soft prompt), `freeze_for_prompt_tuning()`.
- `src/prompt_tuning/train_prompt_tuning.py` — reuses `train_gpt.py`'s argument parser, dataset construction, and HF `Trainer` plumbing (`make_trainer`) unmodified; only the model constructor differs.
- `tests/test_soft_prompt.py` — smoke test using a tiny random `GPT2Config` decoder and fabricated tensors (no real data or the multi-hundred-MB pretrained checkpoint required, since neither is available in this sandbox). Run: `python tests/test_soft_prompt.py`.
- `experiments/run_actual_baseline.sh`, `experiments/run_novelty_prompt_tuning.sh` — the two comparable run configs.

## Dataset note

BCI-IV-2a's 4 motor-imagery classes stand in for "mental-state conditions" here — no focused/mind-wandering EEG dataset ships with NeuroGPT (only TUH for pretraining and BCI-IV-2a for fine-tuning). The mechanism is dataset-agnostic: swapping in a real attention-state dataset only requires a different `Dataset` class and `--num-decoding-classes`.

## Running it

Both scripts assume the released pretrained checkpoint is at `pretrained_model/pytorch_model.bin` (download from the [HuggingFace repo](https://huggingface.co/wenhuic/Neuro-GPT)) and BCI-IV-2a data at `../bci2a_egg_npz/` (sibling to this repo), matching the vendor's own `scripts/finetune.sh` convention. Neither the checkpoint nor the dataset is available in this environment — download them and run from your own GPU/Colab:

```bash
cd experiments
./run_actual_baseline.sh          # the paper's own fine-tuning approach
./run_novelty_prompt_tuning.sh    # our soft prompt-tuning novelty
```

Leave the original `--freeze-*` flags at their defaults (`False`) when using `train_prompt_tuning.py` — `--freeze-backbone` (default `True`) does the freezing instead, uniformly across encoder/embedder/decoder-transformer.

## Verification performed here

No GPU, pretrained checkpoint, or BCI-IV-2a data are available in this sandbox, so full training wasn't run. What *was* verified: `tests/test_soft_prompt.py` passes — the soft-prompt menu prepends the correct shapes, a full forward+backward pass runs in decoding mode, `num_tokens_per_condition=0` degenerates to a no-op, and freezing leaves only the prompt menu + pooler + decoding head trainable. Real training/comparison is left to the commands above on your own hardware.
