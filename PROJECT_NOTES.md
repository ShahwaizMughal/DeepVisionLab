# Project Notes

This file documents how `DeepVisionLab_Experiments.ipynb` (the original, single-file Colab notebook) was turned into this repository. It is not part of the main README because it describes the packaging process, not the research itself.

## What was extracted from the original notebook

- **`src/data.py`** — the CIFAR-10 transform pipelines (baseline, augmented baseline, and ResNet-18 transfer-learning transforms) and `DataLoader` construction, generalized into reusable functions with parameters (batch size, augment on/off) instead of separately duplicated cells.
- **`src/models.py`** — the `BaselineCNN` class and the ResNet-18 construction logic (`resnet18(weights=...)` + replaced final layer), unified into one `build_resnet18(pretrained=...)` function since the notebook's "from scratch" and "pretrained" cells were otherwise identical except for the `weights` argument.
- **`src/train.py`** — the `train_model` function, copied with no logic changes (same optimizer, loss, and per-epoch tracking).
- **`src/evaluate.py`** — the `evaluate_model` function (renamed `get_predictions` for clarity) plus the classification-report and confusion-matrix plotting code, which were inline in the notebook.
- **`src/benchmark.py`** — the `measure_inference` function, copied with no logic changes.

## What was cleaned

- Removed one empty markdown cell that had no content.
- Reorganized the notebook's cells (no code or outputs removed) under 14 numbered Markdown sections (Project Overview through Conclusion) so the notebook reads as a structured report rather than a linear scratchpad.
- Added Discussion, Limitations, and Conclusion sections to the notebook itself, mirroring the README, since the original notebook ended right after the final benchmark print statement with no written interpretation.

## What was newly generated

- **`results/figures/`** — eight PNG plots (training curves, augmentation ablation comparison, from-scratch-vs-pretrained comparison, accuracy comparison bar chart, accuracy-vs-parameters, accuracy-vs-latency, throughput comparison) regenerated from the exact per-epoch numbers printed in the notebook's executed cell outputs. No epoch values or metrics were invented; where the notebook did not benchmark a model, that combination is simply absent from the relevant figures/tables.
- **`results/metrics/model_comparison.csv`** and **`results/metrics/training_history.csv`** — tabular versions of the same transcribed numbers.
- **`README.md`**, **`requirements.txt`**, **`.gitignore`**, **`LICENSE`** (MIT) — standard repository scaffolding.
- Reproducibility notes in the README, clarifying that no fixed random seed was used in the original runs, so exact numeric reproduction is not guaranteed even though the code is unchanged.

## Limitations in reproducibility

- The original notebook was run on Google Colab with a single Tesla T4 GPU; no random seed was set, so re-running the notebook or `src/` modules will reproduce the same trends but not bit-identical accuracy numbers.
- Inference latency/throughput were only benchmarked for the baseline CNN and the pretrained ResNet-18 in the original notebook (both models happened to be held in the `model` / `resnet` variables used by the benchmarking cell). The from-scratch ResNet-18 was not separately re-benchmarked, so its latency/throughput are not reported as measured values anywhere in this repository.
