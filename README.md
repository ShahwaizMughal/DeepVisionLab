# Deep Learning-Based Image Classification: Transfer Learning and Computational Efficiency Analysis

## Overview

This project investigates image classification on CIFAR-10 by comparing a custom-built convolutional neural network against ResNet-18 under two training regimes (from scratch and via ImageNet transfer learning), and by isolating the effect of data augmentation on the custom CNN. Beyond predictive accuracy, the project measures inference latency and throughput for each model to characterize the accuracy-versus-efficiency trade-off that matters in practical deployment settings.

The full experimental workflow — data exploration, model implementation, training, evaluation, and benchmarking — is implemented in [`notebooks/DeepVisionLab_Experiments.ipynb`](notebooks/DeepVisionLab_Experiments.ipynb), with reusable components extracted into the [`src/`](src/) package.

## Research Questions

1. How does a deeper residual architecture (ResNet-18) compare with a shallow custom CNN on CIFAR-10?
2. How much does ImageNet pretraining improve performance over training the same architecture from scratch?
3. What is the effect of data augmentation on the baseline CNN under the tested setup?
4. What trade-off exists between predictive performance and computational efficiency (latency/throughput) across architectures?

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60,000 32x32 color images across 10 balanced classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), split into 50,000 training and 10,000 test images. It is loaded via `torchvision.datasets.CIFAR10` and downloaded automatically on first run.

## Methodology

**Preprocessing.** Images are normalized channel-wise using CIFAR-10 statistics (mean `(0.4914, 0.4822, 0.4465)`, std `(0.2470, 0.2435, 0.2616)`) for the baseline CNN pipelines, and resized to 224x224 with ImageNet normalization statistics (from `ResNet18_Weights.DEFAULT`) for the ResNet-18 pipelines.

**Baseline CNN.** A compact 3-block architecture (Conv-BatchNorm-ReLU-MaxPool x3 with 32/64/128 channels) followed by a fully connected classifier with dropout — 620,810 trainable parameters. Trained for 10 epochs with the Adam optimizer (lr=0.001).

**ResNet-18 from scratch.** The standard `torchvision` ResNet-18 architecture (11,181,642 trainable parameters) with randomly initialized weights and a replaced final layer for 10-way classification. Trained for 5 epochs with the Adam optimizer (lr=0.001).

**ResNet-18 transfer learning.** The same architecture initialized from ImageNet weights, fine-tuned for 5 epochs with a lower Adam learning rate (lr=0.0001), appropriate for adapting pretrained features rather than learning from random initialization.

**Data augmentation ablation.** The baseline CNN is trained both with (random horizontal flip + random crop) and without training-time augmentation, holding architecture, optimizer, learning rate, and epoch count fixed, to isolate the effect of augmentation alone.

**Evaluation.** Per-class precision, recall, and F1-score (via `scikit-learn`'s `classification_report`) and a confusion matrix are computed for the best-performing model.

**Computational benchmarking.** Inference latency (ms/image) and throughput (images/sec) are measured with a GPU warm-up pass followed by a timed region bounded by `torch.cuda.synchronize()` calls, avoiding measurement of asynchronous kernel queuing rather than actual compute time.

## Experimental Results

| Model | Best Test Accuracy | Parameters | Epochs | Learning Rate | Latency (ms/img) | Throughput (img/s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline CNN (with augmentation) | 75.46% | 620,810 | 10 | 0.001 | 0.364 | 2749.08 |
| Baseline CNN (without augmentation) | 77.56% | 620,810 | 10 | 0.001 | — | — |
| ResNet-18 (from scratch) | 82.58% | 11,181,642 | 5 | 0.001 | — | — |
| ResNet-18 (ImageNet pretrained) | 94.54% | 11,181,642 | 5 | 0.0001 | 1.716 | 582.60 |

Latency and throughput were benchmarked for the baseline CNN and the pretrained ResNet-18 only; see [Limitations](#limitations). Full per-epoch histories are in [`results/metrics/training_history.csv`](results/metrics/training_history.csv), and the table above is reproduced in [`results/metrics/model_comparison.csv`](results/metrics/model_comparison.csv). Supporting figures are in [`results/figures/`](results/figures/).

## Discussion

**Architecture.** ResNet-18 outperforms the shallow baseline CNN under every training regime, consistent with the benefit of residual connections and depth for this task, at the cost of roughly 18x more parameters.

**Transfer learning.** ImageNet pretraining provides a substantial improvement over training the identical ResNet-18 architecture from scratch (94.54% vs. 82.58%), despite using the same number of epochs and a lower learning rate. This suggests ImageNet features transfer well to CIFAR-10 despite the domain gap in image resolution and content.

**Data augmentation.** Under this specific setup, the augmented baseline CNN (75.46%) performed slightly worse than the non-augmented one (77.56%) after 10 epochs. This is a genuine ablation result rather than an error: with a shallow network and a modest training budget, augmentation may add variance the model hasn't had enough epochs to fully exploit — augmentation's benefits typically emerge more clearly over longer schedules or when a model is otherwise overfitting, neither of which strongly applies here.

**Accuracy-efficiency trade-off.** The baseline CNN is about 4.7x faster per image (and has proportionally higher throughput) than ResNet-18, but trails it by 17-19 accuracy points. This trade-off would need to be weighed against the latency and accuracy requirements of a target deployment scenario.

## Limitations

- **Dataset scope:** results are specific to CIFAR-10 (32x32 images, 10 balanced classes) and may not generalize to higher-resolution or class-imbalanced datasets.
- **Training budget:** the baseline CNN used 10 epochs; both ResNet-18 variants used 5 epochs. Longer training could shift the relative ranking, particularly for the augmentation ablation.
- **No hyperparameter search:** learning rates, batch sizes, and optimizer settings were fixed per experiment rather than tuned, so results reflect one configuration per model rather than each model's best achievable accuracy.
- **Augmentation scope:** only random horizontal flip and random crop (baseline CNN) or flip and rotation (ResNet-18 transfer) were tested; stronger augmentation strategies (e.g., Cutout, MixUp, AutoAugment) were not evaluated.
- **Benchmarking hardware:** latency/throughput figures are specific to a single Tesla T4 GPU (Google Colab) at the batch sizes used during training (128 for the baseline CNN, 64 for ResNet-18); results will differ on other hardware or batch sizes.
- **Partial benchmarking:** inference latency/throughput were measured for the baseline CNN and the pretrained ResNet-18 only. The from-scratch ResNet-18 shares the identical architecture and parameter count as the pretrained one, so its inference cost should be equivalent, but this was not independently measured.

## Reproducibility

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd DeepVisionLab
pip install -r requirements.txt
```

Run the notebook directly:

```bash
jupyter notebook notebooks/DeepVisionLab_Experiments.ipynb
```

Or use the extracted modules in `src/` directly, e.g.:

```python
import torch
from src.data import get_baseline_transforms, get_cifar10_loaders
from src.models import BaselineCNN, count_parameters
from src.train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_tf, test_tf = get_baseline_transforms(augment=True)
train_loader, test_loader = get_cifar10_loaders(train_tf, test_tf)

model = BaselineCNN().to(device)
print("Parameters:", count_parameters(model))

history = train_model(model, train_loader, test_loader, device, epochs=10, learning_rate=0.001)
```

The original experiment configuration (epochs, learning rates, batch sizes, transforms) is preserved as-is in both the notebook and `src/`. A fixed random seed was not used in the original experiments, so exact reproduction of the reported accuracies is not guaranteed — re-running the code will reproduce the same overall trends but not necessarily identical numbers.

## Repository Structure

```text
DeepVisionLab/
│
├── notebooks/
│   └── DeepVisionLab_Experiments.ipynb   # Full experimental workflow
│
├── src/
│   ├── __init__.py
│   ├── data.py                            # CIFAR-10 loading, transforms, DataLoaders
│   ├── models.py                          # BaselineCNN, ResNet-18 builder
│   ├── train.py                           # Training/validation loop
│   ├── evaluate.py                        # Predictions, classification report, confusion matrix
│   └── benchmark.py                       # Inference latency/throughput measurement
│
├── results/
│   ├── figures/                           # Generated plots (PNG)
│   └── metrics/                           # model_comparison.csv, training_history.csv
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Technologies

Python, PyTorch, Torchvision, NumPy, Pandas, Scikit-learn, Matplotlib, CIFAR-10, CNN, ResNet-18, Transfer Learning.

## Research Relevance

This project demonstrates practical experience in deep learning, computer vision, experimental design (controlled ablations, from-scratch vs. pretrained comparisons), model evaluation (per-class metrics, confusion matrices), and computational efficiency analysis (latency/throughput benchmarking) — skills directly applicable to applied ML research.
