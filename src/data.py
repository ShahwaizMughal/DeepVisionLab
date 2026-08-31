"""CIFAR-10 data loading, preprocessing, and augmentation.

Reflects the notebook's three data pipelines:
    1. Augmented pipeline for the baseline CNN (32x32 images).
    2. Non-augmented pipeline for the baseline CNN ablation (32x32 images).
    3. Transfer-learning pipeline for ResNet-18, which resizes images to
       224x224 and normalizes with the statistics tied to the pretrained
       ImageNet weights.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def get_baseline_transforms(augment: bool):
    """Return (train_transform, test_transform) for the baseline CNN.

    Args:
        augment: If True, apply random horizontal flip and random crop to
            the training transform (the notebook's "with augmentation" run).
            If False, only normalize (the "without augmentation" ablation).
    """
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )

    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            ]
        )
    else:
        train_transform = test_transform

    return train_transform, test_transform


def get_transfer_transforms():
    """Return (train_transform, test_transform) for ResNet-18 (224x224 input).

    Uses the mean/std tied to ResNet18_Weights.DEFAULT so pixel statistics
    match what the pretrained backbone was trained on.
    """
    weights = ResNet18_Weights.DEFAULT
    weight_mean = weights.transforms().mean
    weight_std = weights.transforms().std

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=weight_mean, std=weight_std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=weight_mean, std=weight_std),
        ]
    )

    return train_transform, test_transform


def get_cifar10_loaders(
    train_transform,
    test_transform,
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
):
    """Build CIFAR-10 train/test DataLoaders for a given pair of transforms.

    Args:
        train_transform: torchvision transform applied to the training split.
        test_transform: torchvision transform applied to the test split.
        data_root: Directory to download/cache CIFAR-10 in.
        batch_size: Batch size for both loaders.
        num_workers: DataLoader worker processes.

    Returns:
        (train_loader, test_loader)
    """
    train_dataset = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
