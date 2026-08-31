"""Training loop with per-epoch loss/accuracy tracking, as used for all
four experiments (baseline CNN with/without augmentation, ResNet-18
from scratch, ResNet-18 transfer learning).
"""

import time

import torch
import torch.nn as nn


def train_model(
    model,
    train_loader,
    test_loader,
    device,
    epochs: int = 10,
    learning_rate: float = 0.001,
):
    """Train `model` and evaluate on `test_loader` after every epoch.

    Uses cross-entropy loss and the Adam optimizer, matching the notebook's
    experiments. Returns a history dict with per-epoch train/test loss and
    accuracy, plus prints progress each epoch (as in the notebook).

    Args:
        model: A torch.nn.Module already moved to `device`.
        train_loader: Training DataLoader.
        test_loader: Test/validation DataLoader.
        device: torch.device to run on.
        epochs: Number of training epochs.
        learning_rate: Adam learning rate.

    Returns:
        dict with keys "train_loss", "train_acc", "test_loss", "test_acc",
        each a list of per-epoch values.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    start_time = time.time()

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_loss /= len(test_loader)
        test_acc = 100 * correct / total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Test Loss: {test_loss:.4f} "
            f"Test Acc: {test_acc:.2f}%"
        )

    total_time = time.time() - start_time
    print(f"\nTraining time: {total_time:.2f} seconds")

    return history
