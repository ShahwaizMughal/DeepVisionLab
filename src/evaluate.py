"""Evaluation helpers: predictions, classification report, confusion matrix.

Only exposes what the notebook actually computes (precision/recall/F1 via
scikit-learn's classification_report, plus a confusion matrix) rather than
a broader metrics suite that was never run.
"""

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def get_predictions(model, loader, device):
    """Run inference over `loader` and return (y_true, y_pred) as numpy arrays."""
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_predictions)


def print_classification_report(y_true, y_pred, class_names):
    """Print per-class precision/recall/F1 and overall accuracy."""
    print(classification_report(y_true, y_pred, target_names=class_names))


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", ax=None):
    """Plot a confusion matrix for the given predictions.

    Args:
        y_true, y_pred: Arrays of true/predicted class indices.
        class_names: Ordered class label strings.
        title: Plot title.
        ax: Optional matplotlib Axes to draw on.

    Returns:
        The matplotlib Axes the confusion matrix was drawn on.
    """
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    disp.plot(ax=ax, xticks_rotation=45)
    ax.set_title(title)

    return ax
