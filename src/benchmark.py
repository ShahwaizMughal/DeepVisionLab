"""Inference latency and throughput benchmarking.

Methodology: a single warm-up batch is run first (to trigger lazy CUDA
kernel compilation/caching), then `torch.cuda.synchronize()` is called
immediately before and after the timed region so that the elapsed wall
clock time reflects actual GPU compute rather than asynchronous kernel
queuing. Falls back gracefully to CPU-only timing when CUDA is unavailable.
"""

import time

import torch


def measure_inference(model, loader, device, num_batches: int = 100):
    """Measure average per-image latency (ms) and throughput (images/sec).

    Args:
        model: A torch.nn.Module already moved to `device`, in eval mode
            (this function also calls model.eval() defensively).
        loader: DataLoader to draw batches from.
        device: torch.device the model lives on.
        num_batches: Number of batches to time (after the warm-up pass).

    Returns:
        (latency_ms_per_image, throughput_images_per_sec)
    """
    model.eval()

    total_images = 0

    # Warm-up pass, not included in the timed region.
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            _ = model(images)
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break

            images = images.to(device)
            _ = model(images)

            total_images += images.size(0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    latency_ms = (elapsed / total_images) * 1000
    throughput = total_images / elapsed

    return latency_ms, throughput
