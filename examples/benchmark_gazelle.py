"""Benchmark Gaze-LLE throughput, memory and mixed-precision deviation.

Reports frames/s for ViT-B and ViT-L at several batch sizes on every available
device (CPU, MPS, CUDA), plus how far bf16/fp16 autocast drifts from fp32.

Usage::

    python examples/benchmark_gazelle.py [--people 2] [--repeats 5]
"""

import argparse
import logging
import time

import torch

from exordium.video.face.gaze.gazelle import GazelleWrapper

logger = logging.getLogger("benchmark_gazelle")

VARIANTS = ("vitb14_inout", "vitl14_inout_childplay")
BATCH_SIZES = (1, 4, 8)
BOX = (0.3, 0.2, 0.5, 0.5)


def available_devices() -> list[tuple[str, int | None]]:
    """Return ``(label, device_id)`` for CPU and every accelerator present."""
    devices: list[tuple[str, int | None]] = [("cpu", None)]
    if torch.backends.mps.is_available():
        devices.append(("mps", 0))
    if torch.cuda.is_available():
        devices.append(("cuda", 0))
    return devices


def synchronize(device: torch.device) -> None:
    """Block until queued work on ``device`` finishes, so timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def peak_memory_mb(device: torch.device) -> float | None:
    """Peak allocator memory in MB, where the backend reports it."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return None


def time_model(model: GazelleWrapper, batch: int, people: int, repeats: int) -> float:
    """Median frames per second for one batch size."""
    frames = torch.randint(0, 255, (batch, 3, 720, 1280), dtype=torch.uint8)
    boxes = [[BOX] * people for _ in range(batch)]
    model(frames, boxes)  # warm-up
    times = []
    for _ in range(repeats):
        synchronize(model.device)
        start = time.perf_counter()
        model(frames, boxes)
        synchronize(model.device)
        times.append(time.perf_counter() - start)
    return batch / sorted(times)[len(times) // 2]


def autocast_deviation(variant: str, device_id: int | None) -> float:
    """Max absolute heatmap difference between fp32 and autocast."""
    frames = torch.randint(0, 255, (2, 3, 360, 640), dtype=torch.uint8)
    boxes = [[BOX], [BOX]]
    full = GazelleWrapper(variant, device_id=device_id)(frames, boxes)
    low = GazelleWrapper(variant, device_id=device_id, autocast=True)(frames, boxes)
    return float((full.heatmap - low.heatmap).abs().max())


def main() -> None:
    """Run the benchmark grid and log a table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people", type=int, default=2, help="boxed people per frame")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for label, device_id in available_devices():
        for variant in VARIANTS:
            model = GazelleWrapper(variant, device_id=device_id)
            if device_id == 0 and model.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            rates = {b: time_model(model, b, args.people, args.repeats) for b in BATCH_SIZES}
            memory = peak_memory_mb(model.device)
            deviation = autocast_deviation(variant, device_id)
            fast = GazelleWrapper(variant, device_id=device_id, autocast=True)
            fast_fps = time_model(fast, BATCH_SIZES[1], args.people, args.repeats)
            del fast
            row = "  ".join(f"b{b}={fps:6.1f} fps" for b, fps in rates.items())
            mem = "n/a" if memory is None else f"{memory:,.0f} MB"
            logger.info(
                f"{label:5s} {variant:24s} {row}  mem={mem}  "
                f"autocast b{BATCH_SIZES[1]}={fast_fps:6.1f} fps, max|dHeatmap|={deviation:.4f}"
            )
            del model


if __name__ == "__main__":
    main()
