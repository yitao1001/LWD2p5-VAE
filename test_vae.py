#!/usr/bin/env python3
"""Evaluate a saved LWD VAE on untouched test and extrapolation splits.

Run ``python test_vae.py --help``. Reconstructions use the posterior mean.
This script never selects checkpoints or updates model weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from lwd_vae.dataset import LWDVAEDataset
from lwd_vae.evaluation import reconstruction_metrics, summarize_metrics, unrestricted_denormalize


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="generated dataset directory with manifest.json")
    parser.add_argument("--checkpoint", type=Path, required=True, help="trusted local checkpoint from train_vae.py")
    parser.add_argument("--output", type=Path, required=True, help="new directory; existing directories are refused")
    parser.add_argument("--splits", nargs="+", default=None, choices=["val", "test", "ood_dip", "ood_contrast"],
                        help="Default: test and available OOD splits from manifest; ood_dip is legacy-only")
    parser.add_argument("--device", default="auto", help="auto/cpu/mps/cuda/cuda:N")
    parser.add_argument("--batch-size", type=_positive, default=32)
    parser.add_argument("--workers", type=_nonnegative, default=0)
    parser.add_argument("--max-samples", type=_positive, help="evaluate first N samples of each split (smoke checks only)")
    parser.add_argument("--save-reconstructions", action="store_true", help="save raw log10rho float64 [N,1,nz,ny] arrays")
    parser.add_argument("--no-plots", action="store_true", help="omit reconstruction PNG galleries")
    return parser


def _gallery_indices(metadata: list[dict[str, Any]], count: int, limit: int = 9) -> list[int]:
    selected: list[int] = []
    families: set[str] = set()
    for index, record in enumerate(metadata[:count]):
        family = str(record.get("family", "unknown"))
        if family not in families:
            selected.append(index)
            families.add(family)
        if len(selected) == limit:
            return sorted(selected)
    for index in np.linspace(0, count - 1, min(count, limit), dtype=int).tolist():
        if index not in selected:
            selected.append(index)
        if len(selected) == limit:
            break
    return sorted(selected)


def _plot_gallery(
    images: dict[int, tuple[np.ndarray, np.ndarray]],
    records: list[dict[str, Any]],
    dataset: LWDVAEDataset,
    destination: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    indices = sorted(images)
    grid = dataset.config["grid"]
    extent = [*grid["y_bounds_ft"], grid["z_bounds_ft"][1], grid["z_bounds_ft"][0]]
    lo, hi = dataset.log10_bounds
    norm = Normalize(lo, hi)
    max_error = max(float(np.max(np.abs(predicted - true))) for true, predicted in images.values())
    error_norm = Normalize(-max(max_error, 1e-8), max(max_error, 1e-8))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("#051b44")
    cmap.set_over("#ff00a8")
    fig, axes = plt.subplots(len(indices), 3, figsize=(15, 2.65 * len(indices)), squeeze=False, constrained_layout=True)
    physical_image = error_image = None
    for row_index, index in enumerate(indices):
        truth, predicted = images[index]
        record = records[index]
        for column, values in enumerate((truth, predicted, predicted - truth)):
            ax = axes[row_index, column]
            image = ax.imshow(values, origin="upper", extent=extent, aspect="equal", interpolation="nearest", cmap="RdBu_r" if column == 2 else cmap, norm=error_norm if column == 2 else norm)
            ax.set_xlabel("y (ft)")
            ax.set_ylabel("z (ft, down)")
            if column == 0:
                physical_image = image
                ax.set_title(f"Truth | {record['sample_id']} | {record['family']}", fontsize=9)
            elif column == 1:
                ax.set_title(f"Posterior mean | RMSE={record['rmse_log10']:.4f}", fontsize=9)
            else:
                error_image = image
                ax.set_title(f"Prediction - truth | outside bounds={record['out_of_bounds_fraction']:.2%}", fontsize=9)
    fig.colorbar(physical_image, ax=axes[:, :2].ravel().tolist(), location="bottom", fraction=0.025, pad=0.02, extend="both", label="log10(rho / (ohm m)); shared dataset scale")
    fig.colorbar(error_image, ax=axes[:, 2].ravel().tolist(), location="bottom", fraction=0.025, pad=0.02, label="log10 resistivity error; shared gallery scale")
    fig.suptitle(f"{dataset.split}: deterministic VAE reconstructions (one example per available family)", fontsize=12)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def evaluation_splits(dataset_root: Path, requested: list[str] | None) -> list[str]:
    """新数据默认 test/ood_contrast；旧数据仍可评估已有的 ood_dip。"""
    manifest = json.loads((Path(dataset_root).expanduser() / "manifest.json").read_text())
    available = manifest["splits"]
    selected = list(requested) if requested is not None else [
        name for name in ("test", "ood_dip", "ood_contrast")
        if name in available and available[name]["count"] > 0]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select at least one split, without duplicates")
    missing = [name for name in selected if name not in available or available[name]["count"] == 0]
    if missing:
        raise ValueError(f"Requested splits are absent or empty: {missing}")
    return selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, Subset

    from lwd_vae.runtime import load_checkpoint, select_device, validate_data_contract, sha256_file

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation output: {output}")
    selected_splits = evaluation_splits(args.dataset, args.splits)
    device = select_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint.expanduser().resolve(), device=device)
    torch.set_num_threads(checkpoint["training_config"].get("cpu_threads", 4))
    datasets: dict[str, LWDVAEDataset] = {}
    try:
        for split in selected_splits:
            dataset = LWDVAEDataset(args.dataset, split=split, as_torch=True)
            datasets[split] = dataset
            validate_data_contract(checkpoint["data_contract"], dataset, require_manifest=False)
            if not len(dataset):
                raise ValueError(f"cannot evaluate empty split: {split}")
            _ = dataset.metadata
        if not args.no_plots:
            import matplotlib  # noqa: F401 -- fail before creating output if missing
        output.mkdir(parents=True, exist_ok=False)
        model.eval()
        results: dict[str, Any] = {
            "schema_version": 1,
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_best_val_rmse": checkpoint.get("best_val_rmse"),
            "dataset": str(args.dataset.expanduser().resolve()),
            "dataset_manifest_sha256": hashlib.sha256((args.dataset.expanduser() / "manifest.json").read_bytes()).hexdigest(),
            "device": str(device),
            "reconstruction": "decode(encode(x).mode()); no posterior sampling; no clipping",
            "max_samples_per_split": args.max_samples,
            "metric_definitions": {
                "rmse_log10": "sqrt(mean squared error over all cells and samples); log10(ohm m)",
                "mae_log10": "mean absolute error over all cells and samples; log10(ohm m)",
                "edge_mae_log10_per_cell": "mean absolute error of adjacent-cell log10 differences, pooled over both axes; not a derivative per ft",
                "mean_relative_rho_l2": "mean of per-sample ||rho_pred-rho_true||_2 / ||rho_true||_2; null if any ratio overflows float64",
                "log10_relative_rho_l2": "log10 of per-sample relative rho L2; null for exact zero error",
                "out_of_bounds_fraction": "fraction of reconstructed log10rho pixels outside dataset normalization bounds; no clipping applied",
                "rho_exponent_overflow_pixels": "prediction pixels whose exponent 10**log10rho exceeds float64; ratio itself is computed in log space",
                "rho_exponent_underflow_pixels": "prediction pixels below the log10 of float64 smallest subnormal",
                "grouping": "equal samples on a common grid; group and split metrics are not averages of batch means",
            },
            "splits": {},
        }
        for split, dataset in datasets.items():
            count = min(len(dataset), args.max_samples) if args.max_samples is not None else len(dataset)
            subset = Subset(dataset, range(count))
            loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
            destination = output / split
            destination.mkdir()
            saved = None
            if args.save_reconstructions:
                saved = np.lib.format.open_memmap(destination / "reconstructed_log10rho.npy", mode="w+", dtype=np.float64, shape=(count, *dataset.image_shape))
            gallery_indices = set(_gallery_indices(dataset.metadata, count)) if not args.no_plots else set()
            gallery: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            rows: list[dict[str, Any]] = []
            start = 0
            with torch.inference_mode():
                for batch in loader:
                    batch = batch.to(device)
                    reconstruction = model.decode(model.encode(batch).mode())
                    truth = unrestricted_denormalize(batch.detach().cpu().numpy(), dataset.log10_bounds)
                    prediction = unrestricted_denormalize(reconstruction.detach().cpu().numpy(), dataset.log10_bounds)
                    batch_rows = reconstruction_metrics(truth, prediction, dataset.log10_bounds)
                    for local_index, metric in enumerate(batch_rows):
                        index = start + local_index
                        metadata = dataset.metadata[index]
                        row = {"index": index, "sample_id": metadata.get("sample_id", f"{split}:{index:08d}"), "family": metadata.get("family", "unknown"), "contrast_class": metadata.get("parameters", {}).get("contrast_class", "unknown"), **metric}
                        rows.append(row)
                        if index in gallery_indices:
                            gallery[index] = (truth[local_index, 0].copy(), prediction[local_index, 0].copy())
                    if saved is not None:
                        saved[start : start + len(batch_rows)] = prediction
                    start += len(batch_rows)
            if saved is not None:
                saved.flush()
                del saved
            with (destination / "sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summary = summarize_metrics(rows)
            summary["evaluated_count"] = count
            summary["available_count"] = len(dataset)
            summary["partial_split"] = count < len(dataset)
            (destination / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            if gallery:
                _plot_gallery(gallery, rows, dataset, destination / "reconstructions.png")
            results["splits"][split] = summary
            overall = summary["overall"]
            print(f"{split}: N={count}, RMSE(log10rho)={overall['rmse_log10']:.6g}, MAE={overall['mae_log10']:.6g}, outside={overall['out_of_bounds_fraction']:.2%}", flush=True)
        (output / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Evaluation saved to {output}", flush=True)
        return results
    finally:
        for dataset in datasets.values():
            dataset.close()


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
