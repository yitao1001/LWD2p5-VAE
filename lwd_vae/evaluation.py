"""Deterministic VAE reconstruction metrics in physical log-resistivity units.

Metrics never clip decoded images. Relative resistivity errors are evaluated
in log space so a bad decoder cannot silently overflow ``10 ** log10rho``.
The edge metric uses adjacent-cell differences, not derivatives per foot.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def unrestricted_denormalize(values: Any, log10_bounds: Any) -> np.ndarray:
    """Affine decoder output -> log10(ohm m), preserving out-of-range values."""
    bounds = np.asarray(log10_bounds, dtype=np.float64)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError("log10_bounds must contain two finite, increasing values")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("decoder output contains non-finite values")
    result = bounds[0] + (array + 1.0) * ((bounds[1] - bounds[0]) / 2.0)
    if not np.isfinite(result).all():
        raise ValueError("denormalized decoder output contains non-finite values")
    return result


def _log_norm(log_abs: np.ndarray) -> float:
    """Log Euclidean norm from elementwise log absolute values."""
    maximum = float(np.max(log_abs))
    if maximum == -math.inf:
        return -math.inf
    return maximum + 0.5 * math.log(float(np.sum(np.exp(2.0 * (log_abs - maximum)))))


def _relative_rho_l2(truth: np.ndarray, prediction: np.ndarray) -> tuple[float | None, float | None, bool]:
    ln10 = math.log(10.0)
    truth_ln, pred_ln = truth * ln10, prediction * ln10
    absolute_delta = np.abs(pred_ln - truth_ln)
    # |exp(a)-exp(b)| = exp(max(a,b)) * (1-exp(-|a-b|)).
    with np.errstate(divide="ignore", under="ignore"):
        log_difference = np.maximum(truth_ln, pred_ln) + np.log(-np.expm1(-absolute_delta))
    log_numerator = _log_norm(log_difference)
    if log_numerator == -math.inf:
        return 0.0, None, False
    log_relative = log_numerator - _log_norm(truth_ln)
    log10_relative = log_relative / ln10
    if log_relative > math.log(np.finfo(np.float64).max):
        return None, float(log10_relative), True
    return float(math.exp(log_relative)), float(log10_relative), False


def reconstruction_metrics(
    truth: Any, prediction: Any, log10_bounds: Any
) -> list[dict[str, Any]]:
    """Return one record per raw ``[N,1,nz,ny]`` log10-resistivity image.

    ``rmse_log10`` and ``mae_log10`` use all cells. The edge metric averages
    errors in adjacent differences over all horizontal and vertical edges.
    ``relative_rho_l2`` = ||rho_pred-rho_true||_2 / ||rho_true||_2. A ratio
    outside float64 is recorded as null with its finite log10 value and an
    overflow flag. Identity has relative error 0 and log10 error null.
    """
    actual, predicted = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if actual.shape != predicted.shape or actual.ndim != 4 or actual.shape[1] != 1 or 0 in actual.shape:
        raise ValueError("truth and prediction must have matching nonempty [N,1,nz,ny] shapes")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("truth and prediction must contain only finite values")
    bounds = np.asarray(log10_bounds, dtype=np.float64)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError("log10_bounds must contain two finite, increasing values")
    maximum_log10 = math.log10(np.finfo(np.float64).max)
    minimum_log10 = math.log10(float(np.nextafter(np.float64(0), np.float64(1))))
    rows = []
    for image, reconstructed in zip(actual[:, 0], predicted[:, 0]):
        delta = reconstructed - image
        # Scaling avoids overflow when squaring a very poor finite prediction.
        scale = float(np.max(np.abs(delta)))
        rmse = scale * float(np.sqrt(np.mean((delta / scale) ** 2))) if scale else 0.0
        mae = float(np.mean(np.abs(delta)))
        edge_sum, edge_count = 0.0, 0
        for axis in (0, 1):
            errors = np.abs(np.diff(reconstructed, axis=axis) - np.diff(image, axis=axis))
            edge_sum += float(np.sum(errors))
            edge_count += errors.size
        ratio, log_ratio, ratio_overflow = _relative_rho_l2(image, reconstructed)
        rows.append({
            "rmse_log10": rmse,
            "mae_log10": mae,
            "edge_mae_log10_per_cell": edge_sum / edge_count if edge_count else 0.0,
            "relative_rho_l2": ratio,
            "log10_relative_rho_l2": log_ratio,
            "relative_rho_l2_overflow": ratio_overflow,
            "out_of_bounds_fraction": float(np.mean((reconstructed < bounds[0]) | (reconstructed > bounds[1]))),
            "rho_exponent_overflow_pixels": int(np.count_nonzero(reconstructed > maximum_log10)),
            "rho_exponent_underflow_pixels": int(np.count_nonzero(reconstructed < minimum_log10)),
        })
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    rmses = np.asarray([row["rmse_log10"] for row in rows], dtype=np.float64)
    scale = float(np.max(rmses))
    # All dataset images share a grid: equal-sample MSE = pooled-cell MSE.
    result: dict[str, Any] = {
        "count": len(rows),
        "rmse_log10": scale * float(np.sqrt(np.mean((rmses / scale) ** 2))) if scale else 0.0,
        "mean_sample_rmse_log10": float(np.mean(rmses)),
    }
    for key in ("mae_log10", "edge_mae_log10_per_cell", "out_of_bounds_fraction"):
        result[key] = float(np.mean([row[key] for row in rows]))
    ratios = [row["relative_rho_l2"] for row in rows if row["relative_rho_l2"] is not None]
    overflow_count = sum(bool(row["relative_rho_l2_overflow"]) for row in rows)
    # Averages covering overflow samples would be misleading. Keep the complete
    # mean null and give finite-only information under an explicit name.
    finite_mean = math.fsum(value / len(ratios) for value in ratios) if ratios else None
    result["mean_relative_rho_l2"] = finite_mean if not overflow_count else None
    result["mean_finite_relative_rho_l2"] = finite_mean
    result["relative_rho_l2_finite_count"] = len(ratios)
    result["relative_rho_l2_overflow_count"] = overflow_count
    for key in ("rho_exponent_overflow_pixels", "rho_exponent_underflow_pixels"):
        result[key] = sum(int(row[key]) for row in rows)
    return result


def summarize_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate samples and metadata groups without averaging batch means."""
    records = list(rows)
    result: dict[str, Any] = {"overall": _aggregate(records)}
    for field in ("family", "contrast_class"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            groups[str(row.get(field, "unknown"))].append(row)
        result[f"by_{field}"] = {name: _aggregate(group) for name, group in sorted(groups.items())}
    return result


def evaluate_reconstruction(model: Any, loader: Any, device: Any, log10_bounds: Any) -> dict[str, Any]:
    """Validation helper; deterministic posterior mean, restoring model mode."""
    import torch

    was_training = model.training
    rows: list[dict[str, Any]] = []
    model.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device)
                reconstruction = model.decode(model.encode(batch).mode())
                truth = unrestricted_denormalize(batch.detach().cpu().numpy(), log10_bounds)
                predicted = unrestricted_denormalize(reconstruction.detach().cpu().numpy(), log10_bounds)
                rows.extend(reconstruction_metrics(truth, predicted, log10_bounds))
    finally:
        model.train(was_training)
    if not rows:
        raise ValueError("cannot evaluate an empty data loader")
    return _aggregate(rows)
