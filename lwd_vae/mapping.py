"""Explicit physical-grid and ordering adapters for LWD inversion exports.

Model arrays have shape ``(nz, ny)``; axis 0 is increasing z, axis 1 increasing
y. Coordinates are in feet. Values are interpolated in whatever units the
caller supplies: pass *raw log10 resistivity* to interpolate in log space.

These NumPy adapters do not preserve autograd and are for export/verification,
not a latent-space inversion gradient path. Convert tensors explicitly with
``tensor.detach().cpu().numpy()`` only when exporting. For same-grid PyTorch models use
``image.transpose(-2, -1).reshape(-1)`` for one 2-D model's Fortran ordering.
Keep decoder denormalization and this transpose inside the autograd graph.
A different-grid differentiable inversion needs an equivalent interpolation
operator implemented in the framework used by the decoder.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np


def _real_finite(array: Any, name: str) -> np.ndarray:
    result = np.asarray(array)
    if result.dtype.kind not in "fiu" or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return result


def to_inversion_vector(image: Any) -> np.ndarray:
    """Flatten one ``(nz, ny)`` model in Fortran order (z varies fastest)."""
    array = _real_finite(image, "image")
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError("image must be a nonempty 2-D array shaped (nz, ny)")
    return np.array(array, order="F", copy=True).reshape(-1, order="F")


def from_inversion_vector(vector: Any, shape: tuple[int, int]) -> np.ndarray:
    """Restore a Fortran-ordered inversion vector to ``shape=(nz, ny)``."""
    array = _real_finite(vector, "vector")
    if array.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    try:
        nz, ny = (operator.index(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("shape must contain the two integers (nz, ny)") from exc
    if nz <= 0 or ny <= 0 or array.size != nz * ny:
        raise ValueError(f"vector size {array.size} is incompatible with shape {(nz, ny)}")
    return array.reshape((nz, ny), order="F").copy()


def _edges(values: Any, name: str) -> np.ndarray:
    edges = _real_finite(values, name).astype(np.float64)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError(f"{name} must be a strictly increasing 1-D cell-edge array")
    return edges


def _inside(source: np.ndarray, target: np.ndarray, axis: str) -> None:
    tolerance = 32 * np.finfo(np.float64).eps * max(1.0, abs(source[0]), abs(source[-1]))
    if target[0] < source[0] - tolerance or target[-1] > source[-1] + tolerance:
        raise ValueError(f"target {axis} edges extend beyond the source physical domain")


def map_to_inversion_grid(
    image: Any,
    source_y_edges_ft: Any,
    source_z_edges_ft: Any,
    target_y_edges_ft: Any,
    target_z_edges_ft: Any,
    *,
    as_vector: bool = False,
) -> np.ndarray:
    """Map a model by bilinear interpolation of physical cell centers.

    Nonuniform source and target cells are supported. Every target cell must
    lie inside the source physical domain, as defined by its *edges*. Within
    the outer half source cells, the nearest center value is held constant;
    no data beyond the source domain are invented. This also permits a source
    with one cell along an axis. This is pointwise interpolation, not a
    conservative cell-average remapping operation.

    Returns float64 ``(target_nz, target_ny)``; ``as_vector=True`` returns its
    Fortran-ordered vector. Source and target coordinates must both be feet;
    this function never rescales model extents or infers coordinate units.
    """
    array = _real_finite(image, "image")
    sy = _edges(source_y_edges_ft, "source_y_edges_ft")
    sz = _edges(source_z_edges_ft, "source_z_edges_ft")
    ty = _edges(target_y_edges_ft, "target_y_edges_ft")
    tz = _edges(target_z_edges_ft, "target_z_edges_ft")
    expected = (sz.size - 1, sy.size - 1)
    if array.shape != expected:
        raise ValueError(f"image shape {array.shape} does not match source grid {expected}")
    _inside(sy, ty, "y")
    _inside(sz, tz, "z")
    syc, szc = (sy[:-1] + sy[1:]) / 2, (sz[:-1] + sz[1:]) / 2
    tyc, tzc = (ty[:-1] + ty[1:]) / 2, (tz[:-1] + tz[1:]) / 2
    along_y = np.empty((array.shape[0], tyc.size), dtype=np.float64)
    for row in range(array.shape[0]):
        along_y[row] = np.interp(tyc, syc, array[row])
    mapped = np.empty((tzc.size, tyc.size), dtype=np.float64)
    for column in range(tyc.size):
        mapped[:, column] = np.interp(tzc, szc, along_y[:, column])
    return to_inversion_vector(mapped) if as_vector else mapped
