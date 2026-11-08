"""Memory-mapped VAE training data with one dataset-wide log-resistivity scale.

Images retain their physical orientation and rectangular ``(1, nz, ny)`` shape.
No flips, rotations, resizing, or per-image normalization are applied. PyTorch
is optional and is imported only when ``as_torch=True`` is requested.
"""

from __future__ import annotations

import json
import operator
import os
from pathlib import Path
from typing import Any

import numpy as np


def _bounds(log10_bounds: Any) -> tuple[float, float]:
    values = np.asarray(log10_bounds, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all() or values[0] >= values[1]:
        raise ValueError("log10_bounds must contain two finite, increasing values")
    return float(values[0]), float(values[1])


def _finite_real(values: Any, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain finite real numbers")
    return array


def normalize_log10rho(values: Any, log10_bounds: Any) -> np.ndarray:
    """Map raw log10(ohm m) to [-1, 1], using fixed dataset-wide bounds.

    The result is float32, suitable for a VAE input. Values outside the stated
    bounds raise an error instead of being silently clipped or rescaled.
    """
    lo, hi = _bounds(log10_bounds)
    array = _finite_real(values, "log10 resistivity")
    tolerance = 8 * np.finfo(np.float32).eps * max(1.0, abs(lo), abs(hi))
    if np.any(array < lo - tolerance) or np.any(array > hi + tolerance):
        raise ValueError(f"log10 resistivity is outside [{lo}, {hi}]")
    result = 2.0 * (array.astype(np.float32) - lo) / (hi - lo) - 1.0
    return np.clip(result, -1.0, 1.0).astype(np.float32, copy=False)


def denormalize_log10rho(values: Any, log10_bounds: Any) -> np.ndarray:
    """Invert :func:`normalize_log10rho`; return raw log10(ohm m).

    This NumPy function is for data preparation/export, not differentiable
    inversion. In PyTorch use ``lo + (decoded + 1) * (hi - lo) / 2`` directly.
    """
    lo, hi = _bounds(log10_bounds)
    array = _finite_real(values, "normalized model")
    tolerance = 8 * np.finfo(np.float32).eps
    if np.any(array < -1.0 - tolerance) or np.any(array > 1.0 + tolerance):
        raise ValueError("normalized model is outside [-1, 1]")
    array = np.clip(array.astype(np.float32), -1.0, 1.0)
    return (lo + (array + 1.0) * (hi - lo) / 2.0).astype(np.float32, copy=False)


class LWDVAEDataset:
    """Read one split without loading the model array into RAM.

    ``dataset[i]`` returns a float32 normalized image of shape ``(1, nz, ny)``.
    With ``return_metadata=True`` it returns ``(image, metadata_dict)``. With
    ``as_torch=True`` the image is a CPU tensor; the class otherwise requires
    only NumPy and can still be used with a PyTorch DataLoader as a map-style
    dataset. The memmap is reopened after worker fork or pickle/unpickle.

    Array shape and dtype are checked when constructed. Finite values and the
    raw resistivity range are checked for every accessed sample; call
    :meth:`validate_all` for an explicit full-file scan before training.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        split: str = "train",
        as_torch: bool = False,
        return_metadata: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.as_torch = bool(as_torch)
        self.return_metadata = bool(return_metadata)
        with (self.root / "manifest.json").open(encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        version = self.manifest.get("schema_version")
        if version not in (1, 2):
            raise ValueError("unsupported manifest schema_version; expected 1 (legacy) or 2")
        self.config = self.manifest["config"]
        # v1 同样采用地质 y/z 坐标，但含随机地层倾角；保留旧数据可读性。
        self.coordinate_frame = self.manifest.get("coordinate_frame", "global_aligned_yz")
        if self.coordinate_frame != "global_aligned_yz" or (version == 2 and (
                self.manifest.get("coordinate_frame") != "global_aligned_yz"
                or self.config.get("coordinate_frame") != "global_aligned_yz")):
            raise ValueError("Expected matching global_aligned_yz coordinate frame; no implicit rotation")
        grid = self.config["grid"]
        self.nz, self.ny = int(grid["nz"]), int(grid["ny"])
        if self.nz <= 0 or self.ny <= 0 or self.nz != grid["nz"] or self.ny != grid["ny"]:
            raise ValueError("grid nz and ny must be positive integers")
        self.image_shape = (1, self.nz, self.ny)
        self.log10_bounds = _bounds(self.config["resistivity"]["log10_bounds"])
        self.normalization = {
            "kind": "global_log10_affine",
            "log10_bounds": list(self.log10_bounds),
            "normalized_bounds": [-1.0, 1.0],
            "formula": "2 * (log10rho - lo) / (hi - lo) - 1",
        }
        try:
            split_info = self.manifest["splits"][split]
        except KeyError as exc:
            raise ValueError(f"unknown dataset split: {split!r}") from exc
        self.count = int(split_info["count"])
        if self.count < 0 or self.count != split_info["count"]:
            raise ValueError("split count must be a nonnegative integer")
        self.array_path = self._local_path(split_info["path"])
        self.metadata_path = self._local_path(split_info["metadata"])
        self._array = None
        self._array_pid = None
        self._metadata = None
        # Read just the header and establish format validity, then release it.
        self._open_array()
        self.close()
        if self.as_torch:
            try:
                import torch  # noqa: F401
            except ImportError as exc:
                raise ImportError("as_torch=True requires an installed PyTorch") from exc

    def _local_path(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("manifest paths must remain within the dataset root")
        return path

    def _open_array(self) -> np.memmap:
        pid = os.getpid()
        if self._array is None or self._array_pid != pid:
            self.close()
            array = np.load(self.array_path, mmap_mode="r", allow_pickle=False)
            expected = (self.count, *self.image_shape)
            if array.shape != expected:
                raise ValueError(f"dataset shape {array.shape} does not match {expected}")
            if array.dtype != np.dtype(np.float32):
                raise ValueError(f"dataset dtype must be float32, got {array.dtype}")
            self._array = array
            self._array_pid = pid
        return self._array

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Any:
        index = operator.index(index)
        if index < 0:
            index += self.count
        if index < 0 or index >= self.count:
            raise IndexError("dataset index out of range")
        try:
            image = normalize_log10rho(self._open_array()[index], self.log10_bounds)
        except ValueError as exc:
            raise ValueError(f"invalid sample {index} in split {self.split!r}: {exc}") from exc
        # Ensure returned data owns its memory and is writable for tensor use.
        image = np.array(image, dtype=np.float32, order="C", copy=True)
        if self.as_torch:
            import torch

            image = torch.from_numpy(image)
        if self.return_metadata:
            return image, dict(self.metadata[index])
        return image

    @property
    def metadata(self) -> list[dict[str, Any]]:
        """Sample metadata in array order, loaded only when first requested."""
        if self._metadata is None:
            records = []
            with self.metadata_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid metadata JSON at line {line_number}") from exc
                    if not isinstance(record, dict):
                        raise ValueError(f"metadata line {line_number} must be a JSON object")
                    records.append(record)
            if len(records) != self.count:
                raise ValueError(f"metadata has {len(records)} rows; expected {self.count}")
            self._metadata = records
        return self._metadata

    def validate_all(self, chunk_size: int = 128) -> None:
        """Scan raw values in bounded memory and check metadata row count."""
        chunk_size = operator.index(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        array = self._open_array()
        for start in range(0, self.count, chunk_size):
            try:
                normalize_log10rho(array[start : start + chunk_size], self.log10_bounds)
            except ValueError as exc:
                raise ValueError(f"invalid samples starting at {start}: {exc}") from exc
        _ = self.metadata

    def close(self) -> None:
        """Release this instance's open memory map; subsequent reads reopen it."""
        if self._array is not None:
            mmap = getattr(self._array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._array = None
        self._array_pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_array"] = None
        state["_array_pid"] = None
        return state

    def __enter__(self) -> "LWDVAEDataset":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
