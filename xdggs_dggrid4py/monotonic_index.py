"""Lazy IGEO7/Z7 xarray Index backed by a (R, 2) monotonic-int range table.

`Z7MonotonicIndex` is the lazy counterpart to the eager
``IGEO7Index`` (which wraps a length-N ``PandasIndex`` of cell ids). It
backs the DGGS Zarr Convention v1's ``compression: "ranges"`` archive form,
where the on-disk ``cell_ids`` coordinate is a ``(R, 2)`` array of
``(start_z7, end_z7_inclusive)`` packed Z7 IDs and R is the number of
contiguous monotonic-int runs in the data.

Memory: ``16 B × R + a derived uint64[R+1] of cumulative offsets`` — typically
tens of KB for archives that would have GBs of dense ``cell_ids`` under the
eager index.

This class is **not** registered via ``@register_dggs("igeo7")`` — that name
is owned by ``IGEO7Index``. Instead, the project-local
``z7_xarray_paper.z7_zarr.open_dataset`` reader instantiates this class
directly when it sees ``dggs.compression == "ranges"`` in the archive's
group attributes. Upstream ``xdggs.decode`` only supports 1-D cell-id
coordinates today (per ``xdggs/conventions/xdggs.py``), so we cannot route
through it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from xdggs.index import DGGSIndex

# z7py is on PYTHONPATH in this project; the plugin imports it lazily.
from z7py.z7 import z7_to_monotonic_int, monotonic_int_to_z7
import numba as nb


# ---------------------------------------------------------------------------
# Numba batch helpers (z7py only ships scalar versions)
# ---------------------------------------------------------------------------


@nb.njit(cache=True, parallel=True)
def _z7_to_monotonic_int_batch(raw: np.ndarray, resolution: int) -> np.ndarray:
    out = np.empty(raw.shape[0], dtype=np.uint64)
    for i in nb.prange(raw.shape[0]):
        out[i] = z7_to_monotonic_int(raw[i], resolution)
    return out


@nb.njit(cache=True, parallel=True)
def _monotonic_int_to_z7_batch(values: np.ndarray, resolution: int) -> np.ndarray:
    out = np.empty(values.shape[0], dtype=np.uint64)
    for i in nb.prange(values.shape[0]):
        out[i] = monotonic_int_to_z7(values[i], resolution)
    return out


# ---------------------------------------------------------------------------


class Z7MonotonicIndex(DGGSIndex):
    """Lazy Z7 cell-id index backed by a (R, 2) monotonic-int range table.

    Inherits from ``DGGSIndex`` so the ``ds.dggs`` accessor (which uses
    ``isinstance(idx, DGGSIndex)`` to find a DGGS index) accepts us. We
    deliberately do **not** call ``DGGSIndex.__init__`` because that would
    construct a ``PandasIndex`` over the dense cell-ids array — exactly the
    eager materialisation we are designed to avoid. All ``DGGSIndex``
    methods that touch ``self._index`` are overridden below.
    """

    def __init__(
        self,
        range_z7: np.ndarray,
        dim: str,
        level: int,
        grid_info,
    ):
        # NOTE: skip super().__init__ on purpose — see class docstring.
        range_z7 = np.ascontiguousarray(range_z7, dtype=np.uint64)
        if range_z7.ndim != 2 or range_z7.shape[1] != 2:
            raise ValueError(
                f"range_z7 must be a (R, 2) uint64 array, got shape {range_z7.shape}"
            )
        self._range_z7 = range_z7
        self._dim = dim
        self._level = int(level)
        self._grid = grid_info
        # Sentinel — no PandasIndex backing. Anything that reaches for
        # `self._index` is a bug in this class's overrides.
        self._index = None

        if range_z7.size == 0:
            self._range_start_mono = np.empty(0, dtype=np.uint64)
            self._range_end_mono   = np.empty(0, dtype=np.uint64)
            self._data_offsets     = np.array([0], dtype=np.uint64)
            self._n_total = 0
        else:
            self._range_start_mono = _z7_to_monotonic_int_batch(range_z7[:, 0], self._level)
            self._range_end_mono   = _z7_to_monotonic_int_batch(range_z7[:, 1], self._level)
            lengths = (self._range_end_mono - self._range_start_mono + np.uint64(1)).astype(np.uint64)
            self._data_offsets = np.concatenate(
                ([np.uint64(0)], np.cumsum(lengths).astype(np.uint64))
            )
            self._n_total = int(self._data_offsets[-1])

    # ------------------------------------------------------------------ ctors
    @classmethod
    def from_variables(
        cls,
        variables,
        *,
        options,
    ) -> "Z7MonotonicIndex":
        """Construct from a single (R, 2) coord variable.

        Parameters
        ----------
        variables : Mapping[name, xr.Variable]
            Single entry whose value is a 2-D uint64 (R, 2) variable carrying
            the canonical IGEO7 attrs (``grid_name``, ``level``,
            ``igeo7_dggs_vert0_lon``, ``igeo7_wgs84_geodetic_conversion``).
        options : Mapping
            Must contain ``dim`` — the data dimension this index covers
            (length N, distinct from the coord's own dims).
        """
        if len(variables) != 1:
            raise ValueError(f"expected exactly one variable, got {len(variables)}")
        name, var = next(iter(variables.items()))
        if var.ndim != 2 or var.shape[1] != 2:
            raise ValueError(
                f"variable {name!r} must be (R, 2) uint64, got shape {var.shape}"
            )
        dim = options.get("dim") if options else None
        if dim is None:
            raise ValueError(
                "Z7MonotonicIndex.from_variables requires options['dim'] — the "
                "data dimension this index covers (separate from the coord's "
                "own dims)."
            )

        attrs = dict(var.attrs)
        if attrs.get("grid_name", "").lower() != "igeo7":
            raise ValueError(
                f"grid_name {attrs.get('grid_name')!r} is not 'igeo7'"
            )
        level = int(attrs["level"])

        # Build IGEO7Info using the same path IGEO7Index uses, so the same
        # accessors (cell_ids2geographic, cell_boundaries) are available.
        from xdggs_dggrid4py.dependences.grids import GridsConfig, IGEO7Info
        dggrid_meta_config = dict(GridsConfig["igeo7"]["meta_config"])
        vert0_lon = attrs.get(
            "igeo7_dggs_vert0_lon",
            attrs.get("_dggs_vert0_lon", dggrid_meta_config.get("dggs_vert0_lon", 11.20)),
        )
        dggrid_meta_config["dggs_vert0_lon"] = float(vert0_lon)
        igeo7_info = IGEO7Info.from_dict(
            {**attrs, "_dggrid_meta_config": dggrid_meta_config}
        )

        return cls(np.asarray(var.values, dtype=np.uint64), dim, level, igeo7_info)

    # ------------------------------------------------------------------ props
    @property
    def grid_info(self):
        return self._grid

    @property
    def dim(self):
        return self._dim

    @property
    def level(self):
        return self._level

    @property
    def range_table(self) -> np.ndarray:
        """Return the (R, 2) packed-Z7 range table (read-only view)."""
        return self._range_z7

    # ------------------------------------------------------------------ values
    def values(self) -> np.ndarray:
        """Reconstruct the dense length-N uint64 cell-ids array.

        Lazy in spirit (only called by consumers that genuinely need it,
        such as ``ds.dggs.cell_centers()``). The reconstruction is O(N) and
        uses numba for the monotonic-int → Z7 conversion.
        """
        if self._n_total == 0:
            return np.empty(0, dtype=np.uint64)
        # Build all monotonic ints in [start, end] for each range.
        out_mono = np.empty(self._n_total, dtype=np.uint64)
        for i in range(len(self._range_start_mono)):
            offset = int(self._data_offsets[i])
            length = int(self._data_offsets[i + 1]) - offset
            s = int(self._range_start_mono[i])
            out_mono[offset:offset + length] = np.arange(
                s, s + length, dtype=np.uint64
            )
        return _monotonic_int_to_z7_batch(out_mono, self._level)

    def cell_centers(self):
        return self._grid.cell_ids2geographic(self.values())

    def cell_boundaries(self):
        return self._grid.cell_boundaries(self.values())

    # ------------------------------------------------------------------ sel/isel
    def sel(self, labels, method=None):
        from xarray.core.indexing import IndexSelResult
        if method is not None:
            raise ValueError("method is not supported for Z7MonotonicIndex")
        if self._dim not in labels:
            raise KeyError(self._dim)
        target = labels[self._dim]
        scalar_in = not hasattr(target, "__len__")
        target_arr = np.atleast_1d(np.asarray(target, dtype=np.uint64))
        target_mono = _z7_to_monotonic_int_batch(target_arr, self._level)

        # For each query, find its containing range via binary search.
        # searchsorted(starts, t, side='right') - 1 → index of the last range
        # whose start ≤ t. Then verify t ≤ ends[idx].
        idx = np.searchsorted(self._range_start_mono, target_mono, side="right") - 1
        clipped = np.maximum(idx, 0)
        within = (idx >= 0) & (target_mono <= self._range_end_mono[clipped])
        if not within.all():
            missing = target_arr[~within]
            raise KeyError(
                f"{missing.size} Z7 cell IDs not in index "
                f"(first few: {missing[:5].tolist()})"
            )
        positions = (
            self._data_offsets[idx]
            + (target_mono - self._range_start_mono[idx])
        ).astype(np.int64)
        if scalar_in:
            positions = int(positions[0])
        return IndexSelResult({self._dim: positions})

    def isel(self, indexers):
        # Positional selection along self._dim. We don't try to keep a
        # compact range form here; degrade to a PandasIndex over the subset.
        if self._dim not in indexers:
            return self
        ix = indexers[self._dim]
        full_ids = self.values()
        sub = full_ids[ix]
        if np.isscalar(sub):
            sub = np.atleast_1d(sub)
        return xr.indexes.PandasIndex(sub, dim=self._dim)

    # ------------------------------------------------------------------ misc
    def create_variables(self, variables=None):
        """Expose the indexed dim as a lazy 1-D coord variable.

        xarray needs a coord variable for every indexed dim
        (``xr.Coordinates.from_xindex`` calls this). To preserve laziness,
        we wrap ``self.values()`` in a Dask-delayed array — it's only
        materialised when a downstream operation (``ds.cell_ids.values``,
        ``ds.dggs.cell_centers()``, etc.) actually requires the dense
        length-N array. ``ds.sel(cell_ids=...)`` goes through ``self.sel``
        and never triggers the delayed compute.
        """
        import dask
        import dask.array as da

        delayed = dask.delayed(self.values)()
        arr = da.from_delayed(
            delayed,
            shape=(self._n_total,),
            dtype=np.uint64,
            meta=np.array([], dtype=np.uint64),
        )
        var = xr.Variable(
            dims=(self._dim,),
            data=arr,
            attrs={
                "grid_name": "igeo7",
                "level": self._level,
                "igeo7_dggs_vert0_lon": float(self._grid.igeo7_dggs_vert0_lon),
                "igeo7_wgs84_geodetic_conversion": bool(
                    self._grid.igeo7_wgs84_geodetic_conversion
                ),
            },
        )
        return {self._dim: var}

    def equals(self, other, **kwargs) -> bool:
        if not isinstance(other, Z7MonotonicIndex):
            return False
        if self._dim != other._dim or self._level != other._level:
            return False
        return np.array_equal(self._range_z7, other._range_z7)

    def to_pandas_index(self) -> pd.Index:
        return pd.Index(self.values(), name=self._dim)

    def _repr_inline_(self, max_width: int) -> str:
        return (
            f"Z7MonotonicIndex(level={self._level}, "
            f"R={len(self._range_z7)}, N={self._n_total})"
        )

    def _copy(self, deep: bool = True, memo=None):
        rng = self._range_z7.copy() if deep else self._range_z7
        return type(self)(rng, self._dim, self._level, self._grid)

    def __getitem__(self, indexer):
        return self.isel({self._dim: indexer})
