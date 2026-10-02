"""Lazy IGEO7/Z7 xarray Index backed by a (R, 2) monotonic-int range table.

`Z7MonotonicIndex` is the lazy counterpart to the eager
``IGEO7Index`` (which wraps a length-N ``PandasIndex`` of cell ids). It
backs the DGGS Zarr Convention v1's ``compression: "ranges"`` archive form,
where the on-disk ``cell_ids`` coordinate is a ``(R, 2)`` array of
``(start_z7, end_z7_inclusive)`` packed Z7 IDs and R is the number of
contiguous monotonic-int runs in the data.

Memory: ``16 B × R + a derived int64[R+1] of cumulative offsets`` — typically
tens of KB for archives that would have GBs of dense ``cell_ids`` under the
eager index.

Nothing on the selection path is O(N). ``sel`` and ``isel`` work on the range
table (O(k log R) for k requested cells), slices stay range-backed, and the
``cell_ids`` coordinate is a chunked dask array whose chunks are computed
from the range table on demand. Only explicit dense requests
(``values()``, ``ds.cell_ids.values``, ``cell_centers()`` on the whole index,
``to_pandas_index()``) build all N ids — subset first on large archives.

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


#: default length of a ``cell_ids`` coordinate chunk when the caller passes
#: no ``chunks`` option (7^8 cells = one r4 parent at r12, 46 MB of uint64)
DEFAULT_CHUNK = 7 ** 8


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


@nb.njit(cache=True, parallel=True)
def _positions_to_mono(pos, start_mono, offsets):
    """Global positions (in range) -> monotonic ints, O(k log R)."""
    r = np.searchsorted(offsets, pos, side="right") - 1
    out = np.empty(pos.shape[0], dtype=np.uint64)
    for i in nb.prange(pos.shape[0]):
        out[i] = start_mono[r[i]] + np.uint64(pos[i] - offsets[r[i]])
    return out


@nb.njit(cache=True)
def _span_to_z7(p0, p1, start_mono, offsets, resolution):
    """Cell ids at the contiguous positions [p0, p1), O(p1 - p0 + log R)."""
    out = np.empty(p1 - p0, dtype=np.uint64)
    r = np.searchsorted(offsets, p0, side="right") - 1
    p = p0
    while p < p1:
        stop = min(p1, offsets[r + 1])
        base = start_mono[r] + np.uint64(p - offsets[r])
        for j in range(stop - p):
            out[p - p0 + j] = monotonic_int_to_z7(base + np.uint64(j), resolution)
        p = stop
        r += 1
    return out


def _pack_table(start_mono, offsets, level):
    return start_mono, offsets, level


def _coord_chunk(table, p0, p1):
    """Dask task: one chunk of the lazy ``cell_ids`` coordinate."""
    start_mono, offsets, level = table
    return _span_to_z7(np.int64(p0), np.int64(p1), start_mono, offsets, level)


def _runs(mono: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Strictly increasing monotonic ints -> (starts, ends) of maximal runs; None otherwise."""
    if mono.size == 0:
        return np.empty(0, np.uint64), np.empty(0, np.uint64)
    step = np.diff(mono.astype(np.int64))
    if np.any(step <= 0):
        return None
    brk = np.flatnonzero(step != 1)
    starts = np.concatenate((mono[:1], mono[brk + 1]))
    ends = np.concatenate((mono[brk], mono[-1:]))
    return starts.astype(np.uint64), ends.astype(np.uint64)


def _merge_adjacent(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if starts.size < 2:
        return starts, ends
    keep = np.concatenate(([True], starts[1:] != ends[:-1] + np.uint64(1)))
    first = np.flatnonzero(keep)
    last = np.concatenate((first[1:] - 1, [starts.size - 1]))
    return starts[first], ends[last]


# ---------------------------------------------------------------------------


class Z7MonotonicIndex(DGGSIndex):
    """Lazy Z7 cell-id index backed by a (R, 2) monotonic-int range table.

    Inherits from ``DGGSIndex`` so the ``ds.dggs`` accessor (which uses
    ``isinstance(idx, DGGSIndex)`` to find a DGGS index) accepts us. We
    deliberately do **not** call ``DGGSIndex.__init__`` because that would
    construct a ``PandasIndex`` over the dense cell-ids array — exactly the
    eager materialisation we are designed to avoid. All ``DGGSIndex``
    methods that touch ``self._index`` are overridden below.

    Ranges must be sorted, non-overlapping and ``start <= end`` (the
    archive's storage order); this is checked on construction.

    ``chunks`` sets the dask chunking of the ``cell_ids`` coordinate: an int
    (uniform chunk length) or a tuple of chunk lengths summing to N. Matching
    the data variables' chunking keeps ``xr.map_blocks`` and blockwise
    arithmetic aligned; ``isel`` slices carry the chunk boundaries along.
    """

    def __init__(
        self,
        range_z7: np.ndarray,
        dim: str,
        level: int,
        grid_info,
        chunks: int | tuple[int, ...] | None = None,
    ):
        # NOTE: skip super().__init__ on purpose — see class docstring.
        range_z7 = np.ascontiguousarray(range_z7, dtype=np.uint64)
        if range_z7.size == 0:
            range_z7 = range_z7.reshape(0, 2)
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

        if range_z7.shape[0] == 0:
            self._range_start_mono = np.empty(0, dtype=np.uint64)
            self._range_end_mono   = np.empty(0, dtype=np.uint64)
            self._data_offsets     = np.array([0], dtype=np.int64)
            self._n_total = 0
        else:
            self._range_start_mono = _z7_to_monotonic_int_batch(range_z7[:, 0], self._level)
            self._range_end_mono   = _z7_to_monotonic_int_batch(range_z7[:, 1], self._level)
            if np.any(self._range_end_mono < self._range_start_mono):
                raise ValueError("range_z7 has ranges with end < start")
            if np.any(self._range_start_mono[1:] <= self._range_end_mono[:-1]):
                raise ValueError("range_z7 ranges must be sorted and non-overlapping")
            lengths = (self._range_end_mono - self._range_start_mono).astype(np.int64) + 1
            self._data_offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
            self._n_total = int(self._data_offsets[-1])
        self._chunk_bounds = self._normalise_chunks(chunks)

    def _normalise_chunks(self, chunks) -> np.ndarray:
        n = self._n_total
        if chunks is None:
            chunks = DEFAULT_CHUNK
        if isinstance(chunks, (int, np.integer)):
            c = max(int(chunks), 1)
            return np.unique(np.append(np.arange(0, n, c, dtype=np.int64), n))
        sizes = np.asarray(chunks, dtype=np.int64)
        if sizes.sum() != n or np.any(sizes < 0):
            raise ValueError(f"chunks {tuple(chunks)[:5]}... do not sum to N={n}")
        return np.unique(np.concatenate(([0], np.cumsum(sizes))))

    @classmethod
    def _from_mono(cls, starts, ends, dim, level, grid_info, chunks=None) -> "Z7MonotonicIndex":
        starts, ends = _merge_adjacent(np.asarray(starts, np.uint64), np.asarray(ends, np.uint64))
        rng = np.stack([_monotonic_int_to_z7_batch(starts, level),
                        _monotonic_int_to_z7_batch(ends, level)], axis=1)
        return cls(rng, dim, level, grid_info, chunks=chunks)

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
            (length N, distinct from the coord's own dims). Optional
            ``chunks`` — the dask chunking of the lazy ``cell_ids``
            coordinate (int or tuple of lengths, see class docstring).
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

        return cls(np.asarray(var.values, dtype=np.uint64), dim, level, igeo7_info,
                   chunks=options.get("chunks"))

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

    @property
    def size(self) -> int:
        """Number of cells N covered by the index (no materialisation)."""
        return self._n_total

    @property
    def chunks(self) -> tuple[int, ...]:
        """Chunk lengths of the lazy ``cell_ids`` coordinate."""
        return tuple(int(c) for c in np.diff(self._chunk_bounds))

    # ------------------------------------------------------------------ values
    def values(self) -> np.ndarray:
        """Reconstruct the dense length-N uint64 cell-ids array.

        O(N) memory and time (8 B per cell) — only for consumers that
        genuinely need every id, such as ``cell_centers()`` on the whole
        index. Use ``cell_ids_at`` or ``isel`` for parts of large archives.
        """
        return self.cell_ids_at(slice(0, self._n_total))

    def cell_ids_at(self, positions) -> np.ndarray:
        """Cell ids at global positions (int array or slice), without building all N ids.

        O(k log R) for k positions. Negative positions count from the end;
        out-of-range positions raise ``IndexError``.
        """
        n = self._n_total
        if isinstance(positions, slice):
            p0, p1, step = positions.indices(n)
            if step == 1:
                if p1 <= p0:
                    return np.empty(0, dtype=np.uint64)
                return _span_to_z7(np.int64(p0), np.int64(p1), self._range_start_mono,
                                   self._data_offsets, self._level)
            positions = np.arange(p0, p1, step, dtype=np.int64)
        pos = self._normalise_positions(positions)
        if pos.size == 0:
            return np.empty(0, dtype=np.uint64)
        mono = _positions_to_mono(pos.ravel(), self._range_start_mono, self._data_offsets)
        return _monotonic_int_to_z7_batch(mono, self._level).reshape(pos.shape)

    def _normalise_positions(self, positions) -> np.ndarray:
        pos = np.asarray(positions)
        if pos.dtype.kind == "b":
            if pos.shape != (self._n_total,):
                raise IndexError(
                    f"boolean indexer has shape {pos.shape}, expected ({self._n_total},)"
                )
            return np.flatnonzero(pos).astype(np.int64)
        if pos.dtype.kind not in "iu":
            raise IndexError(f"positional indexer must be integer or boolean, got {pos.dtype}")
        pos = pos.astype(np.int64)
        pos = np.where(pos < 0, pos + self._n_total, pos)
        if pos.size and (pos.min() < 0 or pos.max() >= self._n_total):
            raise IndexError(f"positions out of bounds for {self._dim!r} of size {self._n_total}")
        return pos

    def cell_centers(self):
        return self._grid.cell_ids2geographic(self.values())

    def cell_boundaries(self):
        return self._grid.cell_boundaries(self.values())

    # ------------------------------------------------------------------ sel/isel
    def _label_to_position(self, mono: np.ndarray, side: str) -> np.ndarray:
        """Insertion positions of monotonic ints (absent labels allowed).

        side="left": position of the first cell >= label;
        side="right": position just after the last cell <= label.
        """
        if self._n_total == 0:
            return np.zeros(mono.shape, dtype=np.int64)
        s, e, off = self._range_start_mono, self._range_end_mono, self._data_offsets
        if side == "left":
            r = np.searchsorted(e, mono, side="left")              # first range with end >= label
            rc = np.minimum(r, len(s) - 1)
            inside = np.where(mono > s[rc], (mono - s[rc]).astype(np.int64), 0)
            return np.where(r >= len(s), self._n_total, off[rc] + inside)
        r = np.searchsorted(s, mono, side="right") - 1             # last range with start <= label
        rc = np.maximum(r, 0)
        inside = np.minimum(mono - s[rc], e[rc] - s[rc]).astype(np.int64) + 1
        return np.where(r < 0, 0, off[rc] + inside)

    def sel(self, labels, method=None, tolerance=None):
        """Label-based selection on packed Z7 ids, O(k log R).

        - scalar id -> int position (``KeyError`` if absent);
        - array / ``Variable`` / ``DataArray`` of ids -> positions with the
          label's dims (``KeyError`` if any is absent);
        - ``slice(a, b)`` -> positional slice of every stored cell with
          ``a <= id <= b`` (pandas semantics: inclusive, bounds need not be
          present). Selecting all descendants of a parent P is
          ``slice(first_child(P), last_child(P))`` and stays range-backed.
        """
        from xarray.core.indexing import IndexSelResult
        from xarray.core.dataarray import DataArray
        from xarray.core.variable import Variable

        if method is not None:
            raise ValueError(f"method={method!r} is not supported for Z7MonotonicIndex")
        if tolerance is not None:
            raise ValueError("tolerance is not supported for Z7MonotonicIndex")
        if self._dim not in labels:
            raise KeyError(self._dim)
        target = labels[self._dim]

        if isinstance(target, slice):
            if target.step is not None:
                raise ValueError("label slices with a step are not supported for Z7MonotonicIndex")
            p0, p1 = 0, self._n_total
            if target.start is not None:
                m = _z7_to_monotonic_int_batch(np.array([target.start], dtype=np.uint64), self._level)
                p0 = int(self._label_to_position(m, "left")[0])
            if target.stop is not None:
                m = _z7_to_monotonic_int_batch(np.array([target.stop], dtype=np.uint64), self._level)
                p1 = int(self._label_to_position(m, "right")[0])
            return IndexSelResult({self._dim: slice(p0, max(p0, p1))})

        raw = target.data if isinstance(target, (Variable, DataArray)) else target
        target_arr = np.asarray(raw)
        if target_arr.dtype.kind not in "iu":
            raise KeyError(f"Z7MonotonicIndex labels must be integer cell ids, got {target_arr.dtype}")
        flat = np.ascontiguousarray(target_arr.ravel(), dtype=np.uint64)
        target_mono = _z7_to_monotonic_int_batch(flat, self._level)

        # For each query, find its containing range via binary search.
        # searchsorted(starts, t, side='right') - 1 → index of the last range
        # whose start ≤ t. Then verify t ≤ ends[idx].
        idx = np.searchsorted(self._range_start_mono, target_mono, side="right") - 1
        clipped = np.maximum(idx, 0)
        within = (idx >= 0)
        if self._n_total:
            within &= target_mono <= self._range_end_mono[clipped]
        if not within.all():
            missing = flat[~within]
            raise KeyError(
                f"{missing.size} Z7 cell IDs not in index "
                f"(first few: {missing[:5].tolist()})"
            )
        positions = (
            self._data_offsets[clipped]
            + (target_mono - self._range_start_mono[clipped]).astype(np.int64)
        ).reshape(target_arr.shape)
        if target_arr.ndim == 0:
            return IndexSelResult({self._dim: int(positions)})
        if isinstance(target, Variable):
            positions = Variable(target.dims, positions)
        elif isinstance(target, DataArray):
            positions = DataArray(positions, coords=target._coords, dims=target.dims)
        return IndexSelResult({self._dim: positions})

    def isel(self, indexers):
        """Positional selection without materialising the index.

        Contiguous slices stay a ``Z7MonotonicIndex`` (range table clipped,
        coordinate chunk boundaries carried along). Integer / boolean arrays
        resolve only the selected ids: strictly increasing selections are
        re-compressed into ranges, anything else (reordered, repeated) becomes
        an eager ``IGEO7Index`` over the k selected ids, keeping grid_info.
        Scalars and indexers adding new dimensions drop the index, as in
        ``PandasIndex``.
        """
        from xarray.core.variable import Variable

        if self._dim not in indexers:
            return self
        ix = indexers[self._dim]
        if isinstance(ix, Variable):
            if ix.dims != (self._dim,):
                return None
            ix = ix.data
        if not isinstance(ix, slice) and np.ndim(ix) == 0:
            return None
        n = self._n_total

        if isinstance(ix, slice):
            p0, p1, step = ix.indices(n)
            if step == 1:
                if (p0, p1) == (0, n):
                    return self
                return self._slice(p0, max(p0, p1))
            pos = np.arange(p0, p1, step, dtype=np.int64)
        else:
            pos = self._normalise_positions(ix)
            if pos.ndim != 1:
                return None

        mono = (_positions_to_mono(pos, self._range_start_mono, self._data_offsets)
                if pos.size else np.empty(0, np.uint64))
        runs = _runs(mono)
        if runs is not None:
            return self._from_mono(*runs, self._dim, self._level, self._grid)
        from xdggs_dggrid4py.index import IGEO7Index
        return IGEO7Index(_monotonic_int_to_z7_batch(mono, self._level), self._dim, self._grid)

    def _slice(self, p0: int, p1: int) -> "Z7MonotonicIndex":
        b = self._chunk_bounds
        bounds = np.unique(np.clip(b, p0, p1)) - p0
        chunks = tuple(int(c) for c in np.diff(bounds)) or (0,)
        if p1 <= p0:
            return type(self)(np.empty((0, 2), np.uint64), self._dim, self._level, self._grid, chunks=0)
        off = self._data_offsets
        r0 = int(np.searchsorted(off, p0, side="right") - 1)
        r1 = int(np.searchsorted(off, p1 - 1, side="right") - 1)
        starts = self._range_start_mono[r0:r1 + 1].copy()
        ends = self._range_end_mono[r0:r1 + 1].copy()
        starts[0] += np.uint64(p0 - off[r0])
        ends[-1] = self._range_start_mono[r1] + np.uint64(p1 - 1 - off[r1])
        rng = np.stack([_monotonic_int_to_z7_batch(starts, self._level),
                        _monotonic_int_to_z7_batch(ends, self._level)], axis=1)
        return type(self)(rng, self._dim, self._level, self._grid, chunks=chunks)

    # ------------------------------------------------------------------ alignment
    def _check_compatible(self, other) -> None:
        if not isinstance(other, Z7MonotonicIndex):
            raise TypeError(f"cannot align Z7MonotonicIndex with {type(other).__name__}")
        if self._level != other._level or self._grid != other._grid:
            raise ValueError("Alignment with different grid parameters is not supported.")

    def join(self, other, how="inner"):
        """Range-table intersection (inner) / union (outer); O(R1 + R2)."""
        self._check_compatible(other)
        if how == "left":
            return self
        if how == "right":
            return other
        a = np.stack([self._range_start_mono, self._range_end_mono], 1)
        b = np.stack([other._range_start_mono, other._range_end_mono], 1)
        if how == "inner":
            i = j = 0
            out = []
            while i < len(a) and j < len(b):
                lo, hi = max(a[i, 0], b[j, 0]), min(a[i, 1], b[j, 1])
                if lo <= hi:
                    out.append((lo, hi))
                if a[i, 1] < b[j, 1]:
                    i += 1
                else:
                    j += 1
            r = np.array(out, dtype=np.uint64).reshape(-1, 2)
        elif how == "outer":
            r = np.concatenate([a, b])
            r = r[np.argsort(r[:, 0], kind="stable")]
            merged = []
            for s, e in r:
                if merged and s <= merged[-1][1] + np.uint64(1):
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            r = np.array(merged, dtype=np.uint64).reshape(-1, 2)
        else:
            raise ValueError(f"join how={how!r} is not supported for Z7MonotonicIndex")
        return self._from_mono(r[:, 0], r[:, 1], self._dim, self._level, self._grid)

    def reindex_like(self, other):
        """Positions in self of every cell of ``other`` (-1 where absent).

        The result is a length-N(other) int64 indexer — inherent to
        xarray's reindexing — computed range by range without building ids.
        """
        self._check_compatible(other)
        pos = np.full(other._n_total, -1, dtype=np.int64)
        for k in range(len(other._range_start_mono)):
            s, e = other._range_start_mono[k], other._range_end_mono[k]
            o = int(other._data_offsets[k])
            lo = np.searchsorted(self._range_end_mono, s, side="left")
            hi = np.searchsorted(self._range_start_mono, e, side="right")
            for r in range(lo, hi):
                a = max(s, self._range_start_mono[r])
                b = min(e, self._range_end_mono[r])
                n = int(b - a) + 1
                dst = o + int(a - s)
                src = int(self._data_offsets[r]) + int(a - self._range_start_mono[r])
                pos[dst:dst + n] = np.arange(src, src + n, dtype=np.int64)
        return {self._dim: pos}

    @classmethod
    def concat(cls, indexes, dim, positions=None):
        """Concatenate along ``dim``; stays range-backed when the result is sorted."""
        indexes = list(indexes)
        first = indexes[0]
        for other in indexes[1:]:
            first._check_compatible(other)
        starts = np.concatenate([i._range_start_mono for i in indexes])
        ends = np.concatenate([i._range_end_mono for i in indexes])
        chunks = tuple(c for i in indexes for c in i.chunks if c)
        if positions is None and np.all(starts[1:] > ends[:-1]):
            return cls._from_mono(starts, ends, dim, first._level, first._grid, chunks=chunks or None)
        from xdggs_dggrid4py.index import IGEO7Index
        ids = np.concatenate([i.values() for i in indexes])
        if positions is not None:
            order = np.concatenate([np.asarray(p, dtype=np.int64) for p in positions])
            out = np.empty_like(ids)
            out[order] = ids
            ids = out
        return IGEO7Index(ids, dim, first._grid)

    def rename(self, name_dict, dims_dict):
        if self._dim not in dims_dict:
            return self
        return type(self)(self._range_z7, dims_dict[self._dim], self._level, self._grid,
                          chunks=self.chunks or None)

    # ------------------------------------------------------------------ misc
    def create_variables(self, variables=None):
        """Expose the indexed dim as a lazy, chunked 1-D coord variable.

        xarray needs a coord variable for every indexed dim
        (``xr.Coordinates.from_xindex`` calls this). Each dask chunk computes
        only its own ids from the range table (one shared graph key holds the
        O(R) table), so ``ds.cell_ids[:10].values`` or blockwise work touches
        only the chunks it needs. ``.values`` on the whole coordinate still
        builds all N ids.
        """
        import dask.array as da
        from dask.base import tokenize
        from dask.highlevelgraph import HighLevelGraph

        token = tokenize(self._range_z7, self._level, self._chunk_bounds)
        name = f"z7-cell-ids-{token}"
        table_key = f"z7-range-table-{token}"
        dsk: dict = {table_key: (_pack_table, self._range_start_mono, self._data_offsets, self._level)}
        b = self._chunk_bounds
        for i in range(len(b) - 1):
            dsk[(name, i)] = (_coord_chunk, table_key, int(b[i]), int(b[i + 1]))
        chunks = (self.chunks,) if len(b) > 1 else ((0,),)
        if len(b) == 1:
            dsk[(name, 0)] = (np.empty, 0, np.uint64)
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[])
        arr = da.Array(graph, name, chunks=chunks, dtype=np.uint64,
                       meta=np.array([], dtype=np.uint64))
        attrs = {
            "grid_name": "igeo7",
            "level": self._level,
            "igeo7_dggs_vert0_lon": float(self._grid.igeo7_dggs_vert0_lon),
            "igeo7_wgs84_geodetic_conversion": bool(
                self._grid.igeo7_wgs84_geodetic_conversion
            ),
        }
        var = xr.Variable(dims=(self._dim,), data=arr, attrs=attrs)
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

    def __repr__(self) -> str:
        return self._repr_inline_(80)

    def _copy(self, deep: bool = True, memo=None):
        rng = self._range_z7.copy() if deep else self._range_z7
        return type(self)(rng, self._dim, self._level, self._grid, chunks=self.chunks or None)

    def __getitem__(self, indexer):
        return self.isel({self._dim: indexer})
