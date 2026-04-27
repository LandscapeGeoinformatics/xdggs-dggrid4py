import xarray as xr
import dask.array as da
import dask
import numpy as np
import pandas as pd

from xarray.core.indexes import IndexSelResult, PandasIndex
from typing import Any
from xdggs_dggrid4py.utils import (register_zoneId_indexing,
                                   z7base7_to_z7int, z7int_to_z7base7)
from collections.abc import Hashable, Mapping


# IGEO7RangeIndex: get the range info for a chunk in z7 int base7 format.
@dask.delayed
def _create_rangeinfo_per_chunk(chunk: da, refinement_level: int, chunk_id: int, chunk_size: int):
    base7int = z7int_to_z7base7(chunk, refinement_level)
    diffs = np.ediff1d(base7int) != 1
    split = np.split(base7int, np.nonzero(diffs)[0] + 1)
    ranges = np.array([[len(range_), range_[0], range_[-1]] for range_ in split])
    start_pos = np.cumsum(ranges[:, 0]) - ranges[:, 0] + (chunk_id * chunk_size)
    ranges[:, 0] = start_pos
    return da.from_array(ranges)


# IGEO7RangeIndex: return a full zone IDs (z7 int) list from start and end of the range (z7int base7)
@dask.delayed
def _return_fullrange(start_id, stop_id, refinement_level):
    fullrange_z7_zoneIds = np.arange(start_id, stop_id + 1)
    fullrange_z7_zoneIds = z7base7_to_z7int(fullrange_z7_zoneIds, refinement_level)
    return da.from_array(fullrange_z7_zoneIds)


@register_zoneId_indexing(f"{__name__.split('.')[-1]}.RangeIndex")
class IGEO7RangeIndex(xr.Index):

    def __init__(self, ranges: da, refinement_level: int, dim='zone_id', coord_name=None):
        # The ranges array is in the form of [ [start position, start_range(z7base7), end_range(z7base7)], ...(other chunks) ]
        # where start = absolute start position of that chunk, the start and stop range of z7int in base7 for that chunk
        self.ranges = ranges
        self.refinement_level = refinement_level
        self.start, self.stop = da.min(self.ranges[:, 1]), da.max(self.ranges[:, 2])
        self.size = self.ranges[-1][0] + 1
        self.size += 1 if ((self.ranges[-1][2] - self.ranges[-1][1]) == 0) else (self.ranges[-1][2] - self.ranges[-1][1])
        self.dim = dim
        self._name = coord_name if (coord_name is not None) else self.dim

    # The function to locate the cooresponding z7 zone IDs(int) from array positions, that's what forward means
    def forward(self, dim_positions: dict[str, Any]) -> dict[Hashable, Any]:
        positions = dim_positions[self.dim]
        # check which ranges are the input positions belong to, the result is a bool array
        range_pos = (self.ranges[:, 0] > positions[:, None])
        # need to filter out all zero (not match with the above condition)
        found_ranges = da.nonzero(da.sum(range_pos, axis=-1))[0]
        range_pos = range_pos[found_ranges, :]
        positions = positions[found_ranges]
        # using argmax to find the max values, in this case, just 1, it will return the first hit
        # since the above checking bases on greater than, the pervious range is the target.
        range_idx = da.argmax(range_pos, axis=-1).compute() - 1
        labels = self.ranges[range_idx, 1] + (positions - self.ranges[range_idx, 0])
        labels = z7base7_to_z7int(labels, self.refinement_level)
        return {self._name: labels}

    # it is the function to lookup the array positions of the input z7 zone IDs (int), that's what reverse means
    def reverse(self, coord_labels: dict[Hashable, Any]) -> dict[str, Any]:
        labels = coord_labels[self._name]
        labels_base7 = z7int_to_z7base7(labels, self.refinement_level)
        range_pos = (self.ranges[:, 1] <= labels_base7[:, None]) & (self.ranges[:, 2] >= labels_base7[:, None])
        found_ranges = da.nonzero(da.sum(range_pos, axis=-1))[0]
        range_pos = range_pos[found_ranges]
        labels_base7 = labels_base7[found_ranges]
        range_idx = da.argmax(range_pos, axis=-1).compute()
        positions = self.ranges[range_idx, 0] + (labels_base7 - self.ranges[range_idx, 1])
        return {self.dim: positions.compute()}

    def isel(
        self, indexers: Mapping[Any, int | slice | np.ndarray | xr.Variable]
    ) -> xr.Index | None:
        idxer = indexers[self.dim]

        if isinstance(idxer, slice):
            # return RangeIndex(self.transform.slice(idxer))
            raise ValueError("silce is not support currently.")

        elif (isinstance(idxer, xr.Variable) and idxer.ndim > 1) or xr.core.duck_array_ops.ndim(
            idxer
        ) == 0:
            return None
        else:
            values = self.forward({self.dim: np.asarray(idxer)})[
                self._name
            ]
            if isinstance(idxer, xr.Variable):
                new_dim = idxer.dims[0]
            else:
                new_dim = self.dim
            pd_index = pd.Index(values, name=self._name)
            return PandasIndex(pd_index, new_dim, coord_dtype=values.dtype)

    def sel(
        self, labels: dict[Any, Any], method=None, tolerance=None
    ) -> IndexSelResult:
        label = labels[self.dim]

        if isinstance(label, slice):
            raise ValueError("silce is not support currently.")

        positions = self.reverse({self._name: np.asarray(label)})

        dim_indexers = {self.dim: positions[self.dim]}
        result = IndexSelResult(dim_indexers)

        return result

    @classmethod
    def from_array(cls, array, *, dim, name, grid_info):
        if array.ndim != 1:
            raise ValueError("only 1D cell ids are supported")

        if not isinstance(array, da.Array):
            array = da.from_array(array)

        refinement_level = grid_info.level
        chunk_size = array.chunks[0][0]
        [chunk_rangeinfo] = dask.compute([_create_rangeinfo_per_chunk(chunk, refinement_level, i, chunk_size)
                                          for i, chunk in enumerate(array.to_delayed())])

        return cls(da.vstack(chunk_rangeinfo).rechunk('auto'), refinement_level, dim, name)

    @classmethod
    def from_variables(cls, variables, *, options):
        name, var, dim = _extract_cell_id_variable(variables)
        grid_info = IGEO7Info.from_dict(var.attrs | options)

        return cls.from_array(var.data, dim=dim, name=name, grid_info=grid_info)

    def create_variables(
        self, variables: Mapping[Any, xr.Variable] | None = None
    ) -> dict[Hashable, xr.Variable]:
        """Create new coordinate variables from this index
        """
        name = self._name
        if variables is not None and name in variables:
            var = variables[name]
            attrs = var.attrs
            encoding = var.encoding
        else:
            attrs = None
            encoding = None

        chunk_arrays = [
            da.from_delayed(
                _return_fullrange(row[1], row[2], self.refinement_level),
                shape=(row[2] - row[1] + 1,),
                dtype="uint64"
            )
            for row in self.ranges
        ]
        data = da.concatenate(chunk_arrays).rechunk('auto')
        var = xr.Variable(self.dim, data, attrs=attrs, encoding=encoding)
        return {name: var}
