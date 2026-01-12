from collections.abc import Mapping

import xarray as xr
from xarray.indexes import PandasIndex

from xdggs.index import DGGSIndex
from xdggs.grid import DGGSInfo
from xdggs.utils import register_dggs, _extract_cell_id_variable, GRID_REGISTRY
from typing import Any
from xdggs_dggrid4py.dependences.grids import IGEO7Info, GridsConfig


@register_dggs("igeo7")
class IGEO7Index(DGGSIndex):
    _grid: DGGSInfo

    def __init__(
        self,
        cell_ids: Any | PandasIndex,
        dim: str,
        grid_info: DGGSInfo,
    ):
        if not isinstance(grid_info, DGGSInfo):
            raise ValueError(f"grid info object has an invalid type: {type(grid_info)}")
        super().__init__(cell_ids, dim, grid_info)

    @classmethod
    def from_variables(cls: type["IGEO7Index"], variables: Mapping[Any, xr.Variable],
                       *, options: Mapping[str, Any],) -> "IGEO7Index":
        _, var, dim = _extract_cell_id_variable(variables)
        attrs = var.attrs.copy()
        attrs["grid_name"] = attrs["grid_name"].upper()
        cls = GRID_REGISTRY.get(attrs["grid_name"].lower())
        if cls is None:
            raise ValueError(f'unknown DGGS grid name: {var.attrs["grid_name"]}.')
        if (attrs["grid_name"] != 'IGEO7'):
            raise ValueError(f'The grid_name({attrs["grid_name"]}) is wrong for the IGEO7Index.')
        dggrid_meta_config = GridsConfig["IGEO7"]["meta_config"]
        dggrid_meta_config.update({"dggs_vert0_lon": attrs.get("_dggs_vert0_lon", 11.20)})
        if (isinstance(var.values[0], str)):
            dggrid_meta_config.update({"input_hier_ndx_form": "DIGIT_STRING"})
            dggrid_meta_config.update({"output_hier_ndx_form": "DIGIT_STRING"})
        attrs.update({"_dggrid_meta_config": dggrid_meta_config})
        igeo7info = IGEO7Info.from_dict(attrs)
        return cls(var.data, dim, igeo7info)

    @property
    def grid_info(self) -> IGEO7Info:
        return self._grid

    def _repr_inline_(self, max_width: int):
        return f"IGEO7Index(level={self._grid.level})"

    def _replace(self, new_index: PandasIndex):
        return type(self)(new_index, self._dim, self._grid)
