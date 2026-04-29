from collections.abc import Mapping

import xarray as xr
from xarray.indexes import PandasIndex

from xdggs.index import DGGSIndex
from xdggs.grid import DGGSInfo
from xdggs.utils import register_dggs, _extract_cell_id_variable
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
        attrs = dict(var.attrs)
        # Merge any options the caller passed via xr.set_options / decode kwargs.
        attrs.update(dict(options))

        if attrs.get("grid_name", "").lower() != "igeo7":
            raise ValueError(
                f"grid_name {attrs.get('grid_name')!r} is not 'igeo7' — "
                "wrong DGGS index for this variable."
            )

        # Build the dynamic DGGRID metafile config: start from the static
        # plugin defaults, overlay the canonical vertex-0 longitude attribute,
        # and switch the hier-ndx form to DIGIT_STRING when cell_ids look like
        # textual Z7 strings rather than the packed uint64 INT64 form.
        dggrid_meta_config = dict(GridsConfig["igeo7"]["meta_config"])
        vert0_lon = attrs.get(
            "igeo7_dggs_vert0_lon",
            attrs.get("_dggs_vert0_lon", dggrid_meta_config.get("dggs_vert0_lon", 11.20)),
        )
        dggrid_meta_config["dggs_vert0_lon"] = float(vert0_lon)
        if len(var.values) and isinstance(var.values[0], str):
            dggrid_meta_config["input_hier_ndx_form"]  = "DIGIT_STRING"
            dggrid_meta_config["output_hier_ndx_form"] = "DIGIT_STRING"

        # `from_dict` filters out unrelated CF/Zarr metadata via
        # translate_parameters; we just inject the dynamic meta config.
        igeo7_info = IGEO7Info.from_dict(
            {**attrs, "_dggrid_meta_config": dggrid_meta_config}
        )
        return cls(var.data, dim, igeo7_info)

    @property
    def grid_info(self) -> IGEO7Info:
        return self._grid

    def _repr_inline_(self, max_width: int):
        return f"IGEO7Index(level={self._grid.level})"

    def _replace(self, new_index: PandasIndex):
        return type(self)(new_index, self._dim, self._grid)

    def to_pandas_index(self):
        return self._index
