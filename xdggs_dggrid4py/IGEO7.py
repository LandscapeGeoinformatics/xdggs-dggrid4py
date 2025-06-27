from collections.abc import Mapping

import numpy as np
import xarray as xr
from xarray.indexes import PandasIndex

from xdggs.index import DGGSIndex
from xdggs.grid import DGGSInfo
from xdggs.utils import register_dggs, _extract_cell_id_variable, GRID_REGISTRY
from typing import Any, ClassVar, Sequence, Hashable, Iterable, List
from dataclasses import dataclass
try:
    from typing import Self, Tuple
except ImportError:  # pragma: no cover
    from typing_extensions import Self
from xdggs_dggrid4py.regridding_methods import *
from xdggs_dggrid4py.utils import _gen_centroid_from_cellids, _gen_polygon_from_cellids, _gen_parents_from_cellids, igeo7regridding_method
from tqdm.auto import tqdm
from dggrid4py import DGGRIDv7
import geopandas as gpd
import shapely
import tempfile
import os
from pyproj import Transformer
from itertools import chain
from concurrent.futures import ProcessPoolExecutor

try:
    dggrid_path = os.environ['DGGRID_PATH']
except KeyError:
    raise Exception("DGGRID_PATH env var not found")


@dataclass(frozen=True)
class IGEO7Info(DGGSInfo):
    src_epsg: str
    coordinate: list
    method: str
    mp: int
    chunk: Tuple[int, int]
    grid_name: str
    valid_parameters: ClassVar[dict[str, Any]] = {'grid_name': ['IGEO7'], "level": range(-1, 15), "method": list(igeo7regridding_method.keys())}

    def __post_init__(self):
        if (self.level not in self.valid_parameters['level']):
            raise ValueError("resolution must be an integer between 0 and 15")
        if (self.method.lower() not in self.valid_parameters["method"]):
            raise ValueError(f"method {self.method.lower()} is not supported.")
        if (self.grid_name.upper() not in self.valid_parameters['grid_name']):
            raise ValueError(f"grid_name {self.grid_name} is not supported.")

    @classmethod
    def from_dict(cls: type[Self], mapping: dict[str, Any]) -> Self:
        params = {k: v for k, v in mapping.items()}
        return cls(**params)

    def to_dict(self: Self) -> dict[str, Any]:
        return {"level": self.level, "src_epsg": self.src_epsg, "coordinate": self.coordinate,
                "grid_name": self.grid_name, "method": self.method, "mp": self.mp, "chunk": self.chunk}

    def cell_ids2geographic(
        self, cell_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        working_dir = tempfile.mkdtemp()
        dggrid = DGGRIDv7(dggrid_path, working_dir=working_dir, silent=True)
        centroids_df = dggrid.grid_cell_centroids_from_cellids(cell_ids, self.grid_name, self.level,
                                                               input_address_type='Z7_STRING', output_address_type='Z7_STRING')
        centroids = centroids_df.geometry.get_coordinates()
        return (centroids['x'].values, centroids['y'].values)

    def geographic2cell_ids(self, lon, lat):
        if (len(lon) != len(lat)):
            lon = np.array(lon)
            lat = np.array(lat)
            lon, lat = np.broadcast_arrays(lon,  lat[:, None])
        centroids = np.stack([lon, lat], axis=-1).reshape(-1, 2)
        centroids = [shapely.Point(c[0], c[1]) for c in centroids]
        centroids = gpd.GeoDataFrame([0] * len(centroids), geometry=centroids, crs='wgs84')
        working_dir = tempfile.mkdtemp()
        dggrid = DGGRIDv7(dggrid_path, working_dir=working_dir, silent=True)
        centroids = dggrid.cells_for_geo_points(centroids, True, self.grid_name, self.level, output_address_type='Z7_STRING')
        return centroids['name'].values


@register_dggs("IGEO7")
@register_dggs("igeo7")
class IGEO7Index(DGGSIndex):
    _grid: DGGSInfo

    def __init__(
        self,
        cell_ids: Any | PandasIndex,
        dim: str,
        grid_info: DGGSInfo,
    ):
        if not isinstance(grid_info, IGEO7Info):
            raise ValueError(f"grid info object has an invalid type: {type(grid_info)}")
        super().__init__(cell_ids, dim, grid_info)

    @classmethod
    def from_variables(cls: type["IGEO7Index"], variables: Mapping[Any, xr.Variable],
                       *, options: Mapping[str, Any],) -> "IGEO7Index":
        _, var, dim = _extract_cell_id_variable(variables)
        grid_name = var.attrs["grid_name"]
        cls = GRID_REGISTRY.get(grid_name)
        if cls is None:
            raise ValueError(f"unknown DGGS grid name: {grid_name}")
        igeo7info = IGEO7Info.from_dict(var.attrs)
        return cls(var.data, dim, igeo7info)

    @classmethod
    def stack(cls, variables: Mapping[Any, xr.Variable], dim: Hashable):
        return cls.from_variables(variables, options={})

    def concat(self, indexes: Sequence[Self], dim: Hashable, positions: Iterable[Iterable[int]] | None = None) -> Self:
        attrs = indexes[0]._grid.to_dict()
        pd_indexes = [idx._pd_index.index for idx in indexes]
        pd_indexes = pd_indexes[0].append(pd_indexes[1:])
        return IGEO7Index.from_variables({dim: xr.Variable(dim, pd_indexes.values, attrs)}, options={})

    def create_variables(self, variables):
        var = list(variables.values())[0]
        var = xr.Variable(self._dim, self._pd_index.index, var.attrs)
        idx_variables = {self._dim: var}
        return idx_variables

    def _repr_inline_(self, max_width: int):
        return f"ISEAIndex(grid_name={self._grid.grid_name}, level={self._grid.level})"

    def sel(self, labels, method=None, tolerance=None):
        if method == "nearest":
            raise ValueError("finding nearest grid cell has no meaning")
        target = np.unique(list(labels.values())[0])
        key = list(labels.keys())[0]
        labels[key] = np.isin(self._pd_index.index.values, target)
        return self._pd_index.sel(labels, method=method, tolerance=tolerance)

    def _replace(self, new_pd_index: PandasIndex):
        return type(self)(new_pd_index, self._dim, self._grid)

    def to_pandas_index(self):
        return self._pd_index.index

    def cell_centers(self, cell_ids: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        data = cell_ids if (cell_ids is not None) else self._pd_index.index.values
        mp = self._grid.mp
        steps = self._grid.chunk[0]
        batch = int(np.ceil(data.shape[0] / steps))
        ntf = tempfile.NamedTemporaryFile()
        centroids = np.memmap(ntf.name, mode='w+', shape=(len(data), 2), dtype='float32')
        with ProcessPoolExecutor(mp) as executor:
            list(tqdm(executor.map(_gen_centroid_from_cellids, *zip(*[(i, steps,
                                   data[(i * steps): ((i * steps) + steps) if (((i * steps) + steps) < len(data)) else len(data)],
                                   self._grid.grid_name, self._grid.level, len(data), ntf.name) for i in range(batch)])), total=batch))
        return (centroids[:, 0], centroids[:, 1])

    def cell_boundaries(self, cell_ids: np.ndarray = None) -> List[shapely.Polygon]:
        data = cell_ids if (cell_ids is not None) else self._pd_index.index.values
        mp = self._grid.mp
        steps = self._grid.chunk[0]
        batch = int(np.ceil(data.shape[0] / steps))
        with ProcessPoolExecutor(mp) as executor:
            result = list(tqdm(executor.map(_gen_polygon_from_cellids, *zip(
                                            *[(data[(i * steps): ((i * steps) + steps) if (((i * steps) + steps) < len(data)) else len(data)],
                                            self._grid.grid_name, self._grid.level) for i in range(batch)])), total=batch))

        result = list(chain(*result))
        return result

    def cell_parents(self, cell_ids: np.ndarray = None, relative_level = -1) -> np.ndarray:
        if (relative_level > 0):
            raise ValueError('relative_level should be negative for parents')
        data = cell_ids if (cell_ids is not None) else self._pd_index.index.values
        mp = self._grid.mp
        steps = self._grid.chunk[0]
        batch = int(np.ceil(data.shape[0] / steps))
        ntf = tempfile.NamedTemporaryFile()
        cellidsize = self._grid.level + relative_level
        parent_cellids = np.memmap(ntf.name, mode='w+', shape=(len(data),), dtype=f'|S{relative_level}')
        with ProcessPoolExecutor(mp) as executor:
            list(tqdm(executor.map(_gen_parents_from_cellids, *zip(*[(i, steps,
                                   data[(i * steps): ((i * steps) + steps) if (((i * steps) + steps) < len(data)) else len(data)],
                                   relative_level, len(data), ntf.name) for i in range(batch)])), total=batch))
        return xr.DataArray(
            parent_cellids.astype(np.str_), coords={'cell_ids': data}, dims='cell_ids'
        )



    def polygon_for_extent(self, geoobj, src_epsg):
        transformer = Transformer.from_crs(f"EPSG:{src_epsg}", "EPSG:4326").transform
        try:
            geoobj = shapely.from_geojson(geoobj)
        except Exception as e:
            print(f'Invalid Extend : {e}')
        geoobj = shapely.ops.transform(transformer, geoobj)
        working_dir = tempfile.mkdtemp()
        dggrid = DGGRIDv7(dggrid_path, working_dir=working_dir, silent=True)
        df = dggrid.grid_cellids_for_extent(self._grid.grid_name, self._grid.level, clip_geom=geoobj, output_address_type='Z7_STRING')
        return df
