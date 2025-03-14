from collections.abc import Mapping

import numpy as np
import xarray as xr
from xarray.indexes import PandasIndex

from xdggs.index import DGGSIndex
from xdggs.grid import DGGSInfo
from xdggs.utils import register_dggs
from typing import Any, ClassVar, Sequence, Hashable, Iterable, List
from dataclasses import dataclass
try:
    from typing import Self, Tuple
except ImportError:  # pragma: no cover
    from typing_extensions import Self

from xdggs_dggrid4py.utils import _gen_cellids, _gen_centroid_from_cellids, _gen_polygon_from_cellids
from tqdm.auto import tqdm
from dggrid4py import DGGRIDv7, dggs_types
import geopandas as gpd
import shapely
import tempfile
import os
import time
from pyproj import Transformer
from itertools import chain
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

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

    valid_parameters: ClassVar[dict[str, Any]] = {"level": range(-1, 15), "method": ["centerpoint", "nearestpoint"]}

    def __post_init__(self):
        if (self.level not in self.valid_parameters['level']):
            raise ValueError("resolution must be an integer between 0 and 15")
        if self.method.lower() not in self.valid_parameters["method"]:
            raise ValueError("method {self.method.lower()} is not supported.")

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
        res = dggrid.guess_zstr_resolution(cell_ids[0], 'IGEO7', input_address_type='Z7_STRING')['resolution'][0]
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
        centroids = gpd.GeoDataFrame([0]*len(centroids), geometry=centroids, crs='wgs84')
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
        working_dir = tempfile.mkdtemp()
        self.dggrid = DGGRIDv7(dggrid_path, working_dir=working_dir, silent=True)
        if not isinstance(grid_info, IGEO7Info):
            raise ValueError(f"grid info object has an invalid type: {type(grid_info)}")
        super().__init__(cell_ids, dim, grid_info)

    @classmethod
    def from_variables(
        cls: type["IGEO7Index"],
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any],
    ) -> "IGEO7Index":

        igeo7 = [k for k in variables.keys() if (variables[k].attrs.get('grid_name','not_found').upper() == 'IGEO7')]
        var = variables[igeo7[0]]
        if (type(var.data) is np.ndarray):
            if (var.data.dtype == 'O'):
                return cls(var.data, 'cell_ids', IGEO7Info.from_dict(var.attrs))
        # prepare to generate hexagon _grid
        resolution = var.attrs.get("level", options.get("level", -1))
        grid_name = var.attrs.get("grid_name", options.get("grid_name", 'IGEO7')).upper()
        coords = var.attrs.get('coordinate', options.get('coordinate'))
        src_epsg = var.attrs.get("src_epsg", options.get("src_epsg", "wgs84"))
        method = var.attrs.get("method", options.get("method", "nearestpoint"))
        mp = var.attrs.get("mp", options.get('mp', 1))
        flipped = True if (coords.index('x') == 1) else False
        c1 = variables[coords[coords.index('x')]].data if (not flipped) else variables[coords[coords.index('y')]].data
        c2 = variables[coords[coords.index('y')]].data if (not flipped) else variables[coords[coords.index('x')]].data
        chunk = var.attrs.get("chunk", options.get('chunk', (len(c1), len(c2))))
        print(f'c1 shape: ({c1.shape}), c2 shape: ({c2.shape})')
        if (grid_name not in dggs_types):
            raise ValueError(f"{grid_name} is not defined in DGGRID")
        cellids = None
        # preparing job list by partitioning the extent into smaller block chunk
        batch = int(np.ceil(c1.shape[0] / chunk[0]))
        batch2 = int(np.ceil(c2.shape[0] / chunk[1]))
        job, job2 = np.meshgrid(np.arange(batch), np.arange(batch2), indexing='ij')
        jobs = np.c_[job.ravel(), job2.ravel()]
        job_size = chunk[0] * chunk[1]
        cellids_length = len(c1) * len(c2)
        xpos, ypos = 0, 1
        if (flipped):
            xpos, ypos = 1, 0
        # Auto Resolution
        maxc1, minc1, maxc2, minc2 = np.max(c1), np.min(c1), np.max(c2), np.min(c2)
        if (resolution == -1):
            if (not flipped):
                resolution = cls._autoResolution(minc1, minc2, maxc1, maxc2, src_epsg, (cellids_length), grid_name)
            else:
                resolution = cls._autoResolution(minc2, minc1, maxc2, maxc1, src_epsg, (cellids_length), grid_name)
        # Generate Cells ID
        print(f"--- Multiprocessing {mp} ---")
        print(f"---Generate Cell ID with resolution {resolution} by {method}, number or job: {len(jobs)}, job size: {job_size}, chunk: {chunk} ---")
        ntf = tempfile.NamedTemporaryFile()
        cellids = np.memmap(ntf.name, mode='w+', shape=(cellids_length,), dtype='<U34')
        start = time.time()
        lock = multiprocessing.Manager().Lock()
        with ProcessPoolExecutor(mp) as executor:
            list(tqdm(executor.map(_gen_cellids,
                                   *zip(*[(job[0], job[1],
                                        c1[(job[0] * chunk[0]): ((job[0] * chunk[0]) + chunk[0]) if (len(c1) > ((job[0] * chunk[0]) + chunk[0])) else len(c1)],
                                        c2[(job[1] * chunk[1]): ((job[1] * chunk[1]) + chunk[1]) if (len(c2) > ((job[1] * chunk[1]) + chunk[1])) else len(c2)],
                                        ntf.name, chunk, (len(c1), len(c2)), grid_name, resolution, method, src_epsg, xpos, ypos, lock)
                                        for job in jobs])), total=len(jobs)))
        print(f'cell generation time: ({time.time()-start})')
        print(f'Cell ID calcultion completed, unique cell id :{np.unique(cellids).shape[0]}')
        arrts = {'level': resolution, 'src_epsg': src_epsg, 'coordinate': coords, 'method': method,
                 'mp': mp, 'chunk': chunk, 'grid_name': grid_name}
        grid_info = IGEO7Info.from_dict(arrts | options)
        return cls(cellids.astype(np.str_), 'cell_ids', grid_info)

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

    def polygon_for_extent(self, geoobj, src_epsg):
        transformer = Transformer.from_crs(f"EPSG:{src_epsg}", "EPSG:4326").transform
        try:
            geoobj = shapely.from_geojson(geoobj)
        except Exception as e:
            print(f'Invalid Extend : {e}')
        geoobj = shapely.ops.transform(transformer, geoobj)
        df = self.dggrid.grid_cellids_for_extent(self._grid.grid_name, self._grid.level, clip_geom=geoobj, output_address_type='Z7_STRING')
        return df

    @classmethod
    def _autoResolution(cls: type["IGEO7Index"], minlng, minlat, maxlng, maxlat, src_epsg, num_data, grid_name):
        dggs = DGGRIDv7(dggrid_path, working_dir=tempfile.mkdtemp(), silent=True)
        print('Calculate Auto resolution')
        print(f'{minlat},{minlng},{maxlat},{maxlng}')
        df = gpd.GeoDataFrame([0], geometry=[shapely.geometry.box(minlng, minlat, maxlng, maxlat)], crs=src_epsg)
        print(f'Total Bounds ({src_epsg}): {df.total_bounds}')
        df = df.to_crs('wgs84')
        print(f'Total Bounds (wgs84): {df.total_bounds}')
        R = 6371
        lon1, lat1, lon2, lat2 = df.total_bounds
        lon1, lon2, lat1, lat2 = np.deg2rad(lon1), np.deg2rad(lon2), np.deg2rad(lat1), np.deg2rad(lat2)
        a = (np.sin((lon2 - lon1) / 2) ** 2 + np.cos(lon1) * np.cos(lon2) * np.sin(0) ** 2)
        d = 2 * np.arcsin(np.sqrt(a))
        area = abs(d * ((np.power(R, 2) * np.sin(lat2)) - (np.power(R, 2) * np.sin(lat1))))
        print(f'Total Bounds Area (km^2): {area}')
        avg_area_per_data = (area / num_data)
        print(f'Area per center point (km^2): {avg_area_per_data}')
        dggrid_resolution = dggs.grid_stats_table('ISEA7H', 30)
        filter_ = dggrid_resolution[dggrid_resolution['Area (km^2)'] < avg_area_per_data]
        resolution = 5
        if (len(filter_) > 0):
            resolution = filter_.iloc[0, 0]
            print(f'Auto resolution : {resolution}, area: {filter_.iloc[0,2]} km2')
        else:
            print(f'Auto resolution failed, using {resolution}')

        return resolution

