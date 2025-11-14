from xdggs_dggrid4py.utils import autoResolution, regridding_method
from xdggs_dggrid4py.dependences.grids import GridsConfig
import xarray as xr
import pandas as pd
import tempfile
from dask.dataframe import from_pandas
from dask.diagnostics import ProgressBar


def regridding(ds: xr.Dataset, grid_name, method="nearestpoint", original_crs=None,
               refinement_level=-1, coordinates=['x', 'y'],
               _wgs84_geodetics_conversion=True, _dggs_vert0_lon=11.20) -> xr.Dataset:

    if (grid_name.upper() not in list(GridsConfig.keys())):
        raise ValueError(f"{__name__} {grid_name} not found in GridsConfig.")
    if (regridding_method.get(method) is None):
        raise ValueError(f"{__name__} {method} not found in regridding_method.")
    try:
        original_crs = ds.spatial_ref.attrs['crs_wkt']
    except AttributeError:
        print(f"{__name__} No `spatial_ref` found in the dataset")
        if (original_crs is None):
            raise Exception(f"{__name__} No original CRS is defined.")
        pass
    if (refinement_level == -1):
        minx, miny = ds[coordinates[0]].min().values, ds[coordinates[1]].min().values
        maxx, maxy = ds[coordinates[0]].max().values, ds[coordinates[1]].max().values
        total_rows = len(ds[coordinates[0]]) * len(ds[coordinates[1]])
        refinement_level = autoResolution(minx, miny, maxx, maxy, original_crs, total_rows)
        print(f"{__name__} autoResolution: {refinement_level}")
    spatial_ref_attrs = ds.spatial_ref.attrs.copy()
    grid_name = grid_name.upper()
    dggrid_meta_config = GridsConfig[grid_name]['meta_config']
    dggrid_meta_config.update({"dggs_vert0_lon": _dggs_vert0_lon})
    dask_df = ds.to_dask_dataframe()
    columns_meta = {c: pd.Series(dtype=t) for c, t in ds.items()}
    columns_meta.update({'zone_id': pd.Series(dtype=object)})
    metadf = from_pandas(pd.DataFrame(columns_meta).set_index('zone_id'))
    tempdir = tempfile.TemporaryDirectory()
    ProgressBar().register()
    dask_df = dask_df.map_partitions(regridding_method[method], coordinates, tempdir.name, grid_name,
                                     refinement_level, original_crs, dggrid_meta_config, meta=metadf).compute()
    ds = dask_df.to_xarray()
    ds = ds.assign_coords({'spatial_ref': 0})
    ds.spatial_ref.attrs = spatial_ref_attrs
    return ds
