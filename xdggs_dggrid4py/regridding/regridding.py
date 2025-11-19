from xdggs_dggrid4py.utils import autoResolution, regridding_method
from xdggs_dggrid4py.dependences.grids import GridsConfig, zone_id_repr_list
import xarray as xr
import dask.array as da
import numpy as np
import pandas as pd
import tempfile
from dask.dataframe import from_pandas
from dask.diagnostics import ProgressBar


def mapblocks_regridding(ds: xr.Dataset, grid_name, method="mapblocks_nearestpoint", coordinates=['x', 'y'], original_crs=None,
                         refinement_level=-1, zone_id_repr="textural", wgs84_geodetics_conversion=True, _dggs_vert0_lon=11.20) -> xr.Dataset:
    if (zone_id_repr.lower() not in list(zone_id_repr_list.keys())):
        raise ValueError(f"{__name__} {zone_id_repr} is not supported.")
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
    minx, miny = ds[coordinates[0]].min().values, ds[coordinates[1]].min().values
    maxx, maxy = ds[coordinates[0]].max().values, ds[coordinates[1]].max().values
    total_rows = len(ds[coordinates[0]]) * len(ds[coordinates[1]])
    auto_rf_level, estimate_number_of_zones = autoResolution(minx, miny, maxx, maxy, original_crs, total_rows, refinement_level)
    print(estimate_number_of_zones)
    if (refinement_level == -1):
        refinement_level = auto_rf_level
        print(f'{__name__} auto refinement level : {refinement_level}')
    spatial_ref_attrs = ds.spatial_ref.attrs.copy()
    ds = ds.drop_vars('spatial_ref')
    grid_name = grid_name.upper()
    dggrid_meta_config = GridsConfig[grid_name]['meta_config']
    dggrid_meta_config.update({"dggs_vert0_lon": _dggs_vert0_lon})
    zone_id_repr = zone_id_repr.lower()
    dggrid_meta_config.update({'input_hier_ndx_form': zone_id_repr_list[zone_id_repr][0],
                               'output_hier_ndx_form': zone_id_repr_list[zone_id_repr][0]})
    ds_dask_array = ds.to_dataarray().data
    # meta template
    columns_meta = {c: pd.Series(dtype=t) for c, t in ds.items()}
    columns_meta.update({'zone_id': pd.Series(dtype=object)})
    metadf = from_pandas(pd.DataFrame(columns_meta)).to_dask_array()
    tempdir = tempfile.TemporaryDirectory()
    ProgressBar().register()
    starting_coordinate = np.array([ds[coordinates[0]].min(), ds[coordinates[1]].max()])
    coordinate_step_size = abs(ds[coordinates[0]][0] - ds[coordinates[0]][1]).values
    # estimate the number of zones per block
    result_block_size = int(estimate_number_of_zones / np.prod(ds_dask_array.numblocks) * 1.5)
    print(result_block_size)

    ds_dask_array = ds_dask_array.map_blocks(regridding_method[method], meta=metadf, drop_axis=0, chunks=(-1, -1),
                                             **{'starting_coordinate': starting_coordinate,
                                                'coordinate_step_size': coordinate_step_size,
                                                'working_dir': tempdir.name,
                                                'result_block_size': result_block_size,
                                                'grid_name': 'IGEO7',
                                                'refinement_level': refinement_level,
                                                'crs': original_crs,
                                                'dggrid_meta_config': dggrid_meta_config,
                                                'wgs84_to_authalic': wgs84_geodetics_conversion,
                                                'zone_id_repr': zone_id_repr
                                                }).compute(scheduler='processes')
    # The result of map_blocks is still in 2D , it replaces the original block (e.g. 200,200) with (result_block_size, number_of_vars+1)
    # So, we have to concate all the blocks then reshape to form the final result (i.e. (num_blocks * result_block_size, number_of_vars+1) ).
    ds_dask_array = da.concatenate([ds_dask_array.blocks.ravel()], axis=1).reshape(-1, len(ds.data_vars) + 1)
    # select zone_id != no_data from the result
    if (zone_id_repr == 'int'):
        ds_dask_array = ds_dask_array[~ da.isnan(ds_dask_array[:, -1])]
    else:
        ds_dask_array = ds_dask_array[da.compute(da.where(ds_dask_array[:, -1] != ''))[0]]
    ds_dask_array = ds_dask_array.to_dask_dataframe(list(ds.data_vars.keys()) + ['zone_id']).set_index('zone_id')
    converted_ds = xr.Dataset.from_dataframe(ds_dask_array)
    for var_name, dtype in ds.data_vars.dtypes.items():
        converted_ds[var_name] = converted_ds.data_vars[var_name].astype(dtype)
    if (zone_id_repr == 'int'):
        converted_ds['zone_id'] = converted_ds['zone_id'].astype(np.uint64)
    converted_ds = converted_ds.assign_coords({'spatial_ref': 0})
    converted_ds.spatial_ref.attrs = spatial_ref_attrs
    converted_ds['zone_id'].attrs = {'grid_name': grid_name, 'level': refinement_level}
    return converted_ds
