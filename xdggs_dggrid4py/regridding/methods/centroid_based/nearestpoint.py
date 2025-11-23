from xdggs_dggrid4py.utils import (
    register_regridding_method,
    _authalic_to_geodetic,
    _geodetic_to_authalic,
    _create_point
)
from xdggs_dggrid4py.dependences.grids import zone_id_repr_list
from dggrid4py import DGGRIDv8
import dask.array as da
import xarray as xr
import numpy as np
import geopandas as gpd
import tempfile
import shapely
import os
import warnings
warnings.filterwarnings("ignore")


@register_regridding_method
def nearestpoint(data: xr.Dataset, original_crs, coordinates, grid_name,
                 refinement_level, dggrid_meta_config, wgs84_to_authalic=True):
    try:
        dggrid_path = os.environ['DGGRID_PATH']
    except KeyError:
        raise Exception("DGGRID_PATH env var not found")
    tempdir = tempfile.TemporaryDirectory()
    dggrid = DGGRIDv8(dggrid_path, tempdir.name, silent=True)

    data_centroids = np.concatenate([data['zone_id'][coordinates[0]].values.reshape(-1, 1),
                                     data['zone_id'][coordinates[1]].values.reshape(-1, 1)], axis=-1)

    data_centroids = np.apply_along_axis(_create_point, -1, data_centroids)
    data_centroids = gpd.GeoSeries(data_centroids, crs=original_crs).to_crs('wgs84')
    clip_bound = _geodetic_to_authalic(shapely.box(*data_centroids.total_bounds), wgs84_to_authalic)[0]
    hex_centroids_df = dggrid.grid_cell_centroids_for_extent(grid_name, refinement_level, clip_geom=clip_bound, **dggrid_meta_config).set_crs('wgs84')
    hex_centroids_df['geometry'] = _authalic_to_geodetic(hex_centroids_df['geometry'], wgs84_to_authalic, False)
    nearest_to_hex_centroids_idx = data_centroids.geometry.sindex.nearest(hex_centroids_df.geometry, return_all=False, return_distance=False)[1]
    data = data.isel({'zone_id': nearest_to_hex_centroids_idx})
    # force not to create index
    data = data.assign_coords(xr.Coordinates({'zone_id': hex_centroids_df['name'].values}, indexes={}))
    data = data.drop_vars(coordinates)
    return data


@register_regridding_method
def mapblocks_nearestpoint(data: da, starting_coordinate: np.array, coordinate_step_size, result_block_size, working_dir,
                           grid_name, refinement_level, crs, dggrid_meta_config, wgs84_to_authalic=True, zone_id_repr='textural',
                           block_info=None):
    # more on block_info :https://docs.dask.org/en/stable/generated/dask.array.map_blocks.html
    try:
        dggrid_path = os.environ['DGGRID_PATH']
    except KeyError:
        raise Exception("DGGRID_PATH env var not found")
    dggrid = DGGRIDv8(dggrid_path, working_dir, silent=True, capture_logs=True)
    # calcuate the x,y coordinates for the block from block_info
    # starting_coordinate gives the starting point of the whole extent in (x_min, y_max)
    # example array-location': [(0, 1), (1800, 2000), (1400, 1407)] , block, y, x
    x_block_location = da.array(block_info[0]['array-location'][2])
    y_block_location = da.array(block_info[0]['array-location'][1])
    x_start, x_end = starting_coordinate[0] + (x_block_location * coordinate_step_size)
    y_start, y_end = starting_coordinate[1] - (y_block_location * coordinate_step_size)
    x_coordinates = np.arange(x_start, x_end, coordinate_step_size)
    y_coordinates = np.arange(y_start, y_end, -coordinate_step_size)
    x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
    # coordinates in (x ,y) format
    points_coordinates = np.c_[x_coordinates.ravel(), y_coordinates.ravel()]

    # reshape the data , swap the data variables with x , e.g. (1, 200, 200) (c,y,x) -> (200, 200, 1) (x,y,c)
    # since the points_coordinates is in (x,y), we also need to flatten the data in the corret format
    data = da.swapaxes(data, 0, -1)
    num_of_data_variables = data.shape[-1]
    data = data.reshape(-1, num_of_data_variables)
    data_points = gpd.GeoSeries([shapely.Point(point[0], point[1]) for point in points_coordinates], crs=crs).to_crs('wgs84')
    clip_bound = shapely.box(*data_points.total_bounds)
    clip_bound = _geodetic_to_authalic(clip_bound, wgs84_to_authalic)[0]
    hex_centroids_df = dggrid.grid_cell_centroids_for_extent(grid_name, refinement_level, clip_geom=clip_bound, **dggrid_meta_config).set_crs('wgs84')
    hex_centroids_df['geometry'] = _authalic_to_geodetic(hex_centroids_df['geometry'], wgs84_to_authalic, False)
    if (zone_id_repr == 'int'):
        hex_centroids_df['name'] = hex_centroids_df['name'].apply(int, base=16)
    nearest_to_hex_centroids_idx = data_points.geometry.sindex.nearest(hex_centroids_df.geometry, return_all=False, return_distance=False)[1]
    no_data = zone_id_repr_list[zone_id_repr][1]
    result_dtype = object if (zone_id_repr != 'int') else np.float64
    result_block = da.full((result_block_size, num_of_data_variables + 1), no_data, dtype=result_dtype, chunks=-1)
    result_block[:len(nearest_to_hex_centroids_idx), :num_of_data_variables] = data[nearest_to_hex_centroids_idx]  # .astype(object)
    result_block[:len(nearest_to_hex_centroids_idx), -1] = hex_centroids_df['name'].values
    return result_block
