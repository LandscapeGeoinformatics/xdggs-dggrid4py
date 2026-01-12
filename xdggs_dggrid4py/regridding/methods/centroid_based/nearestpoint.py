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
from pys2index import S2PointIndex
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
                           grid_name, refinement_level, crs, assign_zones_to_data, dggrid_meta_config, wgs84_to_authalic=True, zone_id_repr='textural',
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
    # coordinates in (x ,y) format. The order of the 1D array follows
    # transposing row to column with fixed y, example:
    #            x       ,   y
    #   [ 564406.00767268, 5557346.30067022],
    #   [ 564422.00767268, 5557346.30067022],
    #   [ 564438.00767268, 5557346.30067022],
    points_coordinates = np.c_[x_coordinates.ravel(), y_coordinates.ravel()]

    # reshape the data , e.g. (1, 200, 200) (c,y,x) -> (200, 200, 1) (x,y,c)
    # since the points_coordinates is in (x,y), we also need to use moveaxis  (c, y, x) -> (y, x, c)
    # and vstack to make sure the x,y ordering
    # example: a=np.array([[[0,1],[2,3]],[[4,5],[6,7]],[[8,9],[10,11]]]) # (3,2,2)
    #          np.vstack(np.moveaxis(a, 0, -1))
    #          array([[ 0,  4,  8],  # (0,0) (x,y)
    #                 [ 1,  5,  9],  # (1,0)
    #                 [ 2,  6, 10],  # (0,1)
    #                 [ 3,  7, 11]]) # (1,1)
    num_of_data_variables = data.shape[0]
    data = np.moveaxis(data, 0, -1)  # (y, x, c)
    data = np.vstack(data)
    data_points = gpd.GeoSeries([shapely.Point(point[0], point[1]) for point in points_coordinates], crs=crs).to_crs('wgs84')
    clip_bound = shapely.box(*data_points.total_bounds)
    clip_bound = _geodetic_to_authalic(clip_bound, wgs84_to_authalic)[0]
    hex_centroids_df = dggrid.grid_cell_centroids_for_extent(grid_name, refinement_level, clip_geom=clip_bound, **dggrid_meta_config).set_crs('wgs84')
    hex_centroids_df['geometry'] = _authalic_to_geodetic(hex_centroids_df['geometry'], wgs84_to_authalic, False)
    if (zone_id_repr == 'int'):
        hex_centroids_df['name'] = hex_centroids_df['name'].apply(int, base=16)
    data_points = data_points.get_coordinates()
    data_points = np.c_[data_points.y, data_points.x]
    hex_centroids = hex_centroids_df.get_coordinates()
    hex_centroids = np.c_[hex_centroids.y, hex_centroids.x]
    if (assign_zones_to_data):
        data_points = S2PointIndex(data_points)
        centroids_idx = data_points.query(hex_centroids)[1]  # the len of the position array = hex_centroids
    else:
        hex_centroids = S2PointIndex(hex_centroids)
        centroids_idx = hex_centroids.query(data_points)[1]  # the len of the position array = dat_points
    no_data = zone_id_repr_list[zone_id_repr][1]
    result_block = da.full((result_block_size, num_of_data_variables + 1), no_data, dtype=object, chunks=-1)
    if (assign_zones_to_data):
        result_block[:len(centroids_idx), :num_of_data_variables] = data[centroids_idx]  # .astype(object)
        result_block[:len(centroids_idx), -1] = hex_centroids_df['name'].values
    else:
        result_block[:len(centroids_idx), :num_of_data_variables] = data
        result_block[:len(centroids_idx), -1] = hex_centroids_df.iloc[centroids_idx]['name'].values
    return result_block
