from xdggs_dggrid4py.utils import (
    register_regridding_method,
    _authalic_to_geodetic,
    _geodetic_to_authalic
)
from dggrid4py import DGGRIDv8
import geopandas as gpd
import pandas as pd
import shapely
import os


@register_regridding_method
def nearestpoint(data: pd.DataFrame, coordinates, tempdir, grid_name, refinement_level,
                 crs, dggrid_meta_config, wgs84_to_authalic=True):
    try:
        dggrid_path = os.environ['DGGRID_PATH']
    except KeyError:
        raise Exception("DGGRID_PATH env var not found")
    dggrid = DGGRIDv8(dggrid_path, tempdir, silent=True)
    data.reset_index(drop=True, inplace=True)
    data_centroids = gpd.GeoSeries([shapely.Point(point[0], point[1]) for point in zip(data[coordinates[0]], data[coordinates[1]])],
                                   crs=crs).to_crs('wgs84')
    clip_bound = shapely.box(*data_centroids.total_bounds)
    clip_bound = _geodetic_to_authalic(clip_bound, wgs84_to_authalic)[0]
    hex_centroids_df = dggrid.grid_cell_centroids_for_extent(grid_name, refinement_level, clip_geom=clip_bound, **dggrid_meta_config).set_crs('wgs84')
    hex_centroids_df['geometry'] = _authalic_to_geodetic(hex_centroids_df['geometry'], wgs84_to_authalic)
    nearest_to_hex_centroids_idx = data_centroids.geometry.sindex.nearest(hex_centroids_df.geometry, return_all=False, return_distance=False)[1]
    data = data.iloc[nearest_to_hex_centroids_idx].copy()
    data['zone_id'] = hex_centroids_df['name']
    data = data.drop(coordinates + ['spatial_ref'], axis=1).set_index('zone_id')
    return data
