from dggrid4py import DGGRIDv7
import geopandas as gpd
import shapely
import tempfile
import numpy as np
import os

try:
    dggrid_path = os.environ['DGGRID_PATH']
except KeyError:
    raise Exception("DGGRID_PATH env var not found")

igeo7regridding_method = {}


def register_igeo7regridding_method(func):
    igeo7regridding_method[func.__name__] = func
    print(f'Registered regridding method {func.__name__}')
    return func


def _gen_centroid_from_cellids(batch, steps, cellids, grid_name, resolution, total_len, centroids_memmap):
    centroids = np.memmap(centroids_memmap, mode='r+', shape=(total_len, 2), dtype='float32')
    temp_dir = tempfile.TemporaryDirectory()
    dggrid = DGGRIDv7(dggrid_path, working_dir=temp_dir.name, silent=True)
    centroids_df = dggrid.grid_cell_centroids_from_cellids(cellids, grid_name, resolution, input_address_type='Z7_STRING',
                                                           output_address_type='Z7_STRING').set_index('name')
    df = gpd.GeoDataFrame(cellids, columns=['cellids'])
    df = df.set_index('cellids')
    df = centroids_df.join(df, how='right')
    centroids_xy = df.geometry.get_coordinates()
    end = (batch * steps) + steps if (((batch * steps) + steps) < total_len) else total_len
    centroids[(batch * steps): end, 0] = centroids_xy['x'].values
    centroids[(batch * steps): end, 1] = centroids_xy['y'].values
    centroids.flush()


def _gen_polygon_from_cellids(cellids, grid_name, resolution):
    temp_dir = tempfile.TemporaryDirectory()
    dggrid = DGGRIDv7(dggrid_path, working_dir=temp_dir.name, silent=True)
    polygon_df = dggrid.grid_cell_polygons_from_cellids(cellids, grid_name, resolution, input_address_type='Z7_STRING',
                                                        output_address_type='Z7_STRING').set_index('name')
    df = gpd.GeoDataFrame(cellids, columns=['cellids'])
    df = df.set_index('cellids')
    df = polygon_df.join(df, how='right')
    return df['geometry'].values

def _gen_parents_from_cellids(batch, steps, cellids, relative_level,total_len, cellids_memmap):
    cellidsize = len(cellids[0]) + relative_level
    parent_cellids = np.memmap(cellids_memmap, mode='r+', shape=(total_len,), dtype=f'|S{cellidsize}')
    end = (batch * steps) + steps if (((batch * steps) + steps) < total_len) else total_len
    parent_cellids[(batch * steps): end] = [c[: relative_level] for c in cellids]
    parent_cellids.flush()

def autoResolution(minlng, minlat, maxlng, maxlat, src_epsg, num_data, grid_name):
    dggs = DGGRIDv7(dggrid_path, working_dir=tempfile.mkdtemp(), silent=True)
    print('Calculate Auto resolution')
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
    est_numberofcells = int(np.ceil(area / dggrid_resolution.iloc[4,2]))
    resolution = 5
    if (len(filter_) > 0):
        resolution = filter_.iloc[0, 0]
        est_numberofcells = int(np.ceil(area / filter_.iloc[0,2]))
        print(f'Auto resolution : {resolution}, area: {filter_.iloc[0,2]} km2')
    else:
        print(f'Auto resolution failed, using {resolution}')

    return resolution, est_numberofcells




