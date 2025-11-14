from dggrid4py import DGGRIDv8
from dggrid4py.auxlat import geoseries_to_authalic, geoseries_to_geodetic
import geopandas as gpd
from geopandas.geoseries import GeoSeries
import shapely
import tempfile
import numpy as np
import os


regridding_method = {}


def register_regridding_method(func):
    regridding_method[func.__name__] = func
    print(f'Registered regridding method {func.__name__}')
    return func

# Alway returns a GeoSeries
def _authalic_to_geodetic(geometry, convert: bool) -> GeoSeries:
    if (not isinstance(geometry, GeoSeries)):
        geometry = GeoSeries(geometry)
    if (not convert):
        return geometry
    return geoseries_to_geodetic(geometry)


# Alway returns a GeoSeries
def _geodetic_to_authalic(geometry, convert: bool) -> GeoSeries:
    if (not isinstance(geometry, GeoSeries)):
        geometry = GeoSeries(geometry)
    if (not convert):
        return geometry
    return geoseries_to_authalic(geometry)


def autoResolution(minlng, minlat, maxlng, maxlat, src_epsg, num_data):
    try:
        dggrid_path = os.environ['DGGRID_PATH']
    except KeyError:
        raise Exception("DGGRID_PATH env var not found")
    dggs = DGGRIDv8(dggrid_path, working_dir=tempfile.mkdtemp(), silent=True)
    print('Calculate Auto resolution')
    df = gpd.GeoDataFrame([0], geometry=[shapely.geometry.box(minlng, minlat, maxlng, maxlat)], crs=src_epsg)
    print(f'Total Bounds ({df.crs}): {df.total_bounds}')
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




