from dggrid4py import DGGRIDv8
from pygeodesy.ellipsoids import Ellipsoids
import geopandas as gpd
from geopandas.geoseries import GeoSeries
import shapely
import tempfile
import numpy as np
import os


regridding_method = {}

wgs84 = Ellipsoids.WGS84


def register_regridding_method(func):
    regridding_method[func.__name__] = func
    print(f'Registered regridding method {func.__name__}')
    return func


def v_authalic_to_geodetic(x):
    return wgs84.auxAuthalic(x, inverse=True)


def v_geodetic_to_authalic(x):
    return wgs84.auxAuthalic(x, inverse=False)


def _create_polygon(list_of_point):
    return shapely.Polygon(list_of_point)


def _create_point(point):
    return shapely.Point(point)


v_authalic_to_geodetic = np.vectorize(v_authalic_to_geodetic)
v_geodetic_to_authalic = np.vectorize(v_geodetic_to_authalic)


# Alway returns a GeoSeries
def _authalic_to_geodetic(geometry, convert: bool, polygon: bool = True) -> GeoSeries:
    if (not isinstance(geometry, GeoSeries)):
        geometry = GeoSeries(geometry)
    if (not convert):
        return geometry
    if (polygon):
        lat_array = geometry.geometry.apply(lambda geom: np.array(geom.exterior.coords.xy[1]))
        lon_array = geometry.geometry.apply(lambda geom: np.array(geom.exterior.coords.xy[0]))
    else:
        lat_array = geometry.geometry.apply(lambda geom: np.array(geom.coords.xy[1]))
        lon_array = geometry.geometry.apply(lambda geom: np.array(geom.coords.xy[0]))
    lat_array = np.stack(lat_array.to_numpy())
    lon_array = np.stack(lon_array.to_numpy())
    lat_array = v_authalic_to_geodetic(lat_array)
    geom = np.stack([lon_array, lat_array], axis=-1)
    if (polygon):
        # stack lon_array,lat_array at the last dim, then convert the last dim to a 2-tuple
        # ex. 40000 polygons = [40000,7, 2] after stack, then change it to [40000,7]
        geom = geom.view(dtype=np.dtype([('x', 'float'), ('y', 'float')]))
        geom = geom.reshape(geom.shape[:-1])
        geom = np.apply_along_axis(_create_polygon, -1, geom)
    else:
        # stack lon_array,lat_array at the last dim, squeeze the dim
        # ex. 40000 polygons = [40000,1, 2] after stack, then squeeze it to [40000, 2]
        geom = geom.squeeze()
        geom = np.apply_along_axis(_create_point, -1, geom)

    return GeoSeries(geom)


# Alway returns a GeoSeries
def _geodetic_to_authalic(geometry, convert: bool, polygon: bool = True) -> GeoSeries:
    if (not isinstance(geometry, GeoSeries)):
        geometry = GeoSeries(geometry)
    if (not convert):
        return geometry
    if (polygon):
        lat_array = geometry.geometry.apply(lambda geom: np.array(geom.exterior.coords.xy[1]))
        lon_array = geometry.geometry.apply(lambda geom: np.array(geom.exterior.coords.xy[0]))
    else:
        lat_array = geometry.geometry.apply(lambda geom: np.array(geom.coords.xy[1]))
        lon_array = geometry.geometry.apply(lambda geom: np.array(geom.coords.xy[0]))
    lat_array = np.stack(lat_array.to_numpy())
    lon_array = np.stack(lon_array.to_numpy())
    lat_array = v_geodetic_to_authalic(lat_array)
    geom = np.stack([lon_array, lat_array], axis=-1)
    if (polygon):
        # stack lon_array,lat_array at the last dim, then convert the last dim to a 2-tuple
        # ex. 40000 polygons = [40000,7, 2] after stack, then change it to [40000,7]
        geom = geom.view(dtype=np.dtype([('x', 'float'), ('y', 'float')]))
        geom = geom.reshape(geom.shape[:-1])
        geom = np.apply_along_axis(_create_polygon, -1, geom)
    else:
        # stack lon_array,lat_array at the last dim, squeeze the dim
        # ex. 40000 polygons = [40000,1, 2] after stack, then squeeze it to [40000, 2]
        geom = geom.squeeze()
        geom = np.apply_along_axis(_create_point, -1, geom)

    return GeoSeries(geom)


def autoResolution(minlng, minlat, maxlng, maxlat, src_epsg, num_data, rf=-1):
    try:
        dggrid_path = os.environ['DGGRID_PATH']
    except KeyError:
        raise Exception("DGGRID_PATH env var not found")
    dggs = DGGRIDv8(dggrid_path, working_dir=tempfile.mkdtemp(), silent=True)
    # print('Calculate Auto resolution')
    df = gpd.GeoDataFrame([0], geometry=[shapely.geometry.box(minlng, minlat, maxlng, maxlat)], crs=src_epsg)
    #print(f'Total Bounds ({df.crs}): {df.total_bounds}')
    df = df.to_crs('wgs84')
    #print(f'Total Bounds (wgs84): {df.total_bounds}')
    R = 6371
    lon1, lat1, lon2, lat2 = df.total_bounds
    lon1, lon2, lat1, lat2 = np.deg2rad(lon1), np.deg2rad(lon2), np.deg2rad(lat1), np.deg2rad(lat2)
    a = (np.sin((lon2 - lon1) / 2) ** 2 + np.cos(lon1) * np.cos(lon2) * np.sin(0) ** 2)
    d = 2 * np.arcsin(np.sqrt(a))
    area = abs(d * ((np.power(R, 2) * np.sin(lat2)) - (np.power(R, 2) * np.sin(lat1))))
    #print(f'Total Bounds Area (km^2): {area}')
    avg_area_per_data = (area / num_data)
    #print(f'Area per center point (km^2): {avg_area_per_data}')
    dggrid_resolution = dggs.grid_stats_table('ISEA7H', 30)
    filter_ = dggrid_resolution[dggrid_resolution['Area (km^2)'] < avg_area_per_data]
    resolution = 5
    if (len(filter_) > 0):
        resolution = filter_.iloc[0, 0]
        #print(f'Auto resolution : {resolution}, area: {filter_.iloc[0,2]} km2')
    if (rf > -1):
        est_numberofcells = int(np.ceil(area / dggrid_resolution.iloc[rf, 2]))
    else:
        est_numberofcells = int(np.ceil(area / dggrid_resolution.iloc[resolution, 2]))
    return resolution, est_numberofcells


