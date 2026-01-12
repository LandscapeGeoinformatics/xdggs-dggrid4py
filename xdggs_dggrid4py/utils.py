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


igeo7_grid_zones_stats = {0: {"Cells": 12, "Area (km^2)": 51006562.1724089, "CLS (km)": 8199.5003701},
                          1: {"Cells": 72, "Area (km^2)": 7286651.7389156, "CLS (km)": 3053.2232428},
                          2: {"Cells": 492, "Area (km^2)": 1040950.2484165, "CLS (km)": 1151.6430095},
                          3: {"Cells": 3432, "Area (km^2)": 148707.1783452, "CLS (km)": 435.1531492},
                          4: {"Cells": 24012, "Area (km^2)": 21243.8826207, "CLS (km)": 164.4655799},
                          5: {"Cells": 168072, "Area (km^2)": 3034.8403744, "CLS (km)": 62.1617764},
                          6: {"Cells": 1176492, "Area (km^2)": 433.5486249, "CLS (km)": 23.4949231},
                          7: {"Cells": 8235432, "Area (km^2)": 61.9355178, "CLS (km)": 8.8802451},
                          8: {"Cells": 57648012, "Area (km^2)": 8.8479311, "CLS (km)": 3.3564171},
                          9: {"Cells": 403536072, "Area (km^2)": 1.2639902, "CLS (km)": 1.2686064},
                          10: {"Cells": 2824752492, "Area (km^2)": 0.18057, "CLS (km)": 0.4794882},
                          11: {"Cells": 19773267432, "Area (km^2)": 0.0257957, "CLS (km)": 0.1812295},
                          12: {"Cells": 138412872012, "Area (km^2)": 0.0036851, "CLS (km)": 0.0684983},
                          13: {"Cells": 968890104072, "Area (km^2)": 0.0005264, "CLS (km)": 0.0258899},
                          14: {"Cells": 6782230728492, "Area (km^2)": 0.0000752, "CLS (km)": 0.0097855},
                          15: {"Cells": 47475615099432, "Area (km^2)": 0.0000107, "CLS (km)": 0.0036986},
                          16: {"Cells": 332329305696012, "Area (km^2)": 0.0000015348198699, "CLS (km)": 0.0013979246590466},
                          17: {"Cells": 2326305139872072, "Area (km^2)": 0.0000002192599814, "CLS (km)": 0.0005283658570631},
                          18: {"Cells": 16284135979104492, "Area (km^2)": 0.0000000313228545, "CLS (km)": 0.0001997035227209},
                          19: {"Cells": 113988951853731432, "Area (km^2)": 0.0000000044746935, "CLS (km)": 0.0000754808367233},
                          20: {"Cells": 797922662976120012, "Area (km^2)": 0.0000000006392419, "CLS (km)": 0.0000285290746744},
                           }


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
        geom = geom.squeeze(axis=1)
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
        geom = geom.squeeze(axis=1)
        geom = np.apply_along_axis(_create_point, -1, geom)
    return GeoSeries(geom)


def autoResolution(minlng, minlat, maxlng, maxlat, src_epsg, num_data, rf=-1):
    df = gpd.GeoDataFrame([0], geometry=[shapely.geometry.box(minlng, minlat, maxlng, maxlat)], crs=src_epsg)
    df = df.to_crs('wgs84')
    R = 6371
    lon1, lat1, lon2, lat2 = df.total_bounds
    lon1, lon2, lat1, lat2 = np.deg2rad(lon1), np.deg2rad(lon2), np.deg2rad(lat1), np.deg2rad(lat2)
    a = (np.sin((lon2 - lon1) / 2) ** 2 + np.cos(lon1) * np.cos(lon2) * np.sin(0) ** 2)
    d = 2 * np.arcsin(np.sqrt(a))
    area = abs(d * ((np.power(R, 2) * np.sin(lat2)) - (np.power(R, 2) * np.sin(lat1))))
    print(f'{__name__} area of extent (km^2): {area}')
    avg_area_per_data = (area / num_data)
    print(f'{__name__} average area per square grid (km^2): {avg_area_per_data}')
    filter_ = [k for k, v in igeo7_grid_zones_stats.items() if (igeo7_grid_zones_stats[k]['Area (km^2)'] < avg_area_per_data)]
    resolution = 5
    if (len(filter_) > 0):
        resolution = filter_[0]
    if (rf > -1):
        est_numberofcells = int(np.ceil(area / igeo7_grid_zones_stats[rf]['Area (km^2)']))
    else:
        est_numberofcells = int(np.ceil(area / igeo7_grid_zones_stats[resolution]['Area (km^2)']))
    return resolution, est_numberofcells
