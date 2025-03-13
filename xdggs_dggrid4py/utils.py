from dggrid4py import DGGRIDv7, dggs_types
import geopandas as gpd
import shapely
import tempfile
import numpy as np
import os

try:
    dggrid_path = os.environ['DGGRID_PATH']
except KeyError:
    raise Exception("DGGRID_PATH env var not found")


def _gen_cellids(i, j, idata, jdata, cellids_memmap, chunk_size, total_size, grid_name, resolution, method, src_epsg, xpos, ypos, lock):
    temp_dir = tempfile.TemporaryDirectory()
    dggrid = DGGRIDv7(dggrid_path, working_dir=temp_dir.name, silent=True)
    ichunk, jchunk = np.meshgrid(idata, jdata, indexing='ij')
    chunk = np.c_[ichunk.ravel(), jchunk.ravel()]
    # offset of i and j
    ioffset = i * chunk_size[0] * total_size[1]
    joffset = j * chunk_size[1]
    chunk = gpd.GeoSeries(gpd.points_from_xy(chunk[:, xpos], chunk[:, ypos]), crs=src_epsg).to_crs('wgs84')
    # nearestpoint
    if (method.lower() == 'nearestpoint'):
        mini, maxi, minj, maxj = np.min(idata), np.max(idata), np.min(jdata), np.max(jdata)
        if (xpos == 0):
            region = gpd.GeoSeries([shapely.geometry.box(mini, minj, maxi, maxj)], crs=src_epsg).to_crs('wgs84')
        else:
            region = gpd.GeoSeries([shapely.geometry.box(minj, mini, maxj, maxi)], crs=src_epsg).to_crs('wgs84')
        result = dggrid.grid_cell_centroids_for_extent(grid_name, resolution, clip_geom=region.geometry.values[0],
                                                     output_address_type='Z7_STRING')
        idx = result.geometry.sindex.nearest(chunk.geometry, return_all=False, return_distance=False)[1]
        cells = result.iloc[idx]['name'].astype(str).values
    # centerpoint
    elif (method.lower() == 'centerpoint'):
        df = gpd.GeoDataFrame([0] * chunk.shape[0], geometry=chunk)
        result = dggrid.cells_for_geo_points(df, True, grid_name, resolution, output_address_type='Z7_STRING')
        cells = result['name'].astype(str).values
    cellids = np.memmap(cellids_memmap, mode='r+', shape=(total_size[0] * total_size[1],), dtype='<U34')
    for x, a in enumerate(range(0, len(cells), len(jdata))):
        start = ioffset + joffset + (x * total_size[1])
        end1 = len(jdata) if (a + len(jdata) < len(cells)) else (len(cells) - a)
        end2 = a + len(jdata) if (a + len(jdata) < len(cells)) else len(cells)
        #print(f'io: {ioffset} jo:{joffset} start: {start}, end1: {end1}, end2: {end2}')
        #print(f'{i} {j} {start} {end1}')
        cellids[start: (start + end1)] = cells[a: end2]
        cellids.flush()


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

