from xdggs_dggrid4py.utils import register_igeo7regridding_method
from dggrid4py import DGGRIDv7
import numpy as np
import geopandas as gpd
import shapely
import tempfile
import os

try:
    dggrid_path = os.environ['DGGRID_PATH']
except KeyError:
    raise Exception("DGGRID_PATH env var not found")


@register_igeo7regridding_method
def nearestpoint(i, j, icoords, jcoords, index_dir, data_shape, igeo7info):
    temp_dir = tempfile.TemporaryDirectory()
    dggrid = DGGRIDv7(dggrid_path, working_dir=temp_dir.name, silent=True)
    ichunk, jchunk = np.meshgrid(icoords, jcoords, indexing='ij')
    chunk_org = np.c_[ichunk.ravel(), jchunk.ravel()]
    xidx, yidx = igeo7info.coordinate.index('x'), igeo7info.coordinate.index('y')
    chunk_size = igeo7info.chunk
    cellidsize=f'|S{igeo7info.level+2}'
    chunk = gpd.GeoSeries(gpd.points_from_xy(chunk_org[:, xidx], chunk_org[:, yidx]), crs=igeo7info.src_epsg).to_crs('wgs84')
    # nearestpoint
    mini, maxi, minj, maxj = np.min(icoords), np.max(icoords), np.min(jcoords), np.max(jcoords)
    if (xidx == 0):
        region = gpd.GeoSeries([shapely.geometry.box(mini, minj, maxi, maxj)], crs=igeo7info.src_epsg).to_crs('wgs84')
    else:
        region = gpd.GeoSeries([shapely.geometry.box(minj, mini, maxj, maxi)], crs=igeo7info.src_epsg).to_crs('wgs84')
    result = dggrid.grid_cell_centroids_for_extent(igeo7info.grid_name, igeo7info.level, clip_geom=region.geometry.values[0],
                                                   output_address_type='Z7_STRING')
    idx = chunk.geometry.sindex.nearest(result.geometry, return_all=False, return_distance=False)[1]
    block_statistic = {'not_assigned': 0, 'reused': 0}
    block_statistic['not_assigned'] = len(chunk) - len(np.unique(idx))
    v, c = np.unique(idx, return_counts=True)
    block_statistic['reused'] = len(v[np.where(c > 1)[0]])
    cells = result['name'].astype(str).values
    cellids_memmap, reindex_memmap = tempfile.mkstemp(dir=index_dir), tempfile.mkstemp(dir=index_dir)
    cellids = np.memmap(cellids_memmap[1], mode='w+', shape=(len(result),), dtype=cellidsize)
    reindex = np.memmap(reindex_memmap[1], mode='w+', shape=(len(result),), dtype=int)
    # offset of i and j , calculate the "global" index for stacked original data
    ioffset = i * chunk_size[0] * data_shape[1]
    joffset = j * chunk_size[1]
    idx = [ a%(len(jcoords)) + (ioffset + joffset + ((a//len(jcoords)) * data_shape[1])) for a in idx]
    cellids[:] = cells
    reindex[:] = idx # add global index offset to local index.
    cellids.flush()
    reindex.flush()
    return (len(cells), cellids_memmap[1], reindex_memmap[1], block_statistic)

