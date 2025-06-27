from xdggs_dggrid4py.utils import register_igeo7regridding_method
from dggrid4py import DGGRIDv7
import numpy as np
import geopandas as gpd
import tempfile
import os

try:
    dggrid_path = os.environ['DGGRID_PATH']
except KeyError:
    raise Exception("DGGRID_PATH env var not found")


@register_igeo7regridding_method
def centerpoint(i, j, icoords, jcoords, index_dir, data_shape, igeo7info):
    temp_dir = tempfile.TemporaryDirectory()
    dggrid = DGGRIDv7(dggrid_path, working_dir=temp_dir.name, silent=True)
    ichunk, jchunk = np.meshgrid(icoords, jcoords, indexing='ij')
    chunk = np.c_[ichunk.ravel(), jchunk.ravel()]
    xidx, yidx = igeo7info.coordinate.index('x'), igeo7info.coordinate.index('y')
    # offset of i and j , calculate the "global" index for stacked original data
    chunk_size = igeo7info.chunk
    ioffset = i * chunk_size[0] * data_shape[1]
    joffset = j * chunk_size[1]
    chunk = gpd.GeoSeries(gpd.points_from_xy(chunk[:, xidx], chunk[:, yidx]), crs=igeo7info.src_epsg).to_crs('wgs84')
    df = gpd.GeoDataFrame([0] * chunk.shape[0], geometry=chunk)
    result = dggrid.cells_for_geo_points(df, True, igeo7info.grid_name, igeo7info.level, output_address_type='Z7_STRING')
    cells = result['name'].astype(str).values
    block_statistic = {'not_assigned': 0, 'reused': 0}
    v, c = np.unique(cells, return_counts=True)
    block_statistic['reused'] = len(v[np.where(c > 1)[0]])

    cellids_memmap = tempfile.mkstemp(dir=index_dir)
    reindex_memmap = tempfile.mkstemp(dir=index_dir)
    cellids = np.memmap(cellids_memmap[1], mode='w+', shape=(len(result),), dtype='|S34')
    reindex = np.memmap(reindex_memmap[1], mode='w+', shape=(len(result),), dtype=int)
    for x, a in enumerate(range(0, len(cells), len(jcoords))):
        start = ioffset + joffset + (x * data_shape[1])
        end = a + len(jcoords) if (a + len(jcoords) < len(cells)) else len(cells)
        # centerpoint doesn't involve selection, len(cellids) == len(original batched coordinates)
        globalidx = start + (np.arange(a, end) % (len(jcoords)))  # add global index offset to local index.
        cellids[a: end] = cells[a: end]
        reindex[a: end] = globalidx
    return (len(cells), cellids_memmap[1], reindex_memmap[1], block_statistic)
