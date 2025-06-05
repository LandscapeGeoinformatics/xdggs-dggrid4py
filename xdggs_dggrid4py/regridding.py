from xdggs_dggrid4py.utils import autoResolution
from xdggs_dggrid4py.IGEO7 import IGEO7Info
from xdggs_dggrid4py.utils import igeo7regridding_method
import xarray as xr
import numpy as np
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm.auto import tqdm

cells_id_array = None


def _initialize(ds: xr.Dataset):
    variables = ds.variables
    igeo7 = [k for k in variables.keys() if (variables[k].attrs.get('grid_name', 'not_found').upper() == 'IGEO7')]
    if (len(igeo7) == 0):
        raise ValueError('Couldn\'t found grid info.')
    var = variables[igeo7[0]]
    if (var.data.dtype == 'O'):
        raise ValueError('Dataset is already converted or incorrect format')
    var.attrs['grid_name'] =  var.attrs['grid_name'].upper()
    # prepare to generate hexagon _grid
    igeo7info = IGEO7Info(**var.attrs)
    xidx, yidx = igeo7info.coordinate.index('x'), igeo7info.coordinate.index('y')
    c1 = variables[igeo7info.coordinate[xidx]].data if (xidx == 0) else variables[igeo7info.coordinate[yidx]].data
    c2 = variables[igeo7info.coordinate[yidx]].data if (xidx == 0) else variables[igeo7info.coordinate[xidx]].data
    chunk = igeo7info.chunk if (igeo7info.chunk is not None) else (len(c1), len(c2))
    print(f'c1 shape: ({c1.shape}), c2 shape: ({c2.shape})', flush=True)
    # preparing job list by partitioning the extent into smaller block chunk
    batch = int(np.ceil(c1.shape[0] / chunk[0]))
    batch2 = int(np.ceil(c2.shape[0] / chunk[1]))
    job, job2 = np.meshgrid(np.arange(batch), np.arange(batch2), indexing='ij')
    jobs = np.c_[job.ravel(), job2.ravel()]
    cellids_length = len(c1) * len(c2)
    # Auto Resolution
    resolution = igeo7info.level
    est_numberofcells = 0
    if (igeo7info.level == -1):
        maxc1, minc1, maxc2, minc2 = np.max(c1), np.min(c1), np.max(c2), np.min(c2)
        if (xidx == 0):
            resolution, est_numberofcells = autoResolution(minc1, minc2, maxc1, maxc2, igeo7info.src_epsg, (cellids_length), igeo7info.grid_name)
        else:
            resolution, est_numberofcells = autoResolution(minc2, minc1, maxc2, maxc1, igeo7info.src_epsg, (cellids_length), igeo7info.grid_name)
        d = igeo7info.to_dict()
        d['level'] = resolution
        igeo7info = IGEO7Info.from_dict(d)

    return igeo7info, resolution, jobs, est_numberofcells, xidx, yidx


def _read_result(batch_cellids_memmap, batch_idx_memmap, end, cellidsize):
    global cells_id_array
    index = np.memmap(batch_idx_memmap, mode='r', shape=(end,), dtype=int)
    cellsid = np.memmap(batch_cellids_memmap, mode='r', shape=(end,), dtype=cellidsize)
    cells_id_array[index] = cellsid


def igeo7regridding(ds: xr.Dataset) -> xr.Dataset:
    global cells_id_array
    variables = ds.variables
    igeo7info, resolution, jobs, est_numberofcells, xidx, yidx = _initialize(ds)
    job_size = igeo7info.chunk[0] * igeo7info.chunk[1]
    chunk = igeo7info.chunk
    c1 = variables[igeo7info.coordinate[xidx]].data if (xidx == 0) else variables[igeo7info.coordinate[yidx]].data
    c2 = variables[igeo7info.coordinate[yidx]].data if (xidx == 0) else variables[igeo7info.coordinate[xidx]].data
    # Generate Cells ID
    print(f"--- Multiprocessing {igeo7info.mp}, jobs: {len(jobs)}, job size: {job_size}, chunk: {igeo7info.chunk}  ---", flush=True)
    print(f"--- Generate cells ID at level {igeo7info.level} by {igeo7info.method}", flush=True)
    ntf = tempfile.TemporaryDirectory()
    regridding_method = igeo7regridding_method[igeo7info.method]
    start = time.time()
    # Regriding
    # Each method should return a list of cellids and the corresponding index (global index) of stacked data
    # it is kind of similar to map reduce processing, each sub-processes produce small bactch result in files and read back by main process
    # each sub-process should return (number of cells, cellids memmap, idx memmap)
    with ProcessPoolExecutor(igeo7info.mp) as executor:
        result = list(tqdm(executor.map(regridding_method,
                               *zip(*[(job[0], job[1],
                                    c1[(job[0] * chunk[0]): ((job[0] * chunk[0]) + chunk[0]) if (len(c1) > ((job[0] * chunk[0]) + chunk[0])) else len(c1)],
                                    c2[(job[1] * chunk[1]): ((job[1] * chunk[1]) + chunk[1]) if (len(c2) > ((job[1] * chunk[1]) + chunk[1])) else len(c2)],
                                    ntf.name, (len(c1), len(c2)), igeo7info)
                                    for job in jobs])), total=len(jobs)))
    number_of_cells = []
    total_not_assigned = 0
    total_reused = 0
    for i in result:
        number_of_cells += [i[0]]
        if (i[3] is not None):
            total_not_assigned += i[3]['not_assigned']
            total_reused += i[3]['reused']
    total_cells = sum(number_of_cells)
    print(f'Re-assign data to cells, number of cells: {total_cells}, not assigned: {total_not_assigned}, reused: {total_reused}', flush=True)
    ds = ds.stack(cell_ids=igeo7info.coordinate, create_index=False).drop_vars(igeo7info.coordinate).chunk({'cell_ids': 'auto'})
    print('Stack completed', flush=True)
    cellids_memmap = tempfile.NamedTemporaryFile()
    # read back result and write it to the master array
    cellidsize = f'|S{igeo7info.level+2}'
    cells_id_array = np.memmap(cellids_memmap, mode='w+', shape=(ds.cell_ids.shape[0],), dtype=cellidsize)
    with ThreadPoolExecutor(3) as executor:
        list(tqdm(executor.map(_read_result,
                               *zip(*[(r[1], r[2], r[0], cellidsize) for i, r in enumerate(result)])),
                  total=len(result)))
    print(cells_id_array[0], flush=True)
    cells_id_array = xr.Coordinates({'cell_ids': cells_id_array}, indexes={})
    print('coordinate created', flush=True)
    ds = ds.assign_coords(cells_id_array)
    print('coordinate assigned', flush=True)
    ds['cell_ids'].attrs = igeo7info.to_dict()
    variables = ds.variables
    old = [k for k in variables.keys() if (variables[k].attrs.get('grid_name', 'not_found').upper() == 'IGEO7')]
    ds[old].attrs = None
    print(f'---Generation completed time: ({time.time()-start}), number of cells: {total_cells} ---', flush=True)
    print('Re-assign data completed', flush=True)
    return ds
