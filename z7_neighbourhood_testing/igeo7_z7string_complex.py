import numpy as np

directions = np.array([
           complex(0,0), # 0
           complex(0.5,np.sqrt(3)/2), #1
           complex(0.5,-np.sqrt(3)/2), #2
           complex(1,0), #3
           complex(-1,0), #4
           complex(-0.5,np.sqrt(3)/2), #5
           complex(-0.5,-np.sqrt(3)/2) #6
])

scale_down = 1 / np.sqrt(7)
# angele from CPI paper by Kevin
angle = np.sqrt(3 / 28)
# Euler complex represetation of ccw 19 degrees
rotate_ccw19 = complex(np.cos(-angle), np.sin(-angle))


def _genpath(cell_id):
    # path = [directions[int(cell_id[0])]]
    path = np.array([[directions[int(c)], (scale_down**i * rotate_ccw19) if (i % 2 != 0) else (scale_down**i)] for i, c in enumerate(cell_id)])
    path = path[:, 0] * path[:, 1]
    path = np.add.accumulate(path)
    return path


# get neighbours of neighbours
def _get_neighbours(neighbours, level, target=[]):
    rotation = rotate_ccw19 if (level % 2 == 1) else 1
    # each neighbours's neighbours will contains original neighbours and the target
    # but it is not correct to exclude those "duplicated" neigbhours to reduce computation.
    # since it's neighbour may be the zero cell itself.
    exclude = target
    neighbours = np.repeat(neighbours, 7).reshape(len(neighbours), -1)
    scaled_rotated_directions = directions * (scale_down**(level) * rotation)
    result = neighbours - scaled_rotated_directions
    idx, jdx = np.where(np.abs(result - exclude) < 1e-9)
    result[idx, jdx] = np.inf
    result[:, 0] = np.inf
    return result


def find_neighbours(cell_id):
    if (cell_id[-1] == '0'):
        return np.array([cell_id[: -1] + f'{i}' for i in range(1, 7)])
    # omit the prefix face id at current stage.
    facesid = cell_id[:2]
    cell_id = cell_id[2:]
    path = _genpath(cell_id)
    # initialize at target cell level
    current_pos = path[-1]
    rotation = rotate_ccw19 if ((len(cell_id) - 1) % 2 == 1) else 1
    neighbours = current_pos + directions[1:] * (scale_down**(len(cell_id) - 1) * rotation)
    # since we start roll back at -2 , we have to push the for loop to run one more time
    # and we can't use any numerical value as it will crash with path location
    new_path = np.concatenate([[np.inf], path])
    neighbours_cellid = []
    # we iterate from the back, start from -2
    for i, p in enumerate(new_path[-2::-1], 2):
        # we are working at level = p + 1
        level = (len(cell_id) - i + 1) if (p != np.inf) else 0
        p = 0 if (p == np.inf) else p
        # get the ratation of p level
        rotation = rotate_ccw19 if ((level - 1) % 2 == 1 and (p != np.inf)) else 1
        # print(f'{level} {p} {rotation}')
        # get the neighbour's neighbours at working level
        nn = _get_neighbours(neighbours, level, [current_pos])
        # get all near by zero cells location at working level, we have to rotate it by rotation at p level
        # here we use the property that zero cells are 3 unit away from each other at 6 direction (center at any zero cell)
        scaled_directions = directions * (np.power(scale_down, level) * rotation)
        # we center at the p
        nearby_zeros = (p + 3 * scaled_directions).reshape(len(directions), 1, 1)
        # expand the dim to perform pair-wise distance between neighbour's neighbours and nearby zeros (7 direction, including center)
        # 7 x neighbours x 7
        distance_matrix = np.abs(np.repeat(nn[np.newaxis, :, :], len(directions), axis=0) - nearby_zeros) / (np.power(scale_down, level))
        nearest_zeros = np.argmin(distance_matrix, axis=0)  # n x 7
        distance_matrix = np.min(distance_matrix, axis=0)  # n x 7
        # print(distance_matrix)
        # There are some cases that the distance to nearest zeros is 0.8 ~ 0.9 while all others are 1
        # However, the difference is not significant to be consider "closest", as the distance between cells is 1 unit.
        # so the distance is clipped with 0.5 (1 unit between each pair of cell, so 0.5 )
        distance_matrix = np.where(1 - distance_matrix < 0.5, 1, distance_matrix)
        # we choose the minimum distance (to the near by zero) as the direction where the neighbours come from.
        # but in the case with all distance is 1 mean that the neighbours itself is the center 0 at p
        reverse_direction = np.argmin(distance_matrix, axis=-1)
        # print(reverse_direction)
        neighbours_cellid.append(reverse_direction)
        # Prepare for next level. Move all neighbours to the neaest zero. Those will become the neighbours in next level up
        nearby_zeros = nearby_zeros.reshape(7, )
        nearest_zeros = nearest_zeros[np.arange(nearest_zeros.shape[0]), reverse_direction]
        neighbours = nearby_zeros[nearest_zeros]
    neighbours_cellid = np.array(neighbours_cellid)
    fid = np.zeros((2, neighbours_cellid.shape[1])).astype(str)
    fid[0, :], fid[1, :] = facesid[1], facesid[0]
    neighbours_cellid = np.concatenate([neighbours_cellid, fid], axis=0)
    neighbours_cellid = np.rot90(neighbours_cellid, axes=(1, 0)).astype(str)
    neighbours_cellid = np.apply_along_axis(''.join, -1, neighbours_cellid)
    return neighbours_cellid




