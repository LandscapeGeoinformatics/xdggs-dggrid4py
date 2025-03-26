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
    path = [directions[int(cell_id[0])]]
    for i, c in enumerate(cell_id[1:], 1):
        scaled_rotated = directions[int(c)] * ((scale_down**i * rotate_ccw19) if (i % 2 != 0) else (scale_down**i))
        path += [path[-1] + scaled_rotated]
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
    path = _genpath(cell_id)
    # initialize at target cell level
    current_pos = path[-1]
    rotation = rotate_ccw19 if ((len(cell_id) - 1) % 2 == 1) else 1
    neighbours = current_pos + directions[1:] * (scale_down**(len(cell_id) - 1) * rotation)
    new_path = [0] + path
    neighbours_cellid = []
    # we iterate from the back, start from -2
    for p in new_path[-2::-1]:
        # we are working at level = p + 1
        level = path.index(p) + 1 if (p != 0) else 0
        p = new_path[0] if (p == 0) else p
        # get the ratation of p level
        rotation = rotate_ccw19 if ((level - 1) % 2 == 1 and (p != 0)) else 1
        # get the neighbour's neighbours at working level
        # print(f'{p} {level} {rotation} {(np.power(scale_down,level))}')
        nn = _get_neighbours(neighbours, level, [current_pos])
        # get all near by zero cells location at p level
        # here we use the property that zero cells are 3 unit away from each other at 6 direction (center at any zero cell)
        scaled_directions = directions * (np.power(scale_down, level) * rotation)
        # we center at the p
        nearby_zeros = (p + 3 * scaled_directions).reshape(len(directions), 1, 1)
        # expand the dim to perform pair-wise distance between neighbour's neighbours and nearby zeros (7 direction, including center)
        distance_matrix = np.abs(np.repeat(nn[np.newaxis, :, :], len(directions), axis=0) - nearby_zeros) / (np.power(scale_down, level))
        distance_matrix = np.min(distance_matrix, axis=0)
        # There are some cases that the distance to nearest zeros is 0.8 ~ 0.9 while all others are 1
        # However, the distance is not significant to be consider "closest",
        # so the distance is clipped with 0.5 (1 unit between each pair of cell, so 0.5 )
        # remind that all distance is 1 mean that the neighbours itself is 0
        distance_matrix = np.where(1 - distance_matrix < 0.5, 1, distance_matrix)
        # we choose the minimum distance (to the near by zero) as the direction where the neighbours come from.
        reverse_direction = np.argmin(distance_matrix, axis=-1)
        # print(reverse_direction)
        neighbours_cellid.append(reverse_direction)
        # Prepare for next level. Move all neighbours to the neaest zero. Those will become the neighbours in next level up
        neighbours = neighbours - directions[reverse_direction] * (np.power(scale_down, level) * rotation)
        nearby_zeros = nearby_zeros.reshape(7, )
        neighbours = np.abs(np.repeat(neighbours[:, np.newaxis], len(directions), axis=-1) - nearby_zeros) / (np.power(scale_down, level))
        neighbours = np.where(neighbours < 1)[1]
        neighbours = nearby_zeros[neighbours]
    neighbours_cellid = np.rot90(np.array(neighbours_cellid), axes=(1, 0)).astype(str)
    neighbours_cellid = np.apply_along_axis(''.join, -1, neighbours_cellid)
    return neighbours_cellid




