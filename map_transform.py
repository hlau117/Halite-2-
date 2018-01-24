import numpy as np

def player_number(player,replay):
    for index,name in enumerate(replay['player_names']):
        if player in name:
            return index

def get_map(replay, player):
    plt_stats = []
    for planet in replay['planets']:
        stats = [(int(planet['x']),int(planet['y'])), planet['id'], int(planet['r'])]
        plt_stats.append(stats)
    player = player
    new_array = np.zeros((replay['num_frames'],replay['width'], replay['height'],3))
    for i in range(replay['num_frames']):
        frame = replay['frames'][i]
        for j in range(len(plt_stats)):
            if str(plt_stats[j][1]) in frame['planets']:
                planet_id = str(plt_stats[j][1])
                planet_x = plt_stats[j][0][0]
                planet_y = plt_stats[j][0][1]
                planet_r = plt_stats[j][2]
                health = frame['planets'][planet_id]['health']
                production = frame['planets'][planet_id]['current_production']
                for x in range(2*planet_r+1):
                    for y in range(2*planet_r+1):
                        new_array[(i,planet_x+planet_r -x,planet_y+planet_r - y, 0)] = health/planet_r**2
                        if frame['planets'][planet_id]['owner'] == player_number(player, replay):
                            new_array[(i,planet_x+planet_r -x,planet_y+planet_r - y, 1)] = 1
                        elif frame['planets'][planet_id]['owner'] == None:
                            new_array[(i,planet_x+planet_r -x,planet_y+planet_r - y, 1)] = 0
                        else:
                            new_array[(i,planet_x+planet_r -x,planet_y+planet_r - y, 1)] = -1
                        new_array[(i,planet_x+planet_r -x,planet_y+planet_r - y, 2)] = production
                        y += 1
                    x += 1
                    y = y + 2*planet_r+1
        for player in frame['ships']:
            for ship in frame['ships'][player]:
                x = int(frame['ships'][player][ship]['x'])
                y = int(frame['ships'][player][ship]['y'])
                new_array[(i,x,y,0)] = new_array[(i,x,y,0)] + frame['ships'][player][ship]['health']
                if int(player) == player_number(player, replay):
                    new_array[(i,x,y,1)] = 1
                else:
                    new_array[(i,x,y,1)] = -1
    return new_array

def get_row_compressor(old_dimension, new_dimension):
    dim_compressor = np.zeros((new_dimension, old_dimension))
    bin_size = float(old_dimension) / new_dimension
    next_bin_break = bin_size
    which_row = 0
    which_column = 0
    while which_row < dim_compressor.shape[0] and which_column < dim_compressor.shape[1]:
        if round(next_bin_break - which_column, 10) >= 1:
            dim_compressor[which_row, which_column] = 1
            which_column += 1
        elif next_bin_break == which_column:

            which_row += 1
            next_bin_break += bin_size
        else:
            partial_credit = next_bin_break - which_column
            dim_compressor[which_row, which_column] = partial_credit
            which_row += 1
            dim_compressor[which_row, which_column] = 1 - partial_credit
            which_column += 1
            next_bin_break += bin_size
    dim_compressor /= bin_size
    return dim_compressor

def get_column_compressor(old_dimension, new_dimension):
    return get_row_compressor(old_dimension, new_dimension).transpose()

def compress_and_average(array, new_shape):
    # Note: new shape should be smaller in both dimensions than old shape
    return np.mat(get_row_compressor(array.shape[0], new_shape[0])) * \
           np.mat(array) * \
           np.mat(get_column_compressor(array.shape[1], new_shape[1]))

def resize(map_array):
    out_map = np.zeros((len(map_array),64,64,3))
    for i in range(len(map_array)):
        out_map[i,:,:,0] = compress_and_average(map_array[i,:,:,0],(64,64))
        out_map[i,:,:,0] = out_map[i,:,:,0]/np.max(out_map[i,:,:,0])
        out_map[i,:,:,1] = compress_and_average(map_array[i,:,:,1],(64,64))
        out_map[i,:,:,2] = compress_and_average(map_array[i,:,:,2],(64,64))
        if np.max(out_map[:,:,2]) > 0:
            out_map[:,:,2] = out_map[:,:,2]/np.max(out_map[:,:,2])
    return out_map

def map_transformer(replay, player):
    return( resize(get_map(replay,player)))

def resize_frame(map_array):
    out_map = np.zeros((64,64,3))
    out_map[:,:,0] = compress_and_average(map_array[:,:,0],(64,64))
    out_map[:,:,0] = out_map[:,:,0]/np.max(out_map[:,:,0])
    out_map[:,:,1] = compress_and_average(map_array[:,:,1],(64,64))
    out_map[:,:,2] = compress_and_average(map_array[:,:,2],(64,64))
    if np.max(out_map[:,:,2]) > 0:
        out_map[:,:,2] = out_map[:,:,2]/np.max(out_map[:,:,2])
    return out_map
