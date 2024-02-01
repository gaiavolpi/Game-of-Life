import matplotlib.pyplot as plt
import numpy.random as npr
import numpy as np
from glob import glob
import os
current_path = os.path.dirname(__file__)

def create_pattern(size, species, offset=[0, 0], noise=False, p_alive = 0.2):
    '''
        Add a given species to the initial seed of the game
    '''
    pattern = np.zeros(size, dtype=int)
    start_row, start_col = offset[0] + size[0] // 2 - 1, offset[1] + size[1] // 2 - 1
    pattern[start_row:start_row + species.shape[0], start_col:start_col + species.shape[1]] = species # 'import' the pattern 

    if noise==True:
        noise = np.zeros(size, dtype=int)
        noise[start_row:(start_row + species.shape[0]), start_col:(start_col + species.shape[1])] = npr.choice(a=[0,1], size=species.shape, p=[1-p_alive, p_alive])
        pattern = np.bitwise_xor(pattern, noise)
        
    return pattern


def create_random(size, p_alive=0.5, world=None):
    '''
        Add random noise to the initial seed of the game
    '''
    if world is None: world = np.zeros(size, dtype=int) 
    random_pattern = npr.choice(a=[0,1], size=size, p=[1-p_alive, p_alive]) # random pattern 
    return np.bitwise_xor(random_pattern, world)

def rle_decoder(file):
    '''
        Decodes Game of Life patterns stored in .rle format converting them into
        2d binary numpy arrays
    '''
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if not line.startswith('#')]
   
    # Parse pattern size from the first line
    size_line = lines.pop(0)
    size = [int(s.split('=')[1]) for s in size_line.split(',')[:2]][::-1]
    
    # Draw blank grid
    grid = np.zeros(size, dtype=np.int8)
    
    # Populate grid according to rle encoding
    is_num = False
    col, row = 0, 0
    num = ''
    for line in lines:
        for c in line:
            if c.isdigit():
                num += c
                is_num = True    
            if c == '$':
                if is_num: skip = int(num)
                else: skip = 1
                row += skip
                col = 0
                is_num = False
                num = ''
            if not c.isdigit() and c != '$':
                if is_num: num = int(num)
                else: num = 1
                
                if c == 'o':
                    for i in range(num): grid[row, col+i] = 1
                col += num
                num = ''
                is_num = False
                    
    return grid


pattern_db = {
    # Database of patterns contained in the folder ./patterns/
    os.path.basename(path)[:-4]:rle_decoder(path) for path in glob(current_path + '/patterns/*.rle')
}
