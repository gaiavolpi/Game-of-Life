from IPython.display import display
from tqdm.notebook import tqdm, trange #loadingbar
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy.random as npr
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from scipy.signal import convolve2d
import numpy as np
import imageio
from importlib.machinery import SourceFileLoader
from IPython.display import HTML
from IPython.display import Video
from gameoflife.create_pattern import *

#plot and grid have to be separate functions because of how matplotlib animations work (imshow is an artist)
def draw_plot(fig, ax, grid):
    '''
        function that draws the state of the world on a grid as a colormap
    '''
    state = np.flip(grid)
    #change the values of on and off here
    #state[state==0] = 
    #state[state==1] = 
    plt.set_cmap('gray')
    return ax.imshow(state)

def draw_grid(title=''):
    '''
        function that defines the visual properties of the grid
    '''
    fig, ax = plt.subplots()
    plt.axis('off')
    plt.grid(True)
    plt.title(title)
    return fig, ax

class GameOfLife:

    def __init__(self, seed, title='', display=True):
        self.seed = seed
        self.state = self.seed
        if display:
            self.fig, self.ax = draw_grid(title)
            self.timer = self.ax.text(1, 1, '', color='white', fontsize=12) # number of generation on top of the axes

    def count_neighbors(self):
        '''
            Returns the count of alive neighbors for each cell of the grid as a 2d ndarray,
            using a convolution with periodic boundary conditions
        '''
        fmap = np.array(
            [[1, 1, 1], 
             [1, 0, 1], 
             [1, 1, 1]]
        )
        return convolve2d(self.state, fmap, mode='same', boundary='wrap') # convolution between the state and fmap

    def update(self, frame, display=True):
        '''
            Computes the state of the next generation using Conway's rules. 
            Returns the plot of the grid at the next generation with a generation counter on top
        '''
        if frame < 1:   # no movement for the first step
            self.state = self.seed
        else: 
            neighbors = self.count_neighbors()
            new_state = self.state.copy()
            
            for i in range(self.state.shape[0]):
                for j in range(self.state.shape[1]):
                    is_alive = self.state[i,j]
                    total = neighbors[i,j]
                    if (total < 2) or (total > 3): new_state[i,j] = 0
                    if total == 2 and is_alive == 1: new_state[i,j] = 1
                    if total == 3: new_state[i,j] = 1
            self.state = new_state
         # set the number of generation 
        if display:
            message = f"Gen.{frame}"
            self.timer.set_text(message)
            return draw_plot(self.fig, self.ax, self.state), self.timer
        else:
            return self.state

class StatGoL:
    
    def __init__(self, seed, title='', display=True, T=0):
        self.seed = seed
        self.state = self.seed
        self.rho = self.global_density()
        self.T = T
        self.display = display
        #Conway rules
        pb0 = np.array([0 if k!=3 else 1 for k in range(9)])
        ps0 = np.array([0 if k!=3 and k!=2 else 1 for k in range(9)])
        self.p0 = np.vstack((pb0, ps0))
        #display plot
        if display:
            self.fig, self.ax = draw_grid(title)
            self.pop = self.ax.text(1, 5, '', color='red', fontsize=12)
            self.timer = self.ax.text(1, 2, '', color='red', fontsize=12) # number of generation on top of the axes

    def count_neighbors(self):
        '''
            Returns the count of alive neighbors for each cell of the grid as a 2d ndarray,
            using a convolution with periodic boundary conditions
        '''
        fmap = np.array(
            [[1, 1, 1], 
             [1, 0, 1], 
             [1, 1, 1]]
        )
        return convolve2d(self.state, fmap, mode='same', boundary='wrap') # convolution between the state and fmap
    
    def global_density(self):
        N = self.state.size
        return np.sum(self.state)/N
    
    def update(self, frame):
        '''
            Computes the state of the next generation using Conway's rules. 
            Returns the plot of the grid at the next generation with a generation counter on top
        '''
        if frame < 1:   # no movement for the first step
            self.state = self.seed
        else:
            neighbors = self.count_neighbors()
            new_state = self.state.copy()
            for i in range(self.state.shape[0]):
                for j in range(self.state.shape[1]):
                    is_alive = self.state[i,j]
                    k = neighbors[i,j]
                    pT = (self.p0+self.rho*self.T)/(1+self.T)
                    do_givebirth = npr.choice([0, 1], p=[1-pT[1,k],pT[1, k]])
                    do_survive = npr.choice([0, 1], p=[1-pT[0,k],pT[0, k]])
                    if is_alive and do_givebirth==0:
                        new_state[i,j]=0
                    if not is_alive and do_survive==1:
                        new_state[i,j]=1
            self.state = new_state
         # set the number of generation 
        self.rho = self.global_density()
        if self.display:
            self.timer.set_text(f"Gen: {frame}")
            self.pop.set_text(f"Density: {self.rho:.2f}")
            return draw_plot(self.fig, self.ax, self.state), self.timer, self.pop
        else:
            return self.state

def play_gol(seed, n_gens, delta_t, title=''):
    '''
        Plays the game given the initial population(the seed) for a number of generations (n_gens)
        at period delta_t. Returns an animation FuncAnimation of plots for all the generations
    '''
    game = GameOfLife(seed, title)
    anim = FuncAnimation(game.fig, game.update, frames=n_gens, interval=delta_t)
    HTML(anim.to_jshtml());
    return anim

def occupancy_rate(seed, n_gens, title=''):
    '''
        Compute the occupancy plot
    '''
    
    game = GameOfLife(seed, title, display=False)

    # Lists to store generation numbers and corresponding occupancy rates
    generation_numbers = []
    occupancy_rates = []

    for frame in range(n_gens):
        # Calculate and store the occupancy rate at each generation
        total_cells = seed.size  #the dimension of the size that we chose  
        alive_cells = np.sum(game.state[game.state==1]) #number of alive cells 
        occupancy_rate = (alive_cells / total_cells) * 100 #occupancy rate as a percentage
        occupancy_rates.append(occupancy_rate)
        generation_numbers.append(frame + 1)

        # Update the game for the next generation
        game.update(frame, display=False)
        
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(generation_numbers, occupancy_rates, linestyle='-', color='b')
    ax.set_title('Occupancy Rate Over Time')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Occupancy Rate (%)')
    ax.grid(True)
    plt.show()

def period_of_replication(specie, size, n_gens): 
    '''
        Computes the period of a given specie
    '''
    
    seed = create_pattern(size, specie) 
    game = GameOfLife(seed, display=False) 
    current_state = np.copy(game.state)
    
    for frame in range(1, n_gens+1): # start from one because there is no update
        game.update(frame, display=False)
        conv = convolve2d(game.state, np.flip(specie), mode='same', boundary='wrap') # convolution between the current state and the specie we want to find
        if np.any(conv == np.sum(specie)): 
            print(f"Pattern repeated after {frame} generations")
            rep = True 
            break
        else:
            rep = False
    if not rep:
        print("Pattern did not repeat within the specified number of generations")

def play_statgol(init, n_gens, delta_t, title='', temp=0):
    '''
        Plays the game given the initial population(the seed) for a number of generations (n_gens)
        at period delta_t. Returns an animation FuncAnimation of plots for all the generations
    '''
    game = StatGoL(init, title, T=temp)
    anim = FuncAnimation(game.fig, game.update, frames=n_gens, interval=delta_t)
    HTML(anim.to_jshtml());
    return anim

def global_density_curve(t_lim=[-2, 1], n_runs=5, n_iter=100, size=[50, 50], n_steps=15, p=0.3, save=False):
    '''
        Compute the density at infinity as a function of temperature
    '''
    t_space = np.logspace(start = t_lim[0], stop = t_lim[1], num = n_steps)
    rho_space = []
    init = create_random(size, p_alive=p)
    for run in tqdm(range(n_runs), desc='Iterating over runs'):
        rho_space.append([])
        for temp in tqdm(t_space, desc='Iterating over temperatures', leave=False):
            game = StatGoL(seed=init, display=False, T=temp)
            for i in range(n_iter):
                game.update(i)
            rho_space[run].append(game.rho)
    
    rho_curve = np.mean(np.array(rho_space), axis=0)
    
    if save:
        np.save(f'./data/density_{t_lim}bounds_{n_runs}runs_{n_iter}iter_{n_steps}steps.npy', rho_curve)
        np.save(f'./data/temperature_{t_lim}bounds_{n_runs}runs_{n_iter}iter_{n_steps}steps.npy', t_space)

    return t_space, rho_curve

def plot_density(t, rho):
    fig, ax = plt.subplots()
    plt.title('Density temperature plot')
    ax.set_xscale('log')
    ax.set_xlabel('T')
    ax.set_ylabel(r'$\rho(\infty)$')
    ax.plot(t, rho, '.', color='black', label='Simulation')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.legend()
    plt.show()
        