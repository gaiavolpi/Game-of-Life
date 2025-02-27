# Implementation of Conway's Game of Life
> Group project for Laboratory of Computational Physics course
> Authors: Giancarlo Saran Gattorno, Lucrezia Rossi, Alberto Salvador, Gaia Volpi

### Project description 

Conway's Game of Life is a cellular automaton that is played on a 2D square grid. Each cell on the grid can be either alive or dead, and they evolve according to the following rules:

- Any live cell with fewer than two or more than three live neighbours dies. 
- Any live cell with two or three live neighbours lives, unchanged, to the next generation.
- Any dead cell with exactly three live neighbours comes to life.

We also implemented a stochastic version of such game, introducing noise in the dynamics parametrized by a fictitious temperature T. As shown in the reference paper [Schulman, L.S., Seiden, P.E. Statistical mechanics of a dynamical system based on Conway's game of Life. J Stat Phys 19, 293–314 (1978)](https://link.springer.com/article/10.1007/BF01011727), the model displays a critical temperature. 


*Final_version.ipynb* contains the main visualization of our project. *Pygame_GoL.ipynb* contains instead the implementation of GoL using pygame, for a better visualization of complex patterns. It is also possible to manually insert a seed using this program. 



