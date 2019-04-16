import numpy as np
from Nelder_Mead_Solver import nelder_mead_solve

# Example script for utilizing this nelder mead solver

def f_noise_free(x0):
    """ Function to minimize"""
    x1 = x0[0]
    x2 = x0[1]
    return (0.065 * x1**4) - (x1**2) + (0.7 * x1) + (0.065 * x2**4) - (x2**2) + (0.9 * x2) + (0.3 * x1 * x2)

if __name__ == '__main__':
    """ Compute the Noise-Free result"""
    initial_guess = np.array([0, 0, -1, 0, 0, 1]).reshape(3,2)              # Initial Value
    result_vertex = nelder_mead_solve(f_noise_free, initial_guess, 5000)    # Solve for max interations 5000
    result_vertex_centroid = result_vertex.sum(axis=0) / 3                  # Find centroid
    print("Centroid is: ", result_vertex_centroid)                          # Print out centroid
    
