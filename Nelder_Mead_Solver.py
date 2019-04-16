import numpy as np

# This solver only applies on minimizing functions with 2 variables
""" Nelder-Mead Algorithm from Wikipedia"""
class NMSolver:
    def __init__(self, f, vertices):
        """ External Inputs"""
        self.f = f
        self.vertices = vertices
        """ Internal Variables """
        self.x0 = np.empty(2)       # Centroid fx
        self.xe = np.empty(2)
        self.xr = np.empty(2)
        self.xc = np.empty(2)
        self.xnp1 = np.empty(2)     # Max fx (Worst)
        self.xn = np.empty(2)       # Second Max (Second Worst)
        self.x1 = np.empty(2)       # Min fx
        """ Parameters """
        self.alpha = 1.0
        self.gamma = 2.0
        self.rho = 0.5
        self.sigma = 0.5

        """ Flags """
        self.flag_loop = False      # Set to True when loop needed

    def order(self):
        self.flag_loop = False

        fx = []
        new_vertax = np.empty([3, 2])
        """ Order vertices according to the values at the vertices """
        for idx in range(3):
            fx.append(self.f(self.vertices[idx]))  # Get fx for all vertices

        f_order = np.argsort(fx)

        for idx in range(0, 3):
            x_pos = f_order[idx]
            new_vertax[idx][0] = self.vertices[x_pos][0]  # Order vertices
            new_vertax[idx][1] = self.vertices[x_pos][1]

        self.xnp1[0] = new_vertax[2][0]     # Update max val x(n+1)
        self.xnp1[1] = new_vertax[2][1]
        self.xn[0] = new_vertax[1][0]       # Update second max val x(n+1)
        self.xn[1] = new_vertax[1][1]
        self.x1[0] = new_vertax[0][0]       # Update min val x(n+1)
        self.x1[1] = new_vertax[0][1]
        self.vertices = new_vertax
        return self

    def calculate_x0(self):

        self.x0[0] = (self.vertices[0][0] + self.vertices[1][0]) / 2.0
        self.x0[1] = (self.vertices[0][1] + self.vertices[1][1]) / 2.0
        return self

    def reflection(self):
        self.xr[0] = self.x0[0] + self.alpha * (self.x0[0] - self.xnp1[0])  # Calculate xr
        self.xr[1] = self.x0[1] + self.alpha * (self.x0[1] - self.xnp1[1])

        fxr = self.f(self.xr)
        fx1 = self.f(self.x1)
        fxn = self.f(self.xn)

        if (fx1 <= fxr) and (fxr < fxn):
            """ Replace worst point """
            self.vertices[2][0] = self.xr[0]
            self.vertices[2][1] = self.xr[1]
            """ Loop to Order """
            self.flag_loop = True
        return self

    def expansion(self):
        fxr = self.f(self.xr)
        fx1 = self.f(self.x1)

        if fxr < fx1:
            self.xe[0] = self.x0[0] + (self.gamma * (self.xr[0] - self.x0[0]))
            self.xe[1] = self.x0[1] + (self.gamma * (self.xr[1] - self.x0[1]))

            fxe = self.f(self.xe)
            if fxe < fxr:
                """ Replace worst point """
                self.vertices[2][0] = self.xe[0]
                self.vertices[2][1] = self.xe[1]
                """ Loop to Order """
                self.flag_loop = True
            else:
                """ Replace worst point """
                self.vertices[2][0] = self.xr[0]
                self.vertices[2][1] = self.xr[1]
                """ Loop to Order """
                self.flag_loop = True
        return self

    def contraction(self):
        self.xc[0] = self.x0[0] + (self.rho * (self.xnp1[0] - self.x0[0]))
        self.xc[1] = self.x0[1] + (self.rho * (self.xnp1[1] - self.x0[1]))

        fxc = self.f(self.xc)
        fxnp1 = self.f(self.xnp1)

        if fxc < fxnp1:
            """ Replace worst point """
            self.vertices[2][0] = self.xc[0]
            self.vertices[2][1] = self.xc[1]
            """ Loop to Order """
            self.flag_loop = True
        return self

    def shrink(self):
        self.vertices[1][0] = self.x1[0] + (self.sigma * (self.vertices[1][0] - self.x1[0]))
        self.vertices[1][1] = self.x1[1] + (self.sigma * (self.vertices[1][1] - self.x1[1]))
        self.vertices[2][0] = self.x1[0] + (self.sigma * (self.vertices[2][0] - self.x1[0]))
        self.vertices[2][1] = self.x1[1] + (self.sigma * (self.vertices[2][1] - self.x1[1]))
        self.flag_loop = True
        return self

    # Get Functions
    def get_flag_loop(self):
        return self.flag_loop

    def get_vertex(self):
        return self.vertices


# Check the converge status base on triangule size
# Input     : 3x2 triangle vertex matrix
# Output    : True for converged, false for not yet converged
def compute_triangle_size(triangle_matrix):
    """ Compute the size of triangle. Require input to be a triangle matrix"""
    x_diff1 = np.sqrt((triangle_matrix[0][0] - triangle_matrix[1][0]) ** 2)
    x_diff2 = np.sqrt((triangle_matrix[1][0] - triangle_matrix[2][0]) ** 2)
    x_diff3 = np.sqrt((triangle_matrix[2][0] - triangle_matrix[0][0]) ** 2)

    y_diff1 = np.sqrt((triangle_matrix[0][1] - triangle_matrix[1][1]) ** 2)
    y_diff2 = np.sqrt((triangle_matrix[1][1] - triangle_matrix[2][1]) ** 2)
    y_diff3 = np.sqrt((triangle_matrix[2][1] - triangle_matrix[0][1]) ** 2)

    avg_x = (x_diff1 + x_diff2 + x_diff3) / 3
    avg_y = (y_diff1 + y_diff2 + y_diff3) / 3

    if avg_x < 0.0001 and avg_y < 0.0001:
        converget = True
    else:
        converget = False

    return converget


# Wrapper function for Nelder Mead Solver
# Required inputs:
# f             : Function to minimize
# vertices      : Initial 3x2 triangule Vertices
# iters         : Steps to solve
# Output: Final Converged Vertices
def nelder_mead_solve(f, vertices, iters):
    """ Initialize Algorithm Object """
    nm_alg = NMSolver(f=f, vertices=vertices)

    """ Iterations """
    for _ in range(iters):
        while True:

            nm_alg.order()
            nm_alg.calculate_x0()
            if nm_alg.get_flag_loop():
                break
            nm_alg.reflection()
            if nm_alg.get_flag_loop():
                break
            nm_alg.expansion()
            if nm_alg.get_flag_loop():
                break
            nm_alg.contraction()
            if nm_alg.get_flag_loop():
                break
            nm_alg.shrink()
            if nm_alg.get_flag_loop():
                break
        # Check if converged by triangule size
        if compute_triangle_size(nm_alg.get_vertex()):
            break

    return nm_alg.get_vertex()