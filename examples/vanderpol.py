import jax
import jax.numpy as jnp
import numpy as np

from IPython.display import HTML
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import hj_reachability as hj

## VanderPol System

class VanderPol(hj.ControlAndDisturbanceAffineDynamics):

    def __init__(self, mu=2., gamma=1., N_Nodes=1, Nodes_u=None, b_u=None, B=None, G=None, ring=False, 
                 max_position_disturbance=0.25,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None): #pointless

        # review alpha, beta, and delta
        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-mu, -gamma]), 
                                        jnp.array([mu, gamma]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Ball(jnp.zeros(2), max_position_disturbance)
        
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

        ## This class allows the construction of networked VdP systems, but if N_nodes = 1 then we have a single system.

        self.mu, self.gamma = mu, gamma
        self.N_Nodes = N_Nodes
        if Nodes_u is None: Nodes_u = np.arange(N_Nodes)
        self.N_x, self.N_u, self.N_d = 2 * N_Nodes, len(Nodes_u), 2 * N_Nodes
        self.C = np.eye(self.N_x)

        ## If no B imported, assume one-to-one node control
        if B is None:
            self.B = np.zeros((self.N_Nodes, self.N_u))
            if b_u is None: b_u = np.ones_like(Nodes_u)

            for ui, uix in enumerate(Nodes_u):
                self.B[uix, ui] = b_u[ui]
        else:
            self.B = B

        if G is None and not(ring):
            self.G = -gamma * np.eye(N_Nodes)
        elif ring:
            self.G = -gamma * np.eye(N_Nodes) + np.diag(np.ones(N_Nodes - 1), -1) + np.diag(np.ones(N_Nodes - 1), 1) + np.diag([1], N_Nodes-1) + np.diag([1], -N_Nodes+1)
        else:
            self.G = G

    def f(self, t, x, u, d):

        xdot = np.zeros_like(x)
        xdot[::2]  = x[1::2]
        xdot[1::2] = self.mu * (1 - (x[::2] ** 2)) * x[1::2] + np.matmul(self.G, x[::2]) + np.matmul(self.B, u)
        xdot += np.matmul(self.C, d)

        return xdot
    
    def open_loop_dynamics(self, state, time):
        _, v, q = state
        return jnp.array([v * jnp.cos(q), v * jnp.sin(q), 0.])

    def control_jacobian(self, state, time):
        v = state[2]
        return jnp.array([
            # [0., 0.],
            [0., 0.],
            [1., 0.],
            [0., v],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [1., 0.],
            [0., 1.],
            [0., 0.],
            # [0., 0.],
        ])

## Settings
dynamics = VanderPol()
# grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(lo=np.array([-6., -10., -np.pi]),
#                                                                            hi=np.array([20., 10., np.pi])),
#                                                                (51, 40, 50),
#                                                                periodic_dims=2)
# values = jnp.linalg.norm(grid.states[..., :2], axis=-1) - 1

# solver_settings = hj.SolverSettings.with_accuracy("low")

grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array([-6., -10., -np.pi]),
                                                                           np.array([20., 10., np.pi])),
                                                               (51, 40, 50),
                                                               periodic_dims=2)
values = jnp.linalg.norm(grid.states[..., :2], axis=-1) - 1

solver_settings = hj.SolverSettings.with_accuracy("very_high",
                                                  hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)


## Target Values
time = 0.
target_time = -2.0
target_values = hj.step(solver_settings, dynamics, grid, time, values, target_time)


## Plot
print(target_values[:, :, 30].T.shape) # --> can't be represented in 2d?

plt.jet()
plt.figure(figsize=(13, 8))
plt.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], target_values[:, :, 30].T)
plt.colorbar()
plt.contour(grid.coordinate_vectors[0],
            grid.coordinate_vectors[1],
            target_values[:, :, 30].T,
            levels=0,
            colors="black",
            linewidths=3)


## All Values
times = np.linspace(0, -2.8, 57)
# values = initial_values
print(values.shape)
all_values = hj.solve(solver_settings, dynamics, grid, times, values)
print(all_values.shape)

## GIF Plot
vmin, vmax = all_values.min(), all_values.max()
levels = np.linspace(round(vmin), round(vmax), round(vmax) - round(vmin) + 1)
fig = plt.figure(figsize=(13, 8))


def render_frame(i, colorbar=False):
    plt.contourf(grid.coordinate_vectors[0],
                 grid.coordinate_vectors[1],
                 all_values[i, :, :, 30].T,
                 vmin=vmin,
                 vmax=vmax,
                 levels=levels)
    if colorbar:
        plt.colorbar()
    plt.contour(grid.coordinate_vectors[0],
                grid.coordinate_vectors[1],
                target_values[:, :, 30].T,
                levels=0,
                colors="black",
                linewidths=3)
    


render_frame(0, True)
animation = HTML(anim.FuncAnimation(fig, render_frame, all_values.shape[0], interval=50).to_html5_video())
plt.close(); animation