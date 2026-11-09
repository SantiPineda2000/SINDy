import numpy as np
import skfem as fem
from scipy.sparse import identity
from scipy.sparse.linalg import splu
from skfem.models import  mass

m = fem.MeshLine(np.linspace(0.0, 100.0, 25))
basis = fem.Basis(m, fem.ElementLineP1())
N = basis.N
def c_sqrd(x):
    return (1.0 + x)**2

@fem.BilinearForm
def weighted_laplace(u, v, w):
    # Extracts the physical spatial coordinates at numerical integration points
    x = w.x[0] 
    return c_sqrd(x) * u.grad[0] * v.grad[0]

# Assamble matrices
K = weighted_laplace.assemble(basis)
M = mass.assemble(basis)
I = identity(N)

time_holder = {'t': 0.0} # Mutable container for time, to access time steps

@fem.LinearForm
def source_term(v, w):
    x = w.x[0]
    t = time_holder['t']
    f_val = (np.pi**2 * c_sqrd(x) - 1.0) * np.sin(np.pi * x) * np.cos(t)
    return f_val * v

# First order reduction setup
A0 = fem.bmat([[I, None],
               [None, M]], "csc") # TODO: ASK WHAT "csc" DOES!
B0 = fem.bmat([[None, I],
               [-K, None]], "csc")

# Crank-Nicolson
dt = .005
theta = .5
A = A0 + theta * B0 * dt
B = A0 - (1. - theta) * B0 * dt
backsolve = splu(A).solve

# Initial conditions
def initial_u(x):
    return np.sin(np.pi * x[0])
U0 = basis.project(initial_u)
U = np.concatenate((U0, np.zeros(N)))

# timestepping
def evolve(t, u):
    while t < 1.0:
        time_holder['t'] = t

        F = source_term.assemble(basis)
        F_system = np.concatenate((np.zeros(N), F))

        rhs = rhs = (A0 - (1.0 - theta) * B0 * dt) @ u + dt * ((1.0 - theta) * F_system_old + theta * F_system) if 'F_system_old' in locals() else (A0 - (1.0 - theta) * B0 * dt) @ u + dt * F_system #type:ignore

        F_system_old = F_system

        t += dt
        time_holder["t"] = t

        F_new = source_term.assemble(basis)
        F_system_new = np.concatenate((np.zeros(N), F_new))

        rhs = (A0 - (1.0 - theta) * B0 * dt) @ u + dt * (1.0 - theta) * F_system + dt * theta * F_system_new
        F_system = F_system_new
        
        u = backsolve(rhs)
        yield t, u

if __name__ == "__main__":
    from pathlib import Path
    import matplotlib.pyplot as plt
    import matplotlib.animation as anim

    save_path = Path("./assets/1DWave")
    fig, ax = plt.subplots(figsize=(10, 4))
    ix = np.argsort(m.p[0])
    
    # Plot initial condition vs ground truth comparison setup
    line_fem, = ax.plot(m.p[0, ix], U0[ix], label='FEM Solution', color='blue')
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.2, 1.2)
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("u(t, x)")

    def update(t_u):
        t, u = t_u
        u_displacement, _ = np.split(u, [M.shape[0]]) #type:ignore
        line_fem.set_ydata(u_displacement[ix])
        ax.set_title(f"1D Wave Equation with c(x) = 1+x | Time t = {t:.3f}")
        return line_fem,

    # Run animation / simulation across t in (0, 1)
    animation = anim.FuncAnimation(fig, update, evolve(0.0, U), interval=30, blit=False)
    animation.save(save_path.with_suffix(".gif"), "imagemagick")