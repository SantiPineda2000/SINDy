import numpy as np
import numpy.typing as npt

from validate1d import Validate1D

def generate_wave_snapshots( 
    T: tuple[float, float],
    N: int,
    M: int,
    Omega: tuple[float, float] = (0.0, 1.0)
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    '''
    Generate snapshots of the 1D wave equation using
    finite differences.

    Parameters
    ---
    `T`: Time horizon (t_0, t_f).

    `N`: Number of space intervals.
    
    `M`: Number of time intervals.

    `Omega`: Spatial domain, defaults to (0.0, 1.0).

    Returns
    ---
    `U`: Numerical solution snapshots, shape (N, n_steps + 1).

    `spatial_grid`: Spatial discretization.

    `time_grid`: Temporal discretization.
    '''
    # Discretizing space and time
    t = np.linspace(T[0], T[1], M+1)
    x = np.linspace(Omega[0], Omega[1], N+1)
    # Calculating Delta t and x
    dt = t[1] - t[0]
    dx = x[1] - x[0]

    # Defining the equation components
    def c_sqrd_ok(x):
        return (1 + x) **2

    def c_sqrd_def(x):
        '''Introduces a reduction of speed in two points of the space grid'''
        mid = int(N * 0.5)
        c_sqrd = np.ones_like(x)
        c_sqrd[:mid-1] = (1+x[:mid-1])**2
        c_sqrd[mid-1:mid+1] = 0.3*(1+x[mid-1:mid+1])**2
        c_sqrd[mid+1:] = ((1+x[mid+1:])**2)
        return c_sqrd

    def f(t,x):
        '''Forcing term equation'''
        return (np.pi**2 * c_sqrd - 1.0) * np.sin(np.pi * x) * np.cos(t)

    # Defining initial conditions
    def u_0(x):
        '''Initial condition at time 0'''
        return np.sin(np.pi * x)

    # Checking for stability
    c_sqrd = c_sqrd_ok(x)
    cfl = np.max(np.abs(c_sqrd)) * dt / dx
    assert cfl < 1, f"Unstable time and space steps, with a cfl={cfl}"

    # Solution matrix
    U = np.zeros((N+1, M+1))

    # Enforcing initial and Dirichlet boundary conditions
    U[1:-1, 0] = u_0(x)[1:-1]

    c_sqrd_int = c_sqrd[1:-1]
    r2 = (dt / dx) ** 2

    # Using Taylor Expansion Initialization j=0
    lap_0 = U[2:, 0] - 2.0 * U[1:-1, 0] + U[:-2, 0]
    U[1:-1, 1] = (
        U[1:-1, 0]
        + 0.5 * (dt ** 2) * (c_sqrd_int * (lap_0 / (dx ** 2)) + f(t[0], x)[1:-1])
    )

    # Step 4: Leapfrog steps for j >= 1
    for j in range(1, M):
        lap_j = U[2:, j] - 2.0 * U[1:-1, j] + U[:-2, j]
        U[1:-1, j + 1] = (
            c_sqrd_int * r2 * lap_j
            + (dt ** 2) * f(t[j], x)[1:-1]
            - U[1:-1, j - 1]
            + 2.0 * U[1:-1, j]
        )
    return U, x, t

def generate_exact_solution(
        space_grid: npt.NDArray,
        time_grid: npt.NDArray,
    ) -> npt.NDArray:
    '''
    Generate the exact solution snapshot matrix of the PDE with homogeneous
    Dirichlet boundary and initial conditions.

    u(t, x) = sin(\\pi * x) cos(t)

    Parameters:
    ---
    `space_grid`: The discretization of space used to solve the system.

    `time_grid`: The discretization of time used to solve the system.

    Returns:
    ---
    `snapshot_matrix`: The exact solution discretized at the provided time and space grids.
    '''
    return np.sin(np.pi * space_grid[:, None]) * np.cos(time_grid[None,:])

if __name__ == "__main__":
    # Solve the equation numerically
    U_num, spatial_grid, time_grid = generate_wave_snapshots((0.0, 1.0), N=20, M=100)
    print(f"Snapshot matrix generated successfully!")
    print(f"Matrix shape (x, time): {U_num.shape}")
    print(f"Spatial grid range: [{spatial_grid[0]:.1f}, {spatial_grid[-1]:.1f}]")
    print(f"Time steps collected: {len(time_grid)}")

    # Generating exact solution from the manufactured equation
    U_ext = generate_exact_solution(spatial_grid, time_grid)

    # Validation of the numerical solution
    val = Validate1D(U_num, U_ext, spatial_grid, time_grid, 'assets')

    val.plot_solution() # Ploting the exact solution on a space-time surface plot.
    val.plot_solution(False) # Ploting the numerical solution on a space-time surface plot.
    val.plot_solution_comparison() # Ploting the comparison heatmaps.
    val.calculate_error_metrics() # Printing the error matrics MAE and l2 to console.