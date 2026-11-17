import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import numpy.typing as npt
import matplotlib.figure as ft

## =========================== Class for basic functions =========================
## Class writen to provide basic functionality for saving plots to a specified directory,
## written for MOR homeworks reused for this project.

class BaseGraphingFunctions:

    sub_dir: str | None

    def __init__(self, sub_dir: str | None = None) -> None:
        '''
        Graphing Tools, to offer graphing options.

        Parameters
        ---
        `sub_dir`: optional subdirectory to save plots at.
        '''
        self.sub_dir = sub_dir
        pass

    def _saveFigure(self, fig: ft.Figure, name: str, ext: str = "png")->None:
        '''
        Save a figure to the current directory.

        Parameters
        ---
        `fig`: a plt figure to save as a png file.

        `name`: image's name.

        `ext`: image extension.
        '''
        filename = f"{name.replace(' ', '')}.{ext}"

        if self.sub_dir != None:
            base_path = Path(self.sub_dir)
            base_path.mkdir(parents=True, exist_ok=True)  # ensure directory exists
            save_path = base_path / filename
        else:
            save_path = Path(filename)

        fig.savefig(save_path, format=ext)  # type: ignore
        return

## =========================== Class for validating the 1D wave equation model =========================
## Writen for this project to simplify graphing for different solutions of the equation.

class Validate1D(BaseGraphingFunctions):

    def __init__(self,
        U_num : npt.NDArray, 
        U_ext: npt.NDArray,
        space_grid: npt.NDArray,
        time_grid: npt.NDArray,
        sub_dir: str | None = None
    ) -> None:
        '''
        Class providing graphic visualizations and numerical tests to validate 
        a numerical PDE solution using an exact one.
        
        Parameters
        ---
        `U_num`: The matrix of numerical solutions of the PDE.

        `U_ext`: The matrix of exact solutions of the PDE.
        
        `space_grid`: The discretization in space used for both solutions.
        
        `time_grid`: The discretization in time used for both solutions.
        
        `sub_dir`: optional subdirectory to save plots at.
        '''
        assert U_num.shape == U_ext.shape, "Matrices cannot be compared!"
        super().__init__(sub_dir)
        self.U_num = U_num
        self.U_ext = U_ext
        self.space_grid = space_grid
        self.time_grid = time_grid
        pass

    def plot_solution(self,
        exact_sol: bool = True
        )-> None:
        '''
        Function to plot a space-time surface plot of the specified solution, at the specified directory `sub_dir`.
        
        Parameters
        ---
        `exact_sol`: if true the exact solution is plotted otherwise the numerical one, defaults to True.
        '''
        U = self.U_ext if exact_sol else self.U_num
        X, T = np.meshgrid(self.space_grid, self.time_grid, indexing="ij")

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(X, T, U, cmap="viridis")

        ax.set_xlabel("Space (x)")
        ax.set_ylabel("Time (t)")
        ax.set_zlabel("Displacement u(t, x)")
        ax.set_title("Wave Equation Solution")
        plt.tight_layout()

        self._saveFigure(fig, f"{'exact' if exact_sol else 'numerical'}_solution")
        return

    def plot_solution_comparison(self) -> None:
        '''Plot 3 heatmaps one for each solution exact, numerical, and the error between these. At the specified `sub_dir`.'''
        error = np.abs(self.U_ext - self.U_num)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        X, T = np.meshgrid(self.time_grid, self.space_grid)

        plots = [
            (self.U_ext, "Exact Solution"),
            (self.U_num, "Numerical Solution"),
            (error, "Absolute Error"),
        ]

        for ax, (data, title) in zip(axes, plots):
            im = ax.pcolormesh(
                X, T, data,
                shading="auto",
                cmap="viridis"
            )
            fig.colorbar(im, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Time (t)")
            ax.set_ylabel("Space (x)")

        plt.tight_layout()
        self._saveFigure(fig, 'exact_vs_numerical')
        return

    def calculate_error_metrics(self,
            returns=False
        ) -> dict | None:
        '''
        Function that calculates the maximum absolute error (MAE) and L^2 error, prints them to console or returns these.

        Parameters
        ---
        `returns`: weather to return the solutions or print them to console. Defaults to False
        '''
        absolute_error = np.abs(self.U_ext - self.U_num)

        max_error = np.max(absolute_error)

        l2_error = (
            np.linalg.norm(self.U_ext - self.U_num)
            / np.linalg.norm(self.U_ext)
        )

        if returns:
            return {
                "mae": max_error,
                "l2": l2_error,
            }
        print(f"--- Error Metrics Computed ---")
        print(f"Maximum Absolute Error:     {max_error:.5e}")
        print(f"L2 Error: {l2_error:.5e}")