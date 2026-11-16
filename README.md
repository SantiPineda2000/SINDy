# Application and analysis of SINDy for wave propagation in damaged material

This repository is intended to track the project of analyzing theoretical wave propagation through damaged material. The wave equation to be modeled is given by

$$
    u_{tt}(t, x) - c^2(x) \Delta u(t, x) = f(t, x) \quad \text{in } (0, T) \times \Omega,
$$

with a space-dependent wave speed $c^2(\mathbf{x}) > 0$. In spatial coordinates $\Omega \subset \mathbb{R}^d$ and time $t \in [0, T]$, equiped with homogeneous Dirichlet boundary conditions and initial conditions,

$$
u(0, x) = u_{0}(x), \quad u_{t}(0, x) = u_{1}(x) \quad \text{in } \Omega
$$

The scope of the project is the following

1. Solve the wave equation numerically in $d=1, 2$, choosing suitable $f, u_0, u_1$ and meaningful $c(x)$.
2. Introduce one defect in the model by a local reduction of the wave speed $c$.
3. Apply SINDy and variants of it to discover the wave problem from data, and compare these results to the full simulation of the system.
4. Apply modifications to the PDE and application of PDE-FIND.
5. Estimate the local defect position, i.e. $c(x)$ using SINDy and variants.
6. **_Optional_:** _Generate a data set with different defects to train a ML model that detect the position of these defects._
