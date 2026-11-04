# SINDy Summary

Since many system's dynamics $f$ often have few active terms in the right-hand side functions. SINDy attempts to approximate these dynamics by the generalized linear model

$$
f(x)\approx\sum_{k=1}^{p}{\theta_{k}(x)\xi} = \Theta(x)\xi
$$

Where $\xi>0$ are as few as possible, which allows solving for terms relevant to the dynamics using sparse regression.

## SINDy Steps

1. Collect time-series data into a matrix, and form a matrix of derivatives.
   $$
   X = [x(t_1), x(t_2), \dots, x(t_m)]^{\top}\\
   \dot{X}= [\dot{x}(t_1) \quad \dot{x}(t_2) \quad \dots \quad \dot{x}(t_m)]^{\top}
   $$
   **_Note_**: _for noisy data use the total variational regularized derivative, or avoid derivatives using discrete-time systems._
2. Construct a library of candidate non-linear functions $\Theta(X)$ from $X$.
   $$
   \Theta(X)=[1 \quad X \quad X^2 \quad \dots \quad X^d \quad \dots \quad \sin{(X)} \quad \dots]
   $$
   Where $X\in \mathbb{R}^d$ is a matrix of vectors containing all possible time series polynomials in state $x$.
3. Represent the dynamical system.
   $$
   \dot{X}=\Theta(X)\Xi
   $$
   Where each $\xi_k$ in $\Xi$ is a vector of coefficients determining the active terms in the $\Theta(X)$ dynamics' $k$-th row.
4. Fit a parsimonious model, which accurately fits using as few terms in $\Xi$ as possible.
   $$
   \xi_k=\argmin_{\xi^{'}_{k}}||\dot{X}_{k}- \Theta(X)\xi^{'}_{k}||_{2} + \lambda ||\xi^{'}_{k}||_{1}
   $$
   Where $\dot{X}_{k}$ is the $k$-th column of $\dot{X}$, and $\lambda$ is a sparcity promoting knob.
5. Select active terms using **STLS** sequential thresholded least-squares, which improves the numerical robustness for noisy overdetermined problems.
6. Synthesize the sparse vectors $\xi_{k}$ into a dynamical system.
   $$
   \dot{x}_{k} = \Theta(x) \xi_{k}
   $$
   Where $\dot{x}_{k}$ is the $k$-th element of $\dot{x}$ and $\Theta(x)$ is a row vector of symbolic functions of $x$.

_**Note:**_ _Check the **Code 7.4** in page 276 for the implementation of the STLS in SINDy._

## Discovering Partial Differential Equations

The algorithm PDE-FIND (partial differential equation functional identification of non-linear dynamics), is an extended version of the **SINDy** framework, which successfully identifies several canonical PDEs from classical physics, purely from noisy data.

### PDE-FIND Steps

1. Arrange spatial time-series data, collected over $m$ time points and $n$ spatial locations, in a single column vector.
   $$
   Y \in \mathbb{C}^{mn}
   $$
2. Arrange additional imputs (eg. known potential, magnitude of complex data, etc.) into a column vector.
   $$
   Q \in \mathbb{C}^{mn}
   $$
3. Construct a libary of $D$ candidate linear and nonlinear terms $\Theta(Y, Q) \in \mathbb{C}^{mn}$. The derivatives are taken for **clean data** _using finite differences_, and for **noisy data** _using **polynomial interpolation**_.
   $$
   \Theta(Y, Q) \in \mathbb{C}^{mn} = [1 \quad Y \quad Y^2 \quad \dots \quad Q \quad \dots Y_x \quad YY_x \dots]
   $$
4. Compute and reshape the time derivative into a column vector $Y_{t}$.
5. Express the PDE evolution as,
   $$
   Y_{t}= \Theta(Y,Q)\xi
   $$
   Where $\xi$ is a sparse vector containing coefficients corresponding to a term on the PDE.
6. Apply ridge regression with hard thresholding, "sequential threshold ridge regression", for a given tolerance and threshold $\lambda$, which gives an sparse approximation to $\xi$.
   $$
   \hat{\xi} = \argmin_{\xi} ||\Theta(Y, Q)\xi-Y_{t}||_{2}^{2} + \epsilon \kappa(\Theta(Y, Q))||\xi||_{0}
   $$
   Where $\kappa(\Theta)$ is the condition number of matrix $\Theta$, providing stronger regularization for ill-conditioned problems. Penalizing $||\xi||_{0}$ discourages overfitting.

_**Note:**_ _check **Algorithm 1** on page 282, STRidge, for the recursive tolerance refinement._
