# 2D-Nonlinear-Optimization-in-MATLAB-Steepest-Descent-Newton-and-Levenberg-Marquardt-
MATLAB implementation and comparison of 2D unconstrained optimization methods on the nonlinear test function   including Steepest Descent, Newton’s method, and Levenberg–Marquardt. Each method is tested with constant step size, exact line search , and Armijo backtracking, with multiple initial conditions.

# 2D Nonlinear Optimization in MATLAB
### Steepest Descent • Newton • Levenberg–Marquardt (Constant Step, Exact Line Search, Armijo)

This repository contains MATLAB scripts for studying and comparing classic **unconstrained optimization algorithms** in 2D on the nonlinear objective:

\[
f(x,y)=x^3 e^{-x^2-y^4}
\]

The project focuses on:
- convergence behavior from different initial points
- effect of step-size strategies (constant vs line-search)
- visual verification via contour plots + optimization trajectories

---

## Objective Function
- Function: \( f(x,y)=x^3 \exp(-x^2-y^4) \)
- Visualizations: surface + contour plots
- Known stationary points include the origin, and a local minimum along \(y=0\) (negative value around ~ -0.41, depending on the exact point used in the report/scripts).

---

## Implemented Methods

### Thema B — Steepest Descent
Steepest descent using the negative gradient direction.
Step-size variants:
- **(a) Constant step** (\(\gamma\) fixed)
- **(b) Exact line search** using `fminbnd`
- **(c) Armijo backtracking**

Initial conditions (as used in the scripts):
- (i) \(x_0=(0,0)\)
- (ii) \(x_0=(-1,-1)\)
- (iii) \(x_0=(1,1)\)

### Thema C — Newton’s Method (2D)
Newton direction computed via gradient + Hessian, with safeguards when the Hessian is ill-conditioned / non-invertible.
Step-size variants:
- **(a) Constant step**
- **(b) Exact line search** with `fminbnd`
- **(c) Armijo backtracking** (and descent fallback if needed)

### Thema D — Levenberg–Marquardt
Levenberg–Marquardt / damped Newton approach using:
\[
d_k = -(H(x_k)+\mu I)^{-1}\nabla f(x_k)
\]
with:
- **(a) Constant step**
- **(b) Exact line search** (where provided)
- **(c) Armijo backtracking** (where provided)

---

## Files & Structure

### Function plotting
- `f_xy.m` — 3D surface + contour of \(f(x,y)\)

### Steepest Descent (Thema B)
- `themaB_a_i.m`, `themaB_a_ii.m`, `themaB_a_iii.m` — constant step
- `themaB_a_ii_const.m`, `themaB_a_iii_const.m` — constant step + contour trajectory
- `themaB_b_ii.m`, `themaB_b_iii.m` — exact line search (`fminbnd`)
- `themaB_b_ii_nonconst.m`, `themaB_b_iii_nonconst.m` — line search + contour trajectory
- `themaB_c_i.m`, `themaB_c_ii.m`, `themaB_c_iii.m` — Armijo backtracking
- `themaB_c_armijo_iii.m` — Armijo variant + trajectory

### Newton (Thema C)
- `themaC_a_i.m`, `themaC_a_ii.m`, `themaC_a_iii.m` — constant step
- `themaC_a_ii_const.m`, `themaC_a_iii_const.m` — constant step + contour trajectory
- `themaC_b_ii.m`, `themaC_b_iii.m` — exact line search (`fminbnd`)
- `themaC_b_ii_nonconst.m`, `themaC_b_iii_nonconst.m` — line search + contour trajectory
- `themaC_c_ii.m`, `themaC_c_iii.m` — Armijo backtracking (+ convergence plots)
- `themaC_c_ii_armijo.m`, `themaC_c_iii_armijo.m` — Armijo + contour trajectory

### Levenberg–Marquardt (Thema D)
- `themaD_a_i.m`, `themaD_a_ii.m`, `themaD_a_iii.m` — constant step (damped Hessian)
- `themaD_a_ii_const.m`, `themaD_a_iii_const.m` — constant step + contour trajectory
- `themaD_b_ii.m`, `themaD_b_ii_nonconst.m` — exact line search (where provided)
- `themaD_c_ii.m`, `themaD_c_ii_armijo.m` — Armijo backtracking (where provided)

### Report
- `report.pdf` — full theory, derivations, and results

---

## How to Run
1. Open MATLAB and set this repository folder as the current directory.
2. Run the scripts you want, e.g.:
   - `f_xy` (function visualization)
   - `themaB_a_ii` (steepest descent, constant step, x0=(-1,-1))
   - `themaC_c_ii` (Newton + Armijo, x0=(-1,-1))
   - `themaD_a_iii` (Levenberg–Marquardt, constant step, x0=(1,1))
3. Each script produces figures such as:
   - convergence plots \(f(x_k)\) vs iteration
   - contour plots with the optimization path \((x_k,y_k)\)

---

## Notes
- Scripts with suffix `_const` typically include **contour + trajectory** visualization.
- Scripts with suffix `_nonconst` usually correspond to **non-constant step selection** (line search) and/or a different implementation style.

---

## Author
George Repanis (AUTH) — Course project submission
