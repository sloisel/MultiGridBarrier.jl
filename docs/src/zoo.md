```@meta
CurrentModule = MultiGridBarrier
```

# The Zoo: a menu of convex variational problems

`MultiGridBarrier.Zoo` packages six classical convex variational problems
as one-line constructors that return an [`MGBProblem`](@ref). The typical usage is

```julia
mg      = amg(fem2d_P1())
problem = Zoo.minimal_surface(mg)
sol     = mgb_solve(problem)
plot(sol)
```

Each problem ships with dimension-agnostic defaults; the same constructor
works for a 1D, 2D, or 3D mesh. Below: one solve per problem at each
dimension, on the coarsest mesh that still draws an interesting picture.
Tolerances are loose (`tol = 1e-3`) and progress bars are suppressed.

```@example zoo
using MultiGridBarrier, PyPlot   # PyPlot enables the plotting extension
nodes1 = collect(range(-1.0, 1.0, length=17))     # 1D
mg1 = amg(fem1d(; nodes=nodes1))
mg2 = amg(subdivide(fem2d_P1(), 3))               # 2D
mg3 = amg(subdivide(fem3d(; k=1), 3))             # 3D — L=3 to resolve active sets
nothing  # hide
```

## 1. Elasto-plastic torsion (Hencky)

```math
\min \int \tfrac{1}{2}|\nabla u|^2 + f u \, dx
\quad \text{s.t.}\quad |\nabla u| \leq \texttt{smax}.
```
A twisted prismatic bar in the elastic-perfectly-plastic regime; the
constraint is the von Mises yield bound on the stress potential.

### 1D
```@example zoo
sol = mgb_solve(Zoo.elastoplastic_torsion(mg1); verbose=false, tol=1e-3)
plot(sol); savefig("zoo_torsion_1d.svg"); nothing  # hide
close()  # hide
```
![](zoo_torsion_1d.svg)

### 2D
```@example zoo
sol = mgb_solve(Zoo.elastoplastic_torsion(mg2); verbose=false, tol=1e-3)
plot(sol); savefig("zoo_torsion_2d.svg"); nothing  # hide
close()  # hide
```
![](zoo_torsion_2d.svg)

### 3D
```@example zoo
sol = mgb_solve(Zoo.elastoplastic_torsion(mg3); verbose=false, tol=1e-3)
fig = plot(sol); savefig(fig, "zoo_torsion_3d.png"); nothing  # hide
```
![](zoo_torsion_3d.png)

## 2. Minimal surface (graph form)

```math
\min \int \sqrt{1 + |\nabla u|^2}\, dx
```
with prescribed Dirichlet boundary trace. Plateau / Bernstein problem. The
dim-aware default `g_u` gives a non-trivial trace so the picture isn't
$u \equiv 0$.

### 1D
```@example zoo
sol = mgb_solve(Zoo.minimal_surface(mg1); verbose=false, tol=1e-3)
plot(sol); savefig("zoo_minsurf_1d.svg"); nothing  # hide
close()  # hide
```
![](zoo_minsurf_1d.svg)

### 2D
```@example zoo
sol = mgb_solve(Zoo.minimal_surface(mg2); verbose=false, tol=1e-3)
plot(sol); savefig("zoo_minsurf_2d.svg"); nothing  # hide
close()  # hide
```
![](zoo_minsurf_2d.svg)

### 3D
```@example zoo
sol = mgb_solve(Zoo.minimal_surface(mg3); verbose=false, tol=1e-3)
fig = plot(sol); savefig(fig, "zoo_minsurf_3d.png"); nothing  # hide
```
![](zoo_minsurf_3d.png)

## 3. p-harmonic maps (vectorial p-Laplacian)

```math
\min \int |\nabla u|_F^p + f \cdot u \, dx,\qquad u:\Omega \to \mathbb{R}^d.
```
The full-gradient (Frobenius) p-energy. Plots show $\|u(x)\|$, a scalar
field.

### 1D
In 1D, $u$ is scalar and this reduces to standard scalar $p$-Laplace.
```@example zoo
sol = mgb_solve(Zoo.p_harmonic(mg1); verbose=false, tol=1e-3)
plot(sol); savefig("zoo_pharm_1d.svg"); nothing  # hide
close()  # hide
```
![](zoo_pharm_1d.svg)

### 2D
```@example zoo
sol = mgb_solve(Zoo.p_harmonic(mg2); verbose=false, tol=1e-3)
u_norm = sqrt.(sol.z[:,1].^2 .+ sol.z[:,2].^2)
plot(sol.geometry, u_norm); savefig("zoo_pharm_2d.svg"); nothing  # hide
close()  # hide
```
![](zoo_pharm_2d.svg)

### 3D
```@example zoo
sol = mgb_solve(Zoo.p_harmonic(mg3); verbose=false, tol=1e-3)
u_norm = sqrt.(sol.z[:,1].^2 .+ sol.z[:,2].^2 .+ sol.z[:,3].^2)
fig = plot(sol.geometry, u_norm); savefig(fig, "zoo_pharm_3d.png"); nothing  # hide
```
![](zoo_pharm_3d.png)

## 4. Norton–Hoff power-law elasticity

```math
\min \int |\varepsilon(u)|_F^p + f \cdot u \, dx,
\qquad \varepsilon(u) = \tfrac{1}{2}(\nabla u + \nabla u^T).
```
Symmetric-gradient power-law (the physically correct model: linear
elasticity at $p=2$, Norton–Hoff creep for $1 \leq p < 2$). Plots show
$\|u(x)\|$.

### 1D
Not defined in 1D — symmetric gradient reduces to scalar gradient. Use
scalar $p$-Poisson or [`Zoo.elastoplastic_torsion`](@ref) instead.

### 2D
```@example zoo
sol = mgb_solve(Zoo.norton_hoff(mg2); verbose=false, tol=1e-3)
u_norm = sqrt.(sol.z[:,1].^2 .+ sol.z[:,2].^2)
plot(sol.geometry, u_norm); savefig("zoo_norton_2d.svg"); nothing  # hide
close()  # hide
```
![](zoo_norton_2d.svg)

### 3D
```@example zoo
sol = mgb_solve(Zoo.norton_hoff(mg3); verbose=false, tol=1e-3)
u_norm = sqrt.(sol.z[:,1].^2 .+ sol.z[:,2].^2 .+ sol.z[:,3].^2)
fig = plot(sol.geometry, u_norm); savefig(fig, "zoo_norton_3d.png"); nothing  # hide
```
![](zoo_norton_3d.png)

## 5. ROF total-variation denoising

```math
\min_u \int |\nabla u| + \tfrac{\lambda}{2}(u - f_{\mathrm{data}})^2 \, dx.
```
Rudin–Osher–Fatemi 1992 — the foundational variational model for
edge-preserving image denoising. We feed a step-like `f_data` so the
denoised `u` is non-trivial.

### 1D
```@example zoo
sol = mgb_solve(Zoo.rof(mg1); verbose=false, tol=1e-3)
plot(sol); savefig("zoo_rof_1d.svg"); nothing  # hide
close()  # hide
```
![](zoo_rof_1d.svg)

### 2D
```@example zoo
sol = mgb_solve(Zoo.rof(mg2); verbose=false, tol=1e-3)
plot(sol); savefig("zoo_rof_2d.svg"); nothing  # hide
close()  # hide
```
![](zoo_rof_2d.svg)

### 3D
```@example zoo
sol = mgb_solve(Zoo.rof(mg3); verbose=false, tol=1e-3)
fig = plot(sol); savefig(fig, "zoo_rof_3d.png"); nothing  # hide
```
![](zoo_rof_3d.png)

## 6. Two-sided obstacle

```math
\min \int \tfrac{1}{2}|\nabla u|^2 + f u \, dx
\quad \text{s.t.}\quad \psi_{\mathrm{lower}}(x) \leq u(x) \leq \psi_{\mathrm{upper}}(x).
```
Membrane pinched between an upper and a lower obstacle.

### 1D
```@example zoo
sol = mgb_solve(Zoo.two_sided_obstacle(mg1); verbose=false, tol=1e-3)
plot(sol); savefig("zoo_obstacle_1d.svg"); nothing  # hide
close()  # hide
```
![](zoo_obstacle_1d.svg)

### 2D
```@example zoo
sol = mgb_solve(Zoo.two_sided_obstacle(mg2); verbose=false, tol=1e-3)
plot(sol); savefig("zoo_obstacle_2d.svg"); nothing  # hide
close()  # hide
```
![](zoo_obstacle_2d.svg)

### 3D
```@example zoo
sol = mgb_solve(Zoo.two_sided_obstacle(mg3); verbose=false, tol=1e-3)
fig = plot(sol); savefig(fig, "zoo_obstacle_3d.png"); nothing  # hide
```
![](zoo_obstacle_3d.png)

# Zoo module reference

```@autodocs
Modules = [MultiGridBarrier.Zoo]
Order   = [:module, :function]
Private = false
```
