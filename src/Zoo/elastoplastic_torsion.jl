@doc raw"""
    elastoplastic_torsion(mg; f, g_u, smax, s_init) -> MGBProblem

Hencky elasto-plastic torsion of a prismatic bar. State $z = (u, s)$, problem
```math
\min \int \tfrac{1}{2}|\nabla u|^2 + f u \, dx
\quad \text{s.t.}\quad |\nabla u| \leq \texttt{smax}\;\;\text{a.e.}
```
encoded as $s \geq |\nabla u|^2$ (slack inequality) and $s \leq \texttt{smax}^2$
(linear constraint). At the optimum $s = |\nabla u|^2$, so $\tfrac{1}{2}\int s$
in the objective recovers the elastic energy $\tfrac{1}{2}\int|\nabla u|^2$.

Returns an `MGBProblem`; solve with `mgb_solve(problem; kwargs...)`.

# Keyword arguments
- `f::Function`: scalar linear forcing. The dim-dependent default
  (`f = 2` in 1D, `4` in 2D, `16` in 3D) is chosen so that the elastic
  solution's $\max|\nabla u|$ overshoots `smax` by roughly $2\times$, which
  places the plastic free boundary well inside the domain — the active
  set ($|\nabla u| = s_{\max}$) covers $\approx 25$–$75\%$ of $\Omega$,
  showing the elastic / plastic transition clearly.
- `g_u::Function = x -> 0`: scalar Dirichlet boundary lift for `u`.
- `smax::Real = 1`: yield bound on `|∇u|`.
- `s_init::Real = smax^2/2`: feasible initial slack (must lie in `(0, smax^2)`).
"""
function elastoplastic_torsion(mg::MultiGrid{T};
        f::Function = (d = _dim(mg); x -> T(d == 1 ? 2 : d == 2 ? 4 : 16)),
        g_u::Function = x -> T(0),
        smax::Real = one(T),
        s_init::Real = T(smax)^2 / 2) where {T}
    d = _dim(mg)
    state_variables = [:u :dirichlet
                       :s :full]
    D = default_D(d)
    nrows = d + 2                       # 1 (u:id) + d (partials) + 1 (s:id)
    smax2 = T(smax)^2

    f_kw, g_kw = _scalar_fg(T, nrows, f, g_u, s_init)

    # s ≥ |∇u|²  (Euclidean-power barrier; p=2 here means α = 2/p = 1).
    Q_slack = convex_Euclidian_power(T; mg=mg, idx=default_idx(d), p=x->T(2))
    # s ≤ smax²  (linear barrier: -s + smax² > 0).
    A_yield = x -> SMatrix{1,1,T}(-one(T))
    b_yield = x -> SVector{1,T}(smax2)
    Q_yield = convex_linear(T; mg=mg, idx=SVector{1,Int}(nrows), A=A_yield, b=b_yield)
    Q = intersect(mg, Q_slack, Q_yield)

    return assemble(mg; state_variables, D, f=f_kw, g=g_kw, Q)
end
