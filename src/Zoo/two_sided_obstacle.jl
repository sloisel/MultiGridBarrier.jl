@doc raw"""
    two_sided_obstacle(mg; f, g_u, ψ_lower, ψ_upper, s_init) -> MGBProblem

Membrane between an upper and a lower obstacle:
```math
\min \int \tfrac{1}{2}|\nabla u|^2 + f u \, dx
\quad \text{s.t.}\quad \psi_{\mathrm{lower}}(x) \leq u(x) \leq \psi_{\mathrm{upper}}(x).
```
Encoded with state $z = (u, s)$, slack $s \geq |\nabla u|^2$, and two linear
barriers $u - \psi_{\mathrm{lower}} > 0$, $\psi_{\mathrm{upper}} - u > 0$.

# Keyword arguments
- `f::Function`: scalar linear forcing. The dim-dependent default
  (`f = 1` in 1D, `2` in 2D, `8` in 3D) drives the elastic solution past
  the default lower obstacle `ψ_lower = -0.1`, so the active set covers
  $\approx 25$–$75\%$ of $\Omega$.
- `g_u::Function = x -> 0`: Dirichlet lift; must lie strictly inside the obstacles.
- `ψ_lower::Function = x -> -0.1`, `ψ_upper::Function = x -> 1`: obstacles.
  Default values place the lower obstacle close enough to zero that the
  default forcing reaches it; the upper obstacle is loose by default.
- `s_init::Real = 10`: feasible initial slack.
"""
function two_sided_obstacle(mg::MultiGrid{T};
        f::Function = (d = _dim(mg); x -> T(d == 1 ? 1 : d == 2 ? 2 : 8)),
        g_u::Function = x -> T(0),
        ψ_lower::Function = x -> T(-0.1),
        ψ_upper::Function = x -> T(1),
        s_init::Real = T(10)) where {T}
    d = _dim(mg)
    state_variables = [:u :dirichlet
                       :s :full]
    D = default_D(d)
    nrows = d + 2

    f_kw, g_kw = _scalar_fg(T, nrows, f, g_u, s_init)

    Q_slack = convex_Euclidian_power(T; mg=mg, idx=default_idx(d), p=x->T(2))

    # Two-sided box on u (y-position 1): u − ψ_lower > 0 AND ψ_upper − u > 0.
    # idx = SVector(1), A is 2×1 (column-major), b is a 2-vector.
    A_box = x -> SMatrix{2,1,T}(one(T), -one(T))
    b_box = let ψl = ψ_lower, ψu = ψ_upper
        x -> SVector{2,T}(-T(ψl(x)), T(ψu(x)))
    end
    Q_box = convex_linear(T; mg=mg, idx=SVector{1,Int}(1), A=A_box, b=b_box)
    Q = intersect(mg, Q_slack, Q_box)

    return assemble(mg; state_variables, D, f=f_kw, g=g_kw, Q)
end
