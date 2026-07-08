@doc raw"""
    p_harmonic(mg; p, f, g_u, s_init) -> MGBProblem

The p-energy for vector-valued maps (the "vectorial p-Laplacian"),
```math
\min \int |\nabla u|_F^p + f \cdot u \, dx
\qquad u : \Omega \to \mathbb{R}^d,
```
where $|\nabla u|_F^2 = \sum_{i,j} (\partial u_i / \partial x_j)^2$ is the
Frobenius norm of the gradient matrix. Minimisers are *p-harmonic maps*
(in the flat-target case). Note: this is **not** elasticity — Frobenius of
the full gradient penalises rigid rotations. For Norton–Hoff power-law
elasticity, see [`norton_hoff`](@ref).

State $z = (u_1, \ldots, u_d, s)$ with slack $s \geq |\nabla u|_F^p$.

# Keyword arguments
- `p::Real = 1.5`: power-law exponent. Default 1.5 is in the genuinely
  non-quadratic regime (`p = 2` collapses to a decoupled system of scalar
  Laplacians and is the least interesting case).
- `f::Function = x -> ntuple(_->T(0.5), d)`: vector linear forcing.
- `g_u::Function`: Dirichlet boundary lift. The dim-aware default is a
  saddle-shaped trace on the first component, zero on the others
  (`(x_1^2,)` in 1D, `(x_1 x_2, 0)` in 2D, `(x_1 x_2 x_3, 0, 0)` in 3D),
  which is non-affine and drives the interior away from zero without
  needing strong forcing.
- `s_init::Real = 100`: feasible initial slack.
"""
function p_harmonic(mg::MultiGrid{T};
        p::Real = T(1.5),
        f::Function = x -> ntuple(_ -> T(0.5), _dim(mg)),
        g_u::Function = (d = _dim(mg);
            d == 1 ? (x -> (T(x[1]^2),)) :
                     (x -> ntuple(i -> i == 1 ? T(prod(x)) : T(0), d))),
        s_init::Real = T(100)) where {T}
    d = _dim(mg)
    p_val = T(p)
    S = _vector_state_setup(T, d, f, g_u, s_init)

    Q = convex_Euclidian_power(T; mg=mg, idx=S.idx, p=x->p_val)

    return assemble(mg; state_variables=S.state_variables, D=S.D, f=S.f_kw, g=S.g_kw, Q)
end
