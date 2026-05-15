@doc raw"""
    p_harmonic(mg; p, f, g_u, s_init) -> NamedTuple

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

    # State: d copies of u (each :dirichlet) and one slack :full.
    state_variables = vcat([[Symbol("u$i") :dirichlet] for i in 1:d]..., [:s :full])
    # D rows: for each u_i, an :id row plus d partials; then s:id at the end.
    rows = Vector{Any}()
    op_syms = (:dx, :dy, :dz)
    for i in 1:d
        push!(rows, [Symbol("u$i") :id])
        for j in 1:d
            push!(rows, [Symbol("u$i") op_syms[j]])
        end
    end
    push!(rows, [:s :id])
    D = vcat(rows...)
    nrows = size(D, 1)                  # = d*(1 + d) + 1

    # Linear functional: f_i on each u_i:id row; 1 on slack; 0 on partials.
    f_kw = let f0 = f, d = d
        x -> begin
            fv = f0(x)
            SVector{nrows,T}(ntuple(k -> begin
                if k == nrows
                    one(T)
                else
                    pos = k - 1
                    i = pos ÷ (d + 1) + 1
                    off = pos - (i - 1) * (d + 1)
                    (1 <= i <= d && off == 0) ? T(fv[i]) : zero(T)
                end
            end, Val(nrows)))
        end
    end
    g_kw = let gu = g_u, d = d
        x -> begin
            gv = gu(x)
            SVector{d + 1,T}(ntuple(k -> k <= d ? T(gv[k]) : T(s_init), Val(d + 1)))
        end
    end

    # idx picks all partials (d² of them) plus the slack.
    nz = d * d + 1
    partial_positions = Int[]
    for i in 1:d, j in 1:d
        push!(partial_positions, (i - 1) * (d + 1) + 1 + j)
    end
    push!(partial_positions, nrows)
    idx = SVector{nz,Int}(partial_positions)

    Q = convex_Euclidian_power(T; mg=mg, idx=idx, p=x->p_val)

    return (; mg, state_variables, D, f=f_kw, g=g_kw, Q)
end
