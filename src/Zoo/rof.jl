@doc raw"""
    rof(mg; f_data, λ, g_u, s_init, r_init) -> NamedTuple

Rudin–Osher–Fatemi total-variation denoising of a scalar field `f_data`:
```math
\min_u \int |\nabla u| + \tfrac{\lambda}{2} (u - f_{\mathrm{data}})^2 \, dx,
```
encoded with state $z = (u, s, r)$ where $s \geq |\nabla u|$ is the TV slack
and $r \geq (u - f_{\mathrm{data}})^2$ is the data-fitting slack. The linear
objective becomes $\int (s + \tfrac{\lambda}{2} r) \, dx$.

# Keyword arguments
- `f_data::Function`: scalar noisy input. Default is a smoothed-step
  signal $0.5 \tanh(5 x_1)$ along the first coordinate.
- `λ::Real = 1`: fidelity weight.
- `g_u::Function`: Dirichlet boundary lift for `u`. Defaults to `f_data`
  so the boundary trace matches the data — otherwise the zero Dirichlet
  BC fights the data near $\partial \Omega$ and drags `u` to zero.
- `s_init::Real = 10`, `r_init::Real = 10`: feasible initial slacks.
"""
function rof(mg::MultiGrid{T};
        f_data::Function = x -> T(0.5) * tanh(T(5) * x[1]),
        λ::Real = T(1),
        g_u::Function = f_data,
        s_init::Real = T(10),
        r_init::Real = T(10)) where {T}
    d = _dim(mg)
    λ_val = T(λ)
    state_variables = [:u :dirichlet
                       :s :full
                       :r :full]
    # D rows: u:id, u:partials..., s:id, r:id. nrows = d + 3.
    rows = Any[[:u :id]]
    op_syms = (:dx, :dy, :dz)
    for j in 1:d
        push!(rows, [:u op_syms[j]])
    end
    push!(rows, [:s :id])
    push!(rows, [:r :id])
    D = vcat(rows...)
    nrows = d + 3

    f_kw = x -> SVector{nrows,T}(ntuple(k -> k == nrows - 1 ? one(T) :
                                          k == nrows ? λ_val / 2 : zero(T), Val(nrows)))
    g_kw = let gu = g_u
        x -> SVector{3,T}(T(gu(x)), T(s_init), T(r_init))
    end

    # TV slack: s ≥ |∇u| (Euclidean-power barrier; p=1 ⇒ α=2, barrier −log(s²−|q|²)).
    tv_idx = SVector{d + 1,Int}(ntuple(k -> k <= d ? 1 + k : nrows - 1, Val(d + 1)))
    Q_tv = convex_Euclidian_power(T; mg=mg, idx=tv_idx, p=x->T(1))

    # Data slack: r ≥ (u − f_data)². idx = (u, r), A = I, b = (−f_data, 0).
    data_idx = SVector{2,Int}(1, nrows)
    A_data = x -> SMatrix{2,2,T}(one(T), zero(T), zero(T), one(T))
    b_data = let fd = f_data
        x -> SVector{2,T}(-T(fd(x)), zero(T))
    end
    Q_data = convex_Euclidian_power(T; mg=mg, idx=data_idx,
                A=A_data, b=b_data, p=x->T(2))

    Q = intersect(mg, Q_tv, Q_data)

    return (; mg, state_variables, D, f=f_kw, g=g_kw, Q)
end
