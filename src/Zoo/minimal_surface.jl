@doc raw"""
    minimal_surface(mg; g_u, s_init) -> MGBProblem

Minimal-surface (Plateau) problem in graph form. State $z = (u, s)$, problem
```math
\min \int \sqrt{1 + |\nabla u|^2}\, dx
```
with prescribed Dirichlet boundary trace `g_u`. The integrand is encoded as
$s \geq \sqrt{1 + |\nabla u|^2}$ via the Euclidean-power barrier with a
non-trivial affine `A, b`: the augmented vector $Ay + b$ packs
`(∇u, 1, 0, s)` so the slack inequality $s^2 \geq |\nabla u|^2 + 1$ is the
shifted Lorentz cone.

# Keyword arguments
- `g_u::Function`: Dirichlet boundary lift. The dim-aware default produces a
  non-trivial picture: $x \mapsto \tfrac{1}{2} x_1^2$ in 1D,
  $x \mapsto \tfrac{1}{2}(x_1^2 - x_2^2)$ (saddle) in 2D, and
  $x \mapsto \tfrac{1}{2}\|x\|^2$ in 3D.
- `s_init::Real = 10`: feasible initial slack (need $s^2 > |\nabla u_0|^2 + 1$).
"""
function minimal_surface(mg::MultiGrid{T};
        g_u::Function = (d = _dim(mg);
            d == 1 ? (x -> T(0.5) * x[1]^2) :
            d == 2 ? (x -> T(0.5) * (x[1]^2 - x[2]^2)) :
                     (x -> T(0.5) * sum(abs2, x))),
        s_init::Real = T(10)) where {T}
    d = _dim(mg)
    state_variables = [:u :dirichlet
                       :s :full]
    D = default_D(d)
    nrows = d + 2                       # 1 (u:id) + d (partials) + 1 (s:id)

    f_kw = x -> SVector{nrows,T}(ntuple(i -> i == nrows ? one(T) : zero(T), Val(nrows)))
    g_kw = let gu = g_u
        x -> SVector{2,T}(T(gu(x)), T(s_init))
    end

    # Build A : ℝ^nz → ℝ^nz and b ∈ ℝ^nz so that Ay+b = (∇u₁,…,∇u_d, 1, 0, s).
    # The trailing zero (at position d+2 ≤ k < nz) pads q to length nz-1; the
    # padding contributes 0 to |q|² so the constraint becomes s² ≥ |∇u|² + 1.
    nz = nrows
    A_ms = let nz = nz, d = d
        x -> SMatrix{nz,nz,T}(ntuple(k -> begin
            i = (k - 1) % nz + 1
            j = (k - 1) ÷ nz + 1
            # Partials are at y-positions 2..d+1 (D row k has :id at pos 1, partials 2..d+1).
            if i <= d && j == i + 1
                one(T)                  # z[i] = ∇u_i = y[i+1]
            elseif i == nz && j == nz
                one(T)                  # z[nz] = s = y[nz]
            else
                zero(T)
            end
        end, Val(nz * nz)))
    end
    b_ms = let nz = nz, d = d
        x -> SVector{nz,T}(ntuple(i -> i == d + 1 ? one(T) : zero(T), Val(nz)))
    end

    Q = convex_Euclidian_power(T; mg=mg,
            idx=SVector{nz,Int}(ntuple(i -> i, Val(nz))),
            A=A_ms, b=b_ms,
            p=x->T(1))                  # α = 2/p = 2 ⇒ barrier −log(s² − |q|²)

    return assemble(mg; state_variables, D, f=f_kw, g=g_kw, Q)
end
