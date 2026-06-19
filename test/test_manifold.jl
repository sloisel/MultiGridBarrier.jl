using MultiGridBarrier
using Test
using LinearAlgebra
using StaticArrays

# Full-solve tests for embedded manifolds: curves in R^2/R^3 and surfaces in R^3.
# Each is a scalar p-Laplace problem WITH a mass term, so the closed (boundaryless)
# manifolds are well-posed:
#
#     min_u  ∫_Γ |∇_Γ u|^p ds  +  ∫_Γ (u - u0)^2 ds          (gradient cone + p=2 data cone)
#
# At p=2 the Euler–Lagrange equation is  -Δ_Γ u + u = u0, so picking
# u0 = (λ_k + 1)·φ_k for a Laplace–Beltrami eigenpair (Δ_Γ φ_k = -λ_k φ_k) makes the
# exact solution u = φ_k. The gradient cone uses the AMBIENT partials :dx,:dy[,:dz]
# (the components of the intrinsic gradient ∇_Γ), so the same builder works in any
# codimension; correctness is therefore embedding-independent, which we also check
# directly (a circle solved in R^2 and in a tilted plane of R^3 must agree).

# Scalar p-Laplace + mass; objective ∫(s + r) with s ≥ |∇_Γu|^p and r ≥ (u-u0)^2.
function _manifold_plap(mg; p, u0::Function, s_init=100.0, r_init=400.0)
    T = Float64; geom = mg.geometry; Da = size(geom.x, 3)   # Da = ambient dimension
    state_variables = [:u :dirichlet; :s :full; :r :full]
    op = (:dx, :dy, :dz)
    rows = Any[[:u :id]]; for j in 1:Da; push!(rows, [:u op[j]]); end
    push!(rows, [:s :id]); push!(rows, [:r :id]); D = vcat(rows...)
    nr = Da + 3                                              # u, Da partials, s, r
    f_kw = x -> SVector{nr,T}(ntuple(k -> (k == nr-1 || k == nr) ? one(T) : zero(T), Val(nr)))
    g_kw = x -> SVector{3,T}(zero(T), T(s_init), T(r_init))
    gidx = SVector{Da+1,Int}(ntuple(k -> k <= Da ? 1+k : nr-1, Val(Da+1)))   # partials + s
    Qg = MultiGridBarrier.convex_Euclidian_power(T; mg=mg, idx=gidx, p=x->T(p))
    didx = SVector{2,Int}(1, nr)                                             # (u, r)
    Ad = x -> SMatrix{2,2,T}(one(T), zero(T), zero(T), one(T))
    bd = x -> SVector{2,T}(-T(u0(x)), zero(T))
    Qd = MultiGridBarrier.convex_Euclidian_power(T; mg=mg, idx=didx, A=Ad, b=bd, p=x->T(2))
    assemble(mg; state_variables, D, f=f_kw, g=g_kw, Q=MultiGridBarrier.intersect(mg, Qg, Qd))
end

# unit circle as N straight Q1 segments (dedup glues the seam ⇒ closed loop)
_circle2(N) = (θ = range(0, 2π, length=N+1); K = Array{Float64,3}(undef, 2, N, 2);
    for e in 1:N
        K[1,e,1]=cos(θ[e]);   K[1,e,2]=sin(θ[e])
        K[2,e,1]=cos(θ[e+1]); K[2,e,2]=sin(θ[e+1])
    end; fem1d(Float64; K=K, ambient=Val(2)))
# the SAME unit circle tilted into a plane of R^3 (isometric ⇒ identical solution)
const _ALPHA = 0.7
_circle3(N) = (θ = range(0, 2π, length=N+1); K = Array{Float64,3}(undef, 2, N, 3);
    f = t -> (cos(t), sin(t)*cos(_ALPHA), sin(t)*sin(_ALPHA));
    for e in 1:N; K[1,e,:] .= f(θ[e]); K[2,e,:] .= f(θ[e+1]); end; fem1d(Float64; K=K, ambient=Val(3)))
_θ2(x) = atan(x[2], x[1])
_θ3(x) = atan(x[2]*cos(_ALPHA) + x[3]*sin(_ALPHA), x[1])

# cubed sphere: 6 faces × m² Q1 quads, corners projected to the unit sphere
function _sphere(m)
    faces = [(u,v)->(1.0,u,v), (u,v)->(-1.0,u,v), (u,v)->(u,1.0,v),
             (u,v)->(u,-1.0,v), (u,v)->(u,v,1.0), (u,v)->(u,v,-1.0)]
    t = collect(range(-1, 1, length=m+1)); Kt = Array{Float64,3}(undef, 4, 6m*m, 3); c = 0
    for f in faces, i in 1:m, j in 1:m
        c += 1
        for (ci,(uu,vv)) in enumerate(((t[i],t[j]), (t[i+1],t[j]), (t[i],t[j+1]), (t[i+1],t[j+1])))
            p = collect(f(uu,vv)); p ./= norm(p); Kt[ci,c,:] .= p
        end
    end
    fem2d(Float64; K=Kt, ambient=Val(3))
end

_nodevals(g, f) = [f(g.x[v,e,:]) for e in 1:size(g.x,2) for v in 1:size(g.x,1)]
_wmean(g, u) = dot(g.w, u) / sum(g.w)

@testset "embedded-manifold p-Laplace + mass solves" begin
    @testset "closed circle (R^2 & R^3), p=2, manufactured u=cos 2θ" begin
        m = 2; uex2(x) = cos(m*_θ2(x)); uex3(x) = cos(m*_θ3(x))
        u02(x) = (m^2+1)*cos(m*_θ2(x)); u03(x) = (m^2+1)*cos(m*_θ3(x))
        errs = Float64[]
        for N in (64, 128)
            s2 = mgb_solve(_manifold_plap(amg(_circle2(N)); p=2.0, u0=u02); verbose=false, tol=1e-9)
            s3 = mgb_solve(_manifold_plap(amg(_circle3(N)); p=2.0, u0=u03); verbose=false, tol=1e-9)
            e2 = maximum(abs.(s2.z[:,1] .- _nodevals(s2.geometry, uex2)))
            e3 = maximum(abs.(s3.z[:,1] .- _nodevals(s3.geometry, uex3)))
            @test e2 < 0.004                              # converges to the exact eigenfunction
            @test e3 ≈ e2 rtol=1e-6                        # R^3 embedding matches R^2 (intrinsic)
            @test maximum(abs.(s2.z[:,1] .- s3.z[:,1])) < 1e-10   # ...to machine precision
            push!(errs, e2)
        end
        @test errs[2] < 0.4 * errs[1]                     # ~O(h^2) under refinement
    end

    @testset "closed circle, p=1.5, embedding independence" begin
        u02(x) = 3*cos(2*_θ2(x)); u03(x) = 3*cos(2*_θ3(x))
        s2 = mgb_solve(_manifold_plap(amg(_circle2(96)); p=1.5, u0=u02); verbose=false, tol=1e-9)
        s3 = mgb_solve(_manifold_plap(amg(_circle3(96)); p=1.5, u0=u03); verbose=false, tol=1e-9)
        @test maximum(abs, s2.z[:,1]) > 0.5                       # nontrivial solution
        @test maximum(abs.(s2.z[:,1] .- s3.z[:,1])) < 1e-10       # R^2 ≡ R^3 for nonlinear p
    end

    @testset "closed sphere in R^3, p=2, manufactured u=z" begin
        u0(x) = 3*x[3]; zex(x) = x[3]                     # -Δ_Γ z = 2z ⇒ u0 = (2+1)z
        e8 = let s = mgb_solve(_manifold_plap(amg(_sphere(8)); p=2.0, u0=u0); verbose=false, tol=1e-9)
            maximum(abs.(s.z[:,1] .- _nodevals(s.geometry, zex)))
        end
        e12 = let s = mgb_solve(_manifold_plap(amg(_sphere(12)); p=2.0, u0=u0); verbose=false, tol=1e-9)
            maximum(abs.(s.z[:,1] .- _nodevals(s.geometry, zex)))
        end
        @test e8 < 0.01                                   # converges to z (a degree-1 harmonic)
        @test e12 < e8                                    # ...and improves under refinement
    end

    @testset "closed sphere in R^3, p=1.5, symmetry/sanity" begin
        s = mgb_solve(_manifold_plap(amg(_sphere(8)); p=1.5, u0=(x->3*x[3])); verbose=false, tol=1e-9)
        @test 0.5 < maximum(abs, s.z[:,1]) < 2.0          # nontrivial, bounded
        @test abs(_wmean(s.geometry, s.z[:,1])) < 0.05    # odd-in-z data ⇒ (near) zero mean
    end
end

# PyVista surface/curve rendering (smoke tests: a non-empty PNG comes back). These
# exercise the e-dispatch added for embedded manifolds: TensorFEM{2,3} → quad surface,
# TensorFEM{1,3}/{1,2} → tubed/lifted curve.
@testset "embedded-manifold plotting (PyVista)" begin
    gs = _sphere(6)                                       # surface in ℝ³ (quad cells)
    figs = plot(gs, _nodevals(gs, x -> x[3]))
    @test figs isa MultiGridBarrier.MGB3DFigure
    @test length(figs.png) > 1000
    g3 = _circle3(48)                                     # curve in ℝ³ (tube)
    @test length(plot(g3, _nodevals(g3, x -> cos(_θ3(x)))).png) > 1000
    @test length(plot(g3, _nodevals(g3, x -> cos(_θ3(x))); tube=nothing).png) > 1000  # bare lines
    g2 = _circle2(48)                                     # curve in ℝ² (lifted to z=0)
    @test length(plot(g2, _nodevals(g2, x -> cos(_θ2(x)))).png) > 1000
end
