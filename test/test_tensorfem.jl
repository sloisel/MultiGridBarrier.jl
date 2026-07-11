using MultiGridBarrier
using Test
using LinearAlgebra

# Fully-qualified to avoid colliding with same-named helpers other test files
# define in `Main` (e.g. test_mesh3d.jl's `_xflat`).
const _TS = MultiGridBarrier.to_sparse
const _BD = MultiGridBarrier.BlockDiag
_flat(g) = reshape(g.x, :, size(g.x, 3))   # flat (V*N, D) node-coordinate view

# m×m unit-square quad mesh on [-1,1]^2 (Q1 corner tensor, tensor corner order).
function _square_quads(::Type{T}, m::Int) where {T}
    xs = collect(range(T(-1), T(1), length=m+1))
    K = Array{T,3}(undef, 4, m*m, 2)
    e = 0
    for jy in 1:m, jx in 1:m
        e += 1
        x0, x1 = xs[jx], xs[jx+1]
        y0, y1 = xs[jy], xs[jy+1]
        cs = T[x0 y0; x1 y0; x0 y1; x1 y1]   # corners (-1,-1),(+1,-1),(-1,+1),(+1,+1)
        for c in 1:4, d in 1:2
            K[c, e, d] = cs[c, d]
        end
    end
    K
end

@testset "TensorFEM Q_k (fem1d/fem2d) operator + quadrature consistency" begin

    # ── 1D: dx differentiates degree-≤k polynomials exactly; CC quadrature
    #        integrates them exactly. ────────────────────────────────────────
    @testset "fem1d k=$k operators" for k in 1:4
        nodes = collect(range(-1.0, 1.0, length=4))     # 3 elements
        geom  = fem1d(; nodes=nodes, k=k)
        x = vec(_flat(geom))
        Dx = _TS(geom.operators[:dx])
        for a in 0:k
            u   = x .^ a
            du  = a == 0 ? zeros(length(x)) : a .* x .^ (a-1)
            @test norm(Dx * u - du) < 1e-9
            # ∫_{-1}^1 x^a = 2/(a+1) for even a, 0 for odd a (mesh tiles [-1,1])
            exact = iseven(a) ? 2 / (a + 1) : 0.0
            @test abs(sum(geom.w .* u) - exact) < 1e-9
        end
        @test geom.operators[:dx] isa _BD
    end

    # ── 2D: dx, dy differentiate tensor monomials x^a y^b (a,b ≤ k) exactly;
    #        tensor CC quadrature integrates them exactly. ───────────────────
    @testset "fem2d k=$k operators" for k in 1:3
        geom = fem2d(; k=k, K=_square_quads(Float64, 2))   # 2×2 quads
        X = _flat(geom); xx = X[:,1]; yy = X[:,2]
        Dx = _TS(geom.operators[:dx])
        Dy = _TS(geom.operators[:dy])
        for a in 0:k, b in 0:k
            u    = (xx .^ a) .* (yy .^ b)
            dudx = (a == 0 ? zeros(length(xx)) : a .* xx .^ (a-1)) .* (yy .^ b)
            dudy = (xx .^ a) .* (b == 0 ? zeros(length(yy)) : b .* yy .^ (b-1))
            @test norm(Dx * u - dudx) < 1e-8
            @test norm(Dy * u - dudy) < 1e-8
            ix = iseven(a) ? 2 / (a + 1) : 0.0
            iy = iseven(b) ? 2 / (b + 1) : 0.0
            @test abs(sum(geom.w .* u) - ix * iy) < 1e-8
        end
        @test geom.operators[:dx] isa _BD
        @test geom.operators[:dy] isa _BD
    end

    # ── 3D: dx, dy, dz differentiate tensor monomials x^a y^b z^c (a,b,c ≤ k)
    #        exactly; tensor CC quadrature integrates them exactly. ───────────
    @testset "fem3d k=$k operators" for k in 1:2
        geom = fem3d(; k=k)                                  # single straight unit cube
        X = _flat(geom); xx = X[:,1]; yy = X[:,2]; zz = X[:,3]
        Dx = _TS(geom.operators[:dx]); Dy = _TS(geom.operators[:dy]); Dz = _TS(geom.operators[:dz])
        for a in 0:k, b in 0:k, c in 0:k
            u    = (xx .^ a) .* (yy .^ b) .* (zz .^ c)
            dudx = (a == 0 ? zeros(length(xx)) : a .* xx .^ (a-1)) .* (yy .^ b) .* (zz .^ c)
            dudy = (xx .^ a) .* (b == 0 ? zeros(length(yy)) : b .* yy .^ (b-1)) .* (zz .^ c)
            dudz = (xx .^ a) .* (yy .^ b) .* (c == 0 ? zeros(length(zz)) : c .* zz .^ (c-1))
            @test norm(Dx * u - dudx) < 1e-8
            @test norm(Dy * u - dudy) < 1e-8
            @test norm(Dz * u - dudz) < 1e-8
            ix = iseven(a) ? 2/(a+1) : 0.0
            iy = iseven(b) ? 2/(b+1) : 0.0
            iz = iseven(c) ? 2/(c+1) : 0.0
            @test abs(sum(geom.w .* u) - ix * iy * iz) < 1e-8
        end
        @test geom.operators[:dz] isa _BD
    end

    # ── find_boundary: single unit quad, every non-interior DOF is on ∂Ω
    #     (k=2 leaves exactly one interior node). ───────────────────────────
    @testset "fem2d find_boundary" begin
        g1 = fem2d(; k=1)                    # one quad, 4 corners, all boundary
        @test length(find_boundary(g1)) == 4
        g2 = fem2d(; k=2)                    # 3×3 nodes, center is the only interior
        nnode = size(g2.x, 1) * size(g2.x, 2)
        @test length(find_boundary(g2)) == nnode - 1
    end

    # ── fem1d interpolate: exact for degree-≤k polynomials (and clamps). ───
    @testset "fem1d interpolate k=$k exact" for k in 1:3
        nodes = collect(range(-1.0, 1.0, length=4))
        geom  = fem1d(; nodes=nodes, k=k)
        xf    = vec(_flat(geom))
        coeffs = ([0.7, -0.4, 0.9, -0.25])[1:k+1]
        poly(t) = sum(coeffs[j+1] * t^j for j in 0:k)
        z = poly.(xf)
        for t in (-0.91, -0.3, 0.07, 0.55, 0.985)
            @test abs(interpolate(geom, z, t) - poly(t)) < 1e-10
        end
        @test interpolate(geom, z, [-3.0, 3.0]) ≈ [z[1], z[end]]   # clamp outside
    end

    # ── End-to-end solves are finite for k=1..3 (1D and 2D). ───────────────
    @testset "fem solves finite" begin
        for k in 1:3
            s1 = mgb_solve(assemble(amg(fem1d(; nodes=collect(range(-1.0,1.0,length=4)), k=k)));
                           p=2.0, verbose=false, tol=1e-6)
            @test all(isfinite, s1.z)
            s2 = mgb_solve(assemble(amg(fem2d(; k=k, K=_square_quads(Float64, 2))));
                           p=2.0, verbose=false, tol=1e-6)
            @test all(isfinite, s2.z)
        end
    end

    @testset "fem1d defaults and curved interpolation" begin
        default_geom = fem1d()
        @test vec(default_geom.x) == [-1.0, 1.0]

        # x(ξ) = ξ + 0.2(1-ξ²), sampled at the Q2 nodes. Interpolation
        # must invert this nonlinear element map before evaluating the Q2 field ξ².
        K = reshape([-1.0, 0.2, 1.0], 3, 1, 1)
        curved = fem1d(; k=2, K=K)
        ξ = 0.3
        tq = ξ + 0.2 * (1 - ξ^2)
        @test interpolate(curved, [1.0, 0.0, 1.0], tq) ≈ ξ^2 atol=1e-13
    end

    @testset "geometric refinement preserves explicit topology" begin
        K = Array{Float64,3}(undef, 4, 2, 2)
        K[:, 1, 1] = [0, 1, 0, 1]
        K[:, 1, 2] = [0, 0, 1, 1]
        K[:, 2, 1] = [1, 2, 1, 2]
        K[:, 2, 2] = [0, 0, 1, 1]
        glued_corners = [1 2; 2 5; 3 4; 4 6]
        slit_corners  = [1 7; 2 5; 3 8; 4 6]
        glued = fem2d(; k=2, K=K, t=tensor_dofmap(glued_corners, 2, Val(2)))
        slit  = fem2d(; k=2, K=K, t=tensor_dofmap(slit_corners, 2, Val(2)))

        @test subdivide(slit, 1).t == slit.t
        @test maximum(subdivide(glued, 2).t) == 45
        @test maximum(subdivide(slit, 2).t) == 50
    end

    @testset "spectral interpolation shapes and abstract inputs" begin
        g1 = spectral1d(; n=6)
        x1 = vec(g1.x)
        f1(x) = x^3 - 2x + 1
        z1 = f1.(x1)
        z1view = @view z1[:]
        @test interpolate(g1, z1view, 0.2f0) ≈ f1(0.2f0)
        qrange = range(-0.5, 0.5, length=5)
        @test interpolate(g1, z1view, qrange) ≈ f1.(qrange)
        qmatrix = reshape(collect(qrange[1:4]), 2, 2)
        @test size(interpolate(g1, z1view, @view(qmatrix[:, :]))) == size(qmatrix)

        g2 = spectral2d(; n=5)
        x2 = reshape(g2.x, :, 2)
        z2 = x2[:, 1].^2 .+ 3 .* x2[:, 2]
        points = [0.2 -0.3; 0.5 0.1]
        @test interpolate(g2, @view(z2[:]), @view(points[1, :])) ≈ -0.86
        @test interpolate(g2, @view(z2[:]), @view(points[:, :])) ≈ [-0.86, 0.55]
    end
end
