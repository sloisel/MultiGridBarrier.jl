using MultiGridBarrier
using Test
using LinearAlgebra

# 3D Q_k FEM (FEM3D = TensorFEM{3}) behavioral tests: construction, isoparametric
# curved hexes, solver, parabolic, and PyVista plotting / animation. (The former
# Mesh3d-submodule unit tests were removed with the submodule; operator/quadrature
# exactness is covered dimension-generically in test_tensorfem.jl, boundary
# detection in test_mixed_bc.jl, and golden solves in test_algebraic.jl.)

_flat3(g) = reshape(g.x, :, size(g.x, 3))

# Build a single curved Q_k hex by bulging the interior/top nodes in +z
# (isoparametric: the displaced node positions curve the element).
function _curved_hex(::Type{T}, k::Int) where {T}
    g = fem3d(T; k=k)                       # straight unit cube; full node tensor in g.x
    X = Array{T,3}(g.x)                      # ((k+1)^3, 1, 3)
    for i in 1:(k+1)^3
        x, y, z = X[i,1,1], X[i,1,2], X[i,1,3]
        X[i,1,3] = z + T(0.1) * (1 - x^2) * (1 - y^2) * (z + 1) / 2
    end
    return X
end

@testset "FEM3D Q_k (hexahedral) tests" begin

@testset "construction k=$k" for k in (1, 2, 3)
    g = fem3d(; k=k)
    @test g.discretization isa FEM3D{Float64}
    @test g.discretization.k == k
    @test size(g.x) == ((k+1)^3, 1, 3)
    @test size(g.discretization.K) == (8, 1, 3)        # Q1 corner tensor
    @test g.operators[:dx] isa MultiGridBarrier.BlockDiag
    @test all(>(0), g.w)                               # strictly positive weights
end

@testset "Q1-corner shorthand == full-node tensor (straight)" begin
    g_full = fem3d(; k=2)
    g_corn = fem3d(; k=2, K=g_full.discretization.K)   # 8 corners -> straight promote
    @test g_full.x ≈ g_corn.x
end

@testset "isoparametric curved hex k=$k" for k in (2, 3)
    Xc = _curved_hex(Float64, k)
    g  = fem3d(; k=k, K=Xc)
    @test g.x ≈ Xc                                     # node positions honoured
    @test all(>(0), g.w)                               # curved map stays orientation-preserving
    sol = mgb_solve(assemble(amg(g)); p=2.0, verbose=false, tol=1e-6)
    @test all(isfinite, sol.z)
end

@testset "3D solver (subdivided)" begin
    sol = mgb_solve(assemble(amg(subdivide(fem3d(; k=1), 2))); tol=1e-6, verbose=false)
    @test sol isa MultiGridBarrier.MGBSOL
    @test all(isfinite, sol.z)
end

@testset "3D parabolic solver" begin
    sol = parabolic_solve(amg(fem3d()); h=0.5, verbose=false)
    @test sol isa MultiGridBarrier.ParabolicSOL
end

@testset "3D PyVista plotting" begin
    geo = fem3d(; k=1)
    u = rand(size(_flat3(geo), 1))
    fig = plot(geo, u; volume=(;), show_grid=false)
    @test fig isa MGB3DFigure
    fig_opts = plot(geo, u; volume=(cmap="magma",), isosurfaces=[0.5],
                    contour_mesh=(color="black",), slice_orthogonal=(x=0.5,))
    @test fig_opts isa MGB3DFigure
    fn = "test_plot.png"; savefig(fig, fn); @test isfile(fn); rm(fn)
    sol = mgb_solve(assemble(amg(fem3d(; k=1))); tol=1e-1, verbose=false)
    @test plot(sol) isa MGB3DFigure
    println("Plotting API tests passed!")
end

@testset "3D parabolic animation" begin
    sol = parabolic_solve(amg(fem3d()); h=0.5, verbose=false)
    @test plot(sol) isa MultiGridBarrier.HTML5anim
end

end # FEM3D Q_k tests
