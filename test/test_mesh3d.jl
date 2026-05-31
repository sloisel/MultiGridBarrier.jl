# Tests for Mesh3d submodule (3D FEM discretization)

using MultiGridBarrier
using MultiGridBarrier.Mesh3d
using LinearAlgebra
using SparseArrays
using Test

# Access internal functions via Mesh3d module
const M3d = MultiGridBarrier.Mesh3d

# Tests pre-date the 3-tensor `Geometry.x` and assume the legacy flat
# `(n_nodes, dim)` matrix layout. Provide a zero-copy flat view helper.
_xflat(g) = reshape(g.x, :, size(g.x, 3))

@testset "Mesh3d Tests" begin

@testset "Polynomial Reproduction" begin
    k = 3
    L = 2
    g = subdivide(fem3d(; k=k), L)

    println("Testing Polynomial Reproduction (k=$k, L=$L)")

    coeffs = rand(4, 4, 4)

    function poly(x, y, z)
        val = 0.0
        for l in 1:4, j in 1:4, i in 1:4
            val += coeffs[i,j,l] * x^(i-1) * y^(j-1) * z^(l-1)
        end
        return val
    end

    x_fine = _xflat(g)
    u_fine = zeros(size(x_fine, 1))
    for i in 1:size(x_fine, 1)
        u_fine[i] = poly(x_fine[i, 1], x_fine[i, 2], x_fine[i, 3])
    end

    n_points = 10
    max_err = 0.0
    for _ in 1:n_points
        pt = 2.0 .* rand(3) .- 1.0
        val_exact = poly(pt[1], pt[2], pt[3])
        val_interp, found = M3d.evaluate_field(g, u_fine, pt)

        if found
            err = abs(val_interp - val_exact)
            max_err = max(max_err, err)
        else
            println("Point $pt not found in mesh!")
        end
    end

    println("Max interpolation error: $max_err")
    @test max_err < 1e-10
end

@testset "Multigrid Operators" begin
    k = 3
    L = 2
    mg = amg(subdivide(fem3d(; k=k), L))

    println("Testing Multigrid Operators")

    Rd = mg.R[:dirichlet]
    n_fine = size(Rd[end], 1)
    # Every level's prolongation maps into the same fine broken basis.
    @test all(size(R, 1) == n_fine for R in Rd)
    # Coarser levels carry no more intrinsic DOFs than finer ones.
    @test all(size(Rd[l], 2) <= size(Rd[l+1], 2) for l in 1:length(Rd)-1)
end

@testset "Dirichlet Boundary" begin
    k = 3
    mg = amg(fem3d(; k=k))
    g = mg.geometry

    println("Testing Dirichlet Boundary")

    D = mg.R[:dirichlet][end]

    n_interior = size(D, 2)
    v = rand(n_interior)
    u = D * v

    boundary_unique_indices, _, node_map = M3d.get_boundary_nodes(_xflat(g), k)
    boundary_set = Set(boundary_unique_indices)

    boundary_element_indices = [i for i in 1:length(u) if node_map[i] in boundary_set]

    boundary_vals = u[boundary_element_indices]
    err = norm(boundary_vals, Inf)
    println("Max boundary value: $err")
    @test err < 1e-14
end

@testset "Projection" begin
    k = 3
    L = 2
    mg = amg(subdivide(fem3d(Float64; k=k), L))
    g = mg.geometry

    println("Testing Projection (k=$k, L=$L)")

    x_g = _xflat(g)
    u = zeros(size(x_g, 1))
    for i in 1:size(x_g, 1)
        x, y, z = x_g[i, :]
        u[i] = cos(pi*x/2) * cos(pi*y/2) * cos(pi*z/2)
    end

    D = mg.R[:dirichlet][end]

    DtD = D' * D
    invDtD = inv(Matrix(DtD))

    P = D * invDtD * D'
    u_proj = P * u

    boundary_unique_indices, _, node_map = M3d.get_boundary_nodes(x_g, k)
    boundary_set = Set(boundary_unique_indices)
    boundary_element_indices = [i for i in 1:length(u) if node_map[i] in boundary_set]

    println("Max u on boundary: $(norm(u[boundary_element_indices], Inf))")

    err = norm(u - u_proj, Inf)
    println("Projection error: $err")
    @test err < 1e-14
end

@testset "Quadrature Tests" begin
    k = 3
    L = 2
    geo = subdivide(fem3d(Float64; k=k), L)

    println("Testing Quadrature (k=$k, L=$L)")

    w = geo.w
    x = _xflat(geo)

    vol = sum(w)
    println("Volume: $vol (expected 8.0)")
    @test abs(vol - 8.0) < 1e-12

    int_x = sum(w .* x[:, 1])
    println("Integral x: $int_x (expected 0.0)")
    @test abs(int_x) < 1e-12

    int_x2 = sum(w .* x[:, 1].^2)
    println("Integral x^2: $int_x2 (expected $(8/3))")
    @test abs(int_x2 - 8/3) < 1e-12
end

@testset "Derivative Tests" begin
    k = 3
    L = 2
    geo = subdivide(fem3d(Float64; k=k), L)

    println("Testing Derivatives (k=$k, L=$L)")

    coeffs = rand(4, 4, 4)

    function poly(x, y, z)
        val = 0.0
        for l in 1:4, j in 1:4, i in 1:4
            val += coeffs[i,j,l] * x^(i-1) * y^(j-1) * z^(l-1)
        end
        return val
    end

    function poly_dx(x, y, z)
        val = 0.0
        for l in 1:4, j in 1:4, i in 2:4
            val += coeffs[i,j,l] * (i-1) * x^(i-2) * y^(j-1) * z^(l-1)
        end
        return val
    end

    function poly_dy(x, y, z)
        val = 0.0
        for l in 1:4, j in 2:4, i in 1:4
            val += coeffs[i,j,l] * (j-1) * x^(i-1) * y^(j-2) * z^(l-1)
        end
        return val
    end

    function poly_dz(x, y, z)
        val = 0.0
        for l in 2:4, j in 1:4, i in 1:4
            val += coeffs[i,j,l] * (l-1) * x^(i-1) * y^(j-1) * z^(l-2)
        end
        return val
    end

    x_fine = _xflat(geo)
    n_nodes = size(x_fine, 1)

    u = zeros(n_nodes)
    u_exact_dx = zeros(n_nodes)
    u_exact_dy = zeros(n_nodes)
    u_exact_dz = zeros(n_nodes)

    for i in 1:n_nodes
        x, y, z = x_fine[i, :]
        u[i] = poly(x, y, z)
        u_exact_dx[i] = poly_dx(x, y, z)
        u_exact_dy[i] = poly_dy(x, y, z)
        u_exact_dz[i] = poly_dz(x, y, z)
    end

    Dx = geo.operators[:dx]
    Dy = geo.operators[:dy]
    Dz = geo.operators[:dz]

    u_dx = Dx * u
    u_dy = Dy * u
    u_dz = Dz * u

    err_dx = norm(u_dx - u_exact_dx, Inf)
    err_dy = norm(u_dy - u_exact_dy, Inf)
    err_dz = norm(u_dz - u_exact_dz, Inf)

    println("Error dx: $err_dx")
    println("Error dy: $err_dy")
    println("Error dz: $err_dz")

    @test err_dx < 1e-12
    @test err_dy < 1e-12
    @test err_dz < 1e-12
end

@testset "FEM3D Struct Tests" begin
    k = 2
    L = 2
    geo = subdivide(fem3d(Float64; k=k), L)

    @test geo.discretization isa FEM3D{Float64}
    @test geo.discretization.k == k

    @test size(geo.discretization.K, 3) == 3   # 3-tensor (8, N, 3)
end

@testset "Point Location Tests" begin
    println("Testing Point Location")

    k = 3
    geo_cube = fem3d(Float64; k=k)

    # Function u(x,y,z) = x + 2y + 3z
    x_cube = _xflat(geo_cube)
    u_cube = zeros(size(x_cube, 1))
    for i in 1:size(x_cube, 1)
        x, y, z = x_cube[i, :]
        u_cube[i] = x + 2*y + 3*z
    end

    pt_in = [0.5, -0.2, 0.1]
    val, found = M3d.evaluate_field(geo_cube, u_cube, pt_in)
    @test found
    @test isapprox(val, pt_in[1] + 2*pt_in[2] + 3*pt_in[3], atol=1e-10)

    pt_out = [2.0, 0.0, 0.0]
    val, found = M3d.evaluate_field(geo_cube, u_cube, pt_out)
    @test !found
    @test val == 0.0
end

@testset "Solver Tests" begin
    println("Testing 3D mgb_solve")

    k = 1
    L = 2

    sol = mgb_solve(assemble(amg(subdivide(fem3d(; k=k), L))); tol=1e-6, maxiter=10, verbose=false)

    println("Solver returned: $(typeof(sol))")
    @test sol isa MultiGridBarrier.MGBSOL
end

@testset "Parabolic Solver Tests" begin
    println("Testing parabolic_solve with FEM3D")

    sol = parabolic_solve(amg(fem3d()); h=0.5, verbose=false)
    @test sol isa MultiGridBarrier.ParabolicSOL
end

@testset "Plotting API Tests" begin
    println("Testing Plotting API...")

    geo = fem3d(; k=1)
    n_nodes = size(_xflat(geo), 1)
    u = rand(n_nodes)

    println("Testing plot(geo, u)...")
    fig = plot(geo, u; volume=(;), show_grid=false)
    @test fig isa M3d.Plotting.MGB3DFigure

    println("Testing plot with options...")
    fig_opts = plot(geo, u;
        volume=(cmap="magma",),
        isosurfaces=[0.5],
        contour_mesh=(color="black",),
        slice_orthogonal=(x=0.5,)
    )
    @test fig_opts isa M3d.Plotting.MGB3DFigure

    println("Testing savefig...")
    filename = "test_plot.png"
    savefig(fig, filename)
    @test isfile(filename)
    rm(filename)

    println("Testing plot(sol)...")
    sol = mgb_solve(assemble(amg(fem3d(; k=1))); maxiter=1, verbose=false)
    fig_sol = plot(sol)
    @test fig_sol isa M3d.Plotting.MGB3DFigure

    println("Plotting API tests passed!")
end

@testset "Parabolic Animation Tests" begin
    println("Testing parabolic animation plot...")

    sol = parabolic_solve(amg(fem3d()); h=0.5, verbose=false)
    fig = plot(sol)
    @test fig isa MultiGridBarrier.HTML5anim
end

end # Mesh3d Tests
