using MultiGridBarrier
using Test
using LinearAlgebra

@testset "Mixed Dirichlet/Neumann BC (find_boundary, dirichlet_nodes)" begin
    # ─── fem1d ─────────────────────────────────────────────────
    # find_boundary returns broken-basis rows: row 1 (left end of element 1)
    # and row 2*n_e (right end of element n_e). nodes5 → 4 elements, 8 rows.
    nodes5 = collect(range(-1.0, 1.0, length=5))
    geom1  = fem1d(; nodes=nodes5)
    @test find_boundary(geom1) == [1, 8]
    sol_def  = mgb_solve(amg(geom1); p=2.0, verbose=false, tol=1e-4)
    sol_left = mgb_solve(amg(geom1; dirichlet_nodes=[1]); p=2.0, verbose=false, tol=1e-4)
    @test all(isfinite, sol_def.z)
    @test all(isfinite, sol_left.z)
    @test norm(sol_left.z - sol_def.z) > 1e-3        # mixed BC actually differs

    # ─── fem2d_P1 ──────────────────────────────────────────────
    geom2 = fem2d_P1()
    bdry2 = find_boundary(geom2)
    @test length(bdry2) >= 1
    @test all(1 .<= bdry2 .<= size(geom2.x, 1))
    half = bdry2[1:max(1, length(bdry2) ÷ 2)]
    sol_def_2  = mgb_solve(amg(geom2); p=2.0, verbose=false, tol=1e-3)
    sol_half_2 = mgb_solve(amg(geom2; dirichlet_nodes=half); p=2.0, verbose=false, tol=1e-3)
    @test all(isfinite, sol_def_2.z)
    @test all(isfinite, sol_half_2.z)

    # ─── fem2d_P2 ──────────────────────────────────────────────
    geom3 = fem2d_P2()
    bdry3 = find_boundary(geom3)
    @test length(bdry3) >= 1
    @test all(1 .<= bdry3 .<= size(geom3.x, 1))
    sol_def_3  = mgb_solve(amg(geom3); p=2.0, verbose=false, tol=1e-3)
    sol_half_3 = mgb_solve(amg(geom3; dirichlet_nodes=bdry3[1:max(1,length(bdry3) ÷ 2)]);
                            p=2.0, verbose=false, tol=1e-3)
    @test all(isfinite, sol_def_3.z)
    @test all(isfinite, sol_half_3.z)

    # ─── fem3d k=1 ─────────────────────────────────────────────
    geom4 = fem3d(; k=1)
    bdry4 = find_boundary(geom4)
    @test length(bdry4) >= 1
    @test all(1 .<= bdry4 .<= size(geom4.x, 1))
    sol_def_4  = mgb_solve(amg(geom4); p=2.0, verbose=false, tol=1e-3)
    sol_half_4 = mgb_solve(amg(geom4; dirichlet_nodes=bdry4[1:max(1,length(bdry4) ÷ 2)]);
                            p=2.0, verbose=false, tol=1e-3)
    @test all(isfinite, sol_def_4.z)
    @test all(isfinite, sol_half_4.z)

    # ─── fem3d k=2 (exercises non-corner boundary DOFs) ────────
    geom4b = fem3d(; k=2)
    bdry4b = find_boundary(geom4b)
    # k=2 unit cube: every interior face DOF is on the boundary. The single
    # internal node (the center of the hex) is the only non-boundary DOF.
    @test length(bdry4b) == size(geom4b.x, 1) - 1
    sol_def_4b = mgb_solve(amg(geom4b); p=2.0, verbose=false, tol=1e-3)
    @test all(isfinite, sol_def_4b.z)

    # ─── spectral methods: find_boundary informational only ────
    @test find_boundary(spectral1d(n=5)) == [1, 5]
    @test length(find_boundary(spectral2d(n=4))) == 4 * 4 - (4-2)^2  # perimeter
end
