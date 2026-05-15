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

@testset ":full per-subspace AMG (fem1d, all-corners Neumann)" begin
    # `fem1d`'s `amg()` builds two distinct AMG hierarchies: `:dirichlet`'s on
    # the interior P1 stiffness `K_full[interior, interior]`, and `:full`'s on
    # the all-corners P1 Neumann stiffness `K_full` (no boundary removal). The
    # two have different dimensions at every coarse level — interior corner
    # count vs all-corner count — and so their refine/coarsen matrices have
    # genuinely distinct shapes.
    nodes = collect(range(-1.0, 1.0, length=9))   # n=9, n_int=7, n_doubled=16
    geom = fem1d(; nodes=nodes)
    mg = amg(geom)
    # AMG-side level just before fine: :dirichlet bridges interior P1 (n_int=7)
    # to broken (n_doubled=16); :full bridges all-corners P1 (n=9) to broken.
    K_amg_dir  = length(mg.refine[:dirichlet]) - 1
    K_amg_full = length(mg.refine[:full]) - 1
    @test size(mg.refine[:dirichlet][K_amg_dir])  == (16, 7)
    @test size(mg.refine[:full][K_amg_full])      == (16, 9)
    # Subspaces at AMG levels are identities sized for each X's grid count.
    @test size(mg.subspaces[:dirichlet][K_amg_dir], 1) == 7
    @test size(mg.subspaces[:full][K_amg_full],     1) == 9
    # End-to-end p-Laplace solve with default state_variables uses :full
    # for the slack — exercises the per-subspace hierarchy through phase 2.
    sol = mgb_solve(mg; p=1.5, verbose=false, tol=1e-4)
    @test all(isfinite, sol.z)
end

@testset ":uniform lift to true constant at every level (regression)" begin
    # For a state variable declared `:uniform`, the level-l prolongation should
    # send a scalar c to the *globally constant* function `c * ones(n_doubled)`
    # at fine — not to the interior-corner indicator the shared (:dirichlet)
    # AMG bridge would otherwise produce.
    nodes = collect(range(-1.0, 1.0, length=17))
    mg = amg(fem1d(; nodes=nodes))
    state_variables = [:u :dirichlet; :s :uniform]
    D = [:u :id; :u :dx; :s :id]
    M, _ = MultiGridBarrier._prepare_amg(mg; state_variables=state_variables, D=D)
    L = length(M.R_fine)
    n_doubled = size(mg.x, 1)
    @test L >= 2  # exercise a real coarse level
    for l in 1:L
        # build a level-l joint iterate where :u is zero and :s = c (a single
        # scalar). The :u block has size = subspaces[:dirichlet][l] cols;
        # :s block has size 1.
        m_u = size(mg.subspaces[:dirichlet][l], 2)
        c   = 0.37
        z   = zeros(Float64, m_u + 1)
        z[end] = c                              # :s coefficient
        lifted = M.R_fine[l] * z                # joint fine-broken-basis vector
        # joint output has nu=2 blocks of length n_doubled each
        @test length(lifted) == 2 * n_doubled
        u_block = lifted[1:n_doubled]            # :u part — should be zero
        s_block = lifted[n_doubled+1:end]        # :s part — should be c*ones
        @test norm(u_block) < 1e-12
        @test all(abs.(s_block .- c) .< 1e-12)
    end
end
