# Pure-P2 (bubble = false) FEM2D_P2: constructors and back-compat, reference
# exactness (midpoint rule, isoparametric dx/dy), the :broken_P1 slack
# subspace, and solve parity with the bubble space on both hierarchy paths.
@testset "pure P2 (fem2d_P2 bubble=false)" begin
    T = Float64
    _k12(x, y) = (round(x; digits=12), round(y; digits=12))

    @testset "constructors and back-compat" begin
        g7 = fem2d_P2()
        @test g7.discretization isa FEM2D_P2{T,true}
        @test size(g7.x, 1) == 7
        g6 = fem2d_P2(bubble=false)
        @test g6.discretization isa FEM2D_P2{T,false}
        @test size(g6.x, 1) == 6
        @test g6.discretization.Kfull == g6.x
        @test g6.discretization.K == g6.x[[1, 3, 5], :, :]
        # the variant is inferred from the row count of K...
        @test fem2d_P2(K=g6.discretization.Kfull).discretization isa FEM2D_P2{T,false}
        @test fem2d_P2(K=g7.discretization.Kfull).discretization isa FEM2D_P2{T,true}
        # ...and a contradicting explicit bubble is refused
        @test_throws ArgumentError fem2d_P2(K=g6.discretization.Kfull, bubble=true)
        # corner-input type constructors: {T} keeps the historical bubble default
        K3 = g6.discretization.K
        @test FEM2D_P2{T}(K3) isa FEM2D_P2{T,true}
        @test FEM2D_P2{T,false}(K3) isa FEM2D_P2{T,false}
    end

    @testset "midpoint rule and operator exactness" begin
        gf = subdivide(fem2d_P2(bubble=false), 3)
        w = reshape(gf.w, 6, :)
        @test all(w[[1, 3, 5], :] .== 0)      # corner weights exactly zero
        @test all(w[[2, 4, 6], :] .> 0)       # midpoint weights positive
        # same mass convention as the bubble space
        @test sum(gf.w) ≈ sum(subdivide(fem2d_P2(), 3).w)
        xf = reshape(gf.x, :, 2)
        f(x, y)  = 1 + 2x - y + x^2 - 3x*y + 2y^2
        fx(x, y) = 2 + 2x - 3y
        fy(x, y) = -1 - 3x + 4y
        vals = [f(xf[i, 1], xf[i, 2]) for i in 1:size(xf, 1)]
        @test maximum(abs.(gf.operators[:dx]*vals .- fx.(xf[:, 1], xf[:, 2]))) < 1e-11
        @test maximum(abs.(gf.operators[:dy]*vals .- fy.(xf[:, 1], xf[:, 2]))) < 1e-11
        # the midpoint rule integrates quadratics identically to the bubble rule
        g7f = subdivide(fem2d_P2(), 3)
        x7 = reshape(g7f.x, :, 2)
        v7 = [f(x7[i, 1], x7[i, 2]) for i in 1:size(x7, 1)]
        @test dot(gf.w, vals) ≈ dot(g7f.w, v7)
        # boundary detection: every reported node sits on the square's boundary
        bd = find_boundary(gf)
        @test !isempty(bd)
        @test all(maximum(abs.(gf.x[v, e, :])) ≈ 1 for (v, e) in bd)
    end

    @testset ":broken_P1 slack and solve parity" begin
        mg = amg(subdivide(fem2d_P2(bubble=false), 3))
        @test haskey(mg.R, :broken_P1)
        sol = mgb_solve(assemble(mg; p=T(1.5)); verbose=false, tol=1e-8)
        s = reshape(sol.z[:, 2], 6, :)
        # the slack is genuinely per-element linear (vertex = adjacent midpoints
        # minus the opposite one), i.e. it lives in :broken_P1
        @test maximum(abs.(s[1, :] .- (s[2, :] .- s[4, :] .+ s[6, :]))) < 1e-10
        @test maximum(abs.(s[3, :] .- (s[2, :] .+ s[4, :] .- s[6, :]))) < 1e-10
        @test maximum(abs.(s[5, :] .- (.-s[2, :] .+ s[4, :] .+ s[6, :]))) < 1e-10

        # parity with the bubble space at shared geometric nodes: the gap is
        # quadrature-crime-sized and shrinks under refinement
        function pair_dev(L)
            a = mgb_solve(assemble(amg(subdivide(fem2d_P2(bubble=false), L)); p=T(1.5));
                          verbose=false, tol=1e-8)
            b = mgb_solve(assemble(amg(subdivide(fem2d_P2(), L)); p=T(1.5));
                          verbose=false, tol=1e-8)
            xa = reshape(a.geometry.x, :, 2)
            xb = reshape(b.geometry.x, :, 2)
            da = Dict(_k12(xa[i, 1], xa[i, 2]) => a.z[i, 1] for i in 1:size(xa, 1))
            maximum(abs(da[_k12(xb[i, 1], xb[i, 2])] - b.z[i, 1])
                    for i in 1:size(xb, 1) if haskey(da, _k12(xb[i, 1], xb[i, 2])))
        end
        d2 = pair_dev(2)
        d3 = pair_dev(3)
        @test d3 < d2
        @test d3 < 0.1

        # geometric_mg path solves too
        solg = mgb_solve(assemble(geometric_mg(fem2d_P2(bubble=false), 3); p=T(1.5));
                         verbose=false, tol=1e-8)
        @test isfinite(norm(solg.z))
        sg = reshape(solg.z[:, 2], 6, :)
        @test maximum(abs.(sg[1, :] .- (sg[2, :] .- sg[4, :] .+ sg[6, :]))) < 1e-10
    end
end
