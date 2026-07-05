# Gmsh import tests (MultiGridBarrierGmshExt). Gmsh is a test-only dependency;
# `using Gmsh` loads the extension. Geometry is scripted through the gmsh API, so
# no mesh files are needed.
#
# The workhorse check is linear reproduction: with f = 0 and Dirichlet data
# g(x) = affine, the p-harmonic minimizer is that affine function, and every
# family here represents affine functions exactly (isoparametric elements
# contain physical linears), so the discrete solution must match to solver
# tolerance wherever the nodal quadrature is exact for the stationarity test
# (P1/P2 triangles; tensor elements on rectangles/boxes). On genuinely curved
# elements the nodal quadrature is inexact, so the disk test uses a loose gap
# plus exact structural checks (area, boundary-node radii).
using MultiGridBarrier, Test
using Gmsh: gmsh

# p-harmonic solve with Dirichlet data `gl` on `pairs`; max |u - gl| at the nodes.
function _lin_gap(geom, pairs; p = 1.5,
                  gl = x -> 1 + 2x[1] + 3x[2] + (length(x) >= 3 ? 4x[3] : 0.0))
    dim = size(geom.x, 3)
    mg = amg(geom; dirichlet_nodes = Dict(:dirichlet => pairs))
    nD = dim + 2
    fq = x -> ntuple(i -> i == nD ? 1.0 : 0.0, nD)   # cost: ∫ s only
    gq = x -> (gl(x), 100.0)
    sol = mgb_solve(assemble(mg; p = p, f = fq, g = gq); verbose = false)
    xf = reshape(sol.geometry.x, :, dim)
    maximum(abs.(sol.z[:, 1] .- [gl(xf[i, :]) for i in 1:size(xf, 1)]))
end

@testset "Gmsh import" begin
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try
        @testset "P1 + P2 triangles (unstructured square)" begin
            gmsh.clear()
            gmsh.model.occ.addRectangle(-1.0, -1.0, 0.0, 2.0, 2.0)
            gmsh.model.occ.synchronize()
            gmsh.model.addPhysicalGroup(1,
                [t for (d, t) in gmsh.model.getEntities(1)], -1, "boundary")
            gmsh.option.setNumber("Mesh.MeshSizeMax", 0.4)
            gmsh.model.mesh.generate(2)
            gm = gmsh_import()
            @test typeof(gm.geometry.discretization) <: FEM2D_P1
            @test length(gm.regions["boundary"]) > 8
            @test _lin_gap(gm.geometry, gm.regions["boundary"]) < 1e-6
            gmsh.model.mesh.setOrder(2)
            gm2 = gmsh_import()
            @test typeof(gm2.geometry.discretization) <: FEM2D_P2
            @test _lin_gap(gm2.geometry, gm2.regions["boundary"]) < 1e-6
        end

        # Quadrilaterals import at ANY order: geometry is resampled at the
        # Chebyshev reference nodes. Tests k = 1..4, including the corner-subset
        # region test on high-order edge DOFs (mixed BC via named groups).
        @testset "Q$k quads (transfinite) + named groups" for k in 1:4
            gmsh.clear()
            gmsh.model.occ.addRectangle(-1.0, -1.0, 0.0, 2.0, 2.0)
            gmsh.model.occ.synchronize()
            for (d, t) in gmsh.model.getEntities(1)
                gmsh.model.mesh.setTransfiniteCurve(t, 6)
            end
            for (d, t) in gmsh.model.getEntities(2)
                gmsh.model.mesh.setTransfiniteSurface(t)
                gmsh.model.mesh.setRecombine(2, t)
            end
            # NB: OCC bounding boxes are padded by ~1e-7, so classify with 1e-6.
            left = Int[]; right = Int[]; all1 = Int[]
            for (d, t) in gmsh.model.getEntities(1)
                push!(all1, t)
                xmin, _, _, xmax, _, _ = gmsh.model.getBoundingBox(d, t)
                (abs(xmin + 1) < 1e-6 && abs(xmax + 1) < 1e-6) && push!(left, t)
                (abs(xmin - 1) < 1e-6 && abs(xmax - 1) < 1e-6) && push!(right, t)
            end
            gmsh.model.addPhysicalGroup(1, all1, -1, "boundary")
            gmsh.model.addPhysicalGroup(1, left, -1, "left")
            gmsh.model.addPhysicalGroup(1, right, -1, "right")
            gmsh.model.mesh.generate(2)
            k > 1 && gmsh.model.mesh.setOrder(k)
            gm = gmsh_import()
            @test typeof(gm.geometry.discretization) <: TensorFEM
            @test gm.geometry.discretization.k == k
            @test _lin_gap(gm.geometry, gm.regions["boundary"]) < 1e-6
            # mixed BC through named groups: Dirichlet u = 1+2x on left+right,
            # natural (Neumann) top/bottom -> exact solution 1+2x
            lr = sort(vcat(gm.regions["left"], gm.regions["right"]))
            @test _lin_gap(gm.geometry, lr; gl = x -> 1 + 2x[1]) < 1e-6
            @test all(abs(gm.geometry.x[v, e, 1] + 1) < 1e-9 for (v, e) in gm.regions["left"])
        end

        @testset "curved disk, Q$k" for k in 2:3
            gmsh.clear()
            gmsh.model.occ.addDisk(0.0, 0.0, 0.0, 1.0, 1.0)
            gmsh.model.occ.synchronize()
            gmsh.model.addPhysicalGroup(1,
                [t for (d, t) in gmsh.model.getEntities(1)], -1, "circle")
            gmsh.option.setNumber("Mesh.MeshSizeMax", 0.35)
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)  # all-quad
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.setOrder(k)
            gm = gmsh_import()
            @test gm.geometry.discretization.k == k
            @test abs(sum(gm.geometry.w) - π) / π < 5e-3          # curved area
            @test all(abs(hypot(gm.geometry.x[v, e, 1], gm.geometry.x[v, e, 2]) - 1) < 1e-6
                      for (v, e) in gm.regions["circle"])          # nodes on the circle
            # Sanity solve on the curved import. Unlike straight elements, a linear
            # function is NOT reproduced to machine precision here: on curved
            # (isoparametric) elements the energy integrand is non-polynomial, so the
            # quadrature — and thus the discrete minimizer — carries an O(curvature)
            # error that grows with order on a coarse mesh. A broken import would give
            # an O(1) gap; this only asserts the solve is fundamentally correct.
            @test _lin_gap(gm.geometry, gm.regions["circle"]) < 1e-2
            gmsh.option.setNumber("Mesh.RecombineAll", 0)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 0)
        end

        # Hexes import at any order too: tensor_dofmap now numbers shared 3D
        # face-interior grids (k >= 3). The transfinite box shares faces between
        # neighbouring hexes in mixed orientations, exercising that gluing.
        @testset "hexes Q$k (transfinite box)" for k in 1:3
            gmsh.clear()
            gmsh.model.occ.addBox(-1.0, -1.0, -1.0, 2.0, 2.0, 2.0)
            gmsh.model.occ.synchronize()
            gmsh.model.addPhysicalGroup(2,
                [t for (d, t) in gmsh.model.getEntities(2)], -1, "boundary")
            for (d, t) in gmsh.model.getEntities(1)
                gmsh.model.mesh.setTransfiniteCurve(t, 3)
            end
            for (d, t) in gmsh.model.getEntities(2)
                gmsh.model.mesh.setTransfiniteSurface(t)
                gmsh.model.mesh.setRecombine(2, t)
            end
            for (d, t) in gmsh.model.getEntities(3)
                gmsh.model.mesh.setTransfiniteVolume(t)
            end
            gmsh.model.mesh.generate(3)
            k > 1 && gmsh.model.mesh.setOrder(k)
            gm = gmsh_import()
            @test gm.geometry.discretization.k == k
            @test _lin_gap(gm.geometry, gm.regions["boundary"]) < 1e-6
        end

        @testset "subdomain physical group (On-style regions)" begin
            gmsh.clear()
            r1 = gmsh.model.occ.addRectangle(-1.0, -1.0, 0.0, 1.0, 2.0)
            r2 = gmsh.model.occ.addRectangle(0.0, -1.0, 0.0, 1.0, 2.0)
            gmsh.model.occ.fragment([(2, r1)], [(2, r2)])
            gmsh.model.occ.synchronize()
            leftsurf = Int[]
            for (d, t) in gmsh.model.getEntities(2)
                xmin, _, _, xmax, _, _ = gmsh.model.getBoundingBox(2, t)
                xmax < 1e-6 && push!(leftsurf, t)   # OCC boxes are padded ~1e-7
            end
            gmsh.model.addPhysicalGroup(2, leftsurf, -1, "left_half")
            gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
            gmsh.model.mesh.generate(2)
            gm = gmsh_import()
            @test length(gm.regions["left_half"]) > 4
            @test all(gm.geometry.x[v, e, 1] < 1e-9 for (v, e) in gm.regions["left_half"])
        end

        @testset "subdomain physical group on quads (tensor path)" begin
            # same split domain, all-quad: subdomain membership on the tensor
            # families goes through the element-tag branch of _regions_tensor
            # (every DOF of every element of the group), not node tags.
            gmsh.clear()
            r1 = gmsh.model.occ.addRectangle(-1.0, -1.0, 0.0, 1.0, 2.0)
            r2 = gmsh.model.occ.addRectangle(0.0, -1.0, 0.0, 1.0, 2.0)
            gmsh.model.occ.fragment([(2, r1)], [(2, r2)])
            gmsh.model.occ.synchronize()
            leftsurf = Int[]
            for (d, t) in gmsh.model.getEntities(2)
                xmin, _, _, xmax, _, _ = gmsh.model.getBoundingBox(2, t)
                xmax < 1e-6 && push!(leftsurf, t)
            end
            gmsh.model.addPhysicalGroup(2, leftsurf, -1, "left_half")
            gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)  # all-quad
            gmsh.model.mesh.generate(2)
            gm = gmsh_import()
            gmsh.option.setNumber("Mesh.RecombineAll", 0)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 0)
            @test typeof(gm.geometry.discretization) <: TensorFEM
            left = gm.regions["left_half"]
            @test !isempty(left)
            @test all(gm.geometry.x[v, e, 1] < 1e-9 for (v, e) in left)
            # whole-element semantics: every DOF of every member element
            es = unique(e for (v, e) in left)
            @test length(left) == size(gm.geometry.x, 1) * length(es)
        end

        @testset "import from a .msh file (gmsh already initialized)" begin
            mktempdir() do dir
                gmsh.clear()
                gmsh.model.occ.addRectangle(-1.0, -1.0, 0.0, 2.0, 2.0)
                gmsh.model.occ.synchronize()
                gmsh.model.addPhysicalGroup(1,
                    [t for (d, t) in gmsh.model.getEntities(1)], -1, "boundary")
                # gmsh.write only saves elements in physical groups, so the
                # surface needs one too (or the file would hold bare lines)
                gmsh.model.addPhysicalGroup(2,
                    [t for (d, t) in gmsh.model.getEntities(2)], -1, "domain")
                gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
                gmsh.model.mesh.generate(2)
                mshpath = joinpath(dir, "square.msh")
                gmsh.write(mshpath)
                gmsh.clear()
                gm = gmsh_import(mshpath)   # meshed file: opened, not re-generated
                @test typeof(gm.geometry.discretization) <: FEM2D_P1
                @test length(gm.regions["boundary"]) > 8
                V, N = size(gm.geometry.x, 1), size(gm.geometry.x, 2)
                @test length(gm.regions["domain"]) == V * N
            end
        end

        @testset "clockwise surface: negative orientation is flipped" begin
            # geo-kernel curve loop wound clockwise -> surface normal -z ->
            # negatively-oriented quads; the import must flip them and still
            # reproduce linears exactly.
            gmsh.clear()
            p1 = gmsh.model.geo.addPoint(-1.0, -1.0, 0.0)
            p2 = gmsh.model.geo.addPoint(1.0, -1.0, 0.0)
            p3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0)
            p4 = gmsh.model.geo.addPoint(-1.0, 1.0, 0.0)
            l1 = gmsh.model.geo.addLine(p1, p4)
            l2 = gmsh.model.geo.addLine(p4, p3)
            l3 = gmsh.model.geo.addLine(p3, p2)
            l4 = gmsh.model.geo.addLine(p2, p1)
            loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
            surf = gmsh.model.geo.addPlaneSurface([loop])
            gmsh.model.geo.synchronize()
            for l in (l1, l2, l3, l4)
                gmsh.model.mesh.setTransfiniteCurve(l, 5)
            end
            gmsh.model.mesh.setTransfiniteSurface(surf)
            gmsh.model.mesh.setRecombine(2, surf)
            gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4], -1, "boundary")
            gmsh.model.mesh.generate(2)
            gm = gmsh_import()
            @test typeof(gm.geometry.discretization) <: TensorFEM
            @test abs(sum(gm.geometry.w) - 4.0) < 1e-9   # positive measure
            @test _lin_gap(gm.geometry, gm.regions["boundary"]) < 1e-6
        end

        @testset "unsupported elements are rejected with hints" begin
            gmsh.clear()
            gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(3)   # tetrahedra
            @test_throws ArgumentError gmsh_import()
            err = try gmsh_import(); "" catch e; sprint(showerror, e) end
            @test occursin("SubdivisionAlgorithm", err)

            # prisms (recombined extrusion of a triangle mesh): generic rejection
            gmsh.clear()
            r = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 1.0, 1.0)
            gmsh.model.occ.extrude([(2, r)], 0.0, 0.0, 1.0, [1], [], true)
            gmsh.model.occ.synchronize()
            gmsh.option.setNumber("Mesh.MeshSizeMax", 0.6)
            gmsh.model.mesh.generate(3)
            @test_throws ArgumentError gmsh_import()

            # a mesh with only line elements, and no mesh at all
            gmsh.clear()
            gmsh.model.occ.addRectangle(-1.0, -1.0, 0.0, 2.0, 2.0)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(1)
            @test_throws ArgumentError gmsh_import()
            gmsh.clear()
            @test_throws ArgumentError gmsh_import()
        end
    finally
        gmsh.finalize()
    end
end

@testset "Gmsh import from a .geo file (gmsh not initialized)" begin
    # gmsh was finalized above, so gmsh_import(path) owns the
    # initialize/finalize lifecycle here, and a .geo file forces the
    # mesh-generation branch.
    @test_throws ArgumentError gmsh_import("no_such_file_xyz.msh")
    mktempdir() do dir
        geopath = joinpath(dir, "square.geo")
        write(geopath, """
            Point(1) = {-1, -1, 0, 0.6};
            Point(2) = {1, -1, 0, 0.6};
            Point(3) = {1, 1, 0, 0.6};
            Point(4) = {-1, 1, 0, 0.6};
            Line(1) = {1, 2};
            Line(2) = {2, 3};
            Line(3) = {3, 4};
            Line(4) = {4, 1};
            Curve Loop(5) = {1, 2, 3, 4};
            Plane Surface(6) = {5};
            Physical Curve("boundary") = {1, 2, 3, 4};
            Physical Surface("domain") = {6};
            """)
        gm = gmsh_import(geopath)
        @test typeof(gm.geometry.discretization) <: FEM2D_P1
        V, N = size(gm.geometry.x, 1), size(gm.geometry.x, 2)
        @test length(gm.regions["boundary"]) > 4
        # the whole-domain group contains every (vertex, element) pair
        @test sort(gm.regions["domain"]) == sort([(v, e) for e in 1:N for v in 1:V])
    end
end
