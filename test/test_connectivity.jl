using Test
using MultiGridBarrier
using MultiGridBarrier: _xflat, _dedupe, _corner_labels_from_t, _tf_corner_local, find_boundary

# Two label vectors induce the same partition (gluing) of their indices iff the
# pairwise map is a bijection — i.e. they agree up to a global relabel.
function same_partition(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    length(a) == length(b) || return false
    f = Dict{Int,Int}(); g = Dict{Int,Int}()
    for (x, y) in zip(a, b)
        haskey(f, x) ? (f[x] == y || return false) : (f[x] = y)
        haskey(g, y) ? (g[y] == x || return false) : (g[y] = x)
    end
    return true
end

# Corner connectivity (2^d, N) recovered from a geom's cached full connectivity.
function corner_conn(geom, k::Int, ::Val{d}) where {d}
    nc = 1 << d
    cornerlocal = ntuple(c -> _tf_corner_local(c, k + 1, Val(d)), nc)
    node_map, _ = _corner_labels_from_t(geom.t, cornerlocal)
    return reshape(node_map, nc, size(geom.t, 2))
end

@testset "tensor_dofmap reproduces dedup gluing (oracle)" begin
    # 1D (k=3): only endpoints are shared; interior nodes are cell-interior.
    g1 = fem1d(nodes = collect(range(-1.0, 1.0, length = 5)), k = 3)
    @test same_partition(vec(g1.t),
                         vec(tensor_dofmap(corner_conn(g1, 3, Val(1)), 3, Val(1))))

    # 2D quads, multi-element via subdivision, k = 1,2,3 (edge orientation for k=3).
    for k in (1, 2, 3)
        g = subdivide(fem2d(k = k), 3)            # many shared edges, both orientations
        @test same_partition(vec(g.t),
                             vec(tensor_dofmap(corner_conn(g, k, Val(2)), k, Val(2))))
    end

    # 3D hexes, multi-element via subdivision, k = 2,3 (shared face-interior
    # grids at k=3, in the subdivision's axis-aligned orientation).
    for k in (2, 3)
        g3 = subdivide(fem3d(k = k), 2)
        @test same_partition(vec(g3.t),
                             vec(tensor_dofmap(corner_conn(g3, k, Val(3)), k, Val(3))))
    end
end

@testset "tensor_dofmap: 3D shared face-interior grids (k≥3), transposed" begin
    # Two hexes sharing the x=1 face; hex B is built with its shared-face axes
    # SWAPPED (local y' -> global z, z' -> global y), so gluing the (k-1)² face
    # interior requires matching the quad face's 8 symmetries — not just a trivial
    # axis-aligned overlap. The dedup geometry is the ground-truth partition.
    for k in (2, 3, 4)
        Kc = Array{Float64,3}(undef, 8, 2, 3)
        for c in 1:8
            b = c - 1; x = b & 1; y = (b >> 1) & 1; z = (b >> 2) & 1
            Kc[c, 1, :] = [x, y, z]           # hex A: [0,1]³, standard orientation
            Kc[c, 2, :] = [1 + x, z, y]       # hex B: [1,2]×[0,1]², face axes swapped
        end
        g = fem3d(k = k, K = Kc)              # coordinate-dedup ground truth
        @test same_partition(vec(g.t),
                             vec(tensor_dofmap(corner_conn(g, k, Val(3)), k, Val(3))))
        # exact node count: two hexes minus the shared (k+1)² face
        @test maximum(tensor_dofmap(corner_conn(g, k, Val(3)), k, Val(3))) ==
              2 * (k + 1)^3 - (k + 1)^2
    end
end

# Two Q2 quads sharing the edge x=1: [0,1]² and [1,2]×[0,1] (corner shorthand).
const K2 = reshape(Float64[0 0; 1 0; 0 1; 1 1;        # element 1 corners (bit order)
                           1 0; 2 0; 1 1; 2 1], 4, 2, 2)
const TC_GLUED = [1 2; 2 5; 3 4; 4 6]                 # shares (1,0)=2 and (1,1)=4
const TC_SLIT  = [1 7; 2 5; 3 8; 4 6]                 # element 2's left corners are distinct

@testset "native (t,x) constructor: supplied connectivity is authoritative" begin
    t_glued = tensor_dofmap(TC_GLUED, 2, Val(2))
    @test maximum(t_glued) == 15                      # 2*9 − 3 shared edge nodes

    geom = fem2d(k = 2, K = K2, t = t_glued)
    @test geom.t == t_glued                           # stored verbatim, not re-deduped

    # Embedded glued mesh: supplied t matches what coordinate dedup would infer.
    @test same_partition(vec(geom.t), vec(fem2d(k = 2, K = K2).t))
end

@testset "slit domain: coincident nodes stay distinct" begin
    t_glued = tensor_dofmap(TC_GLUED, 2, Val(2))
    t_slit  = tensor_dofmap(TC_SLIT,  2, Val(2))
    @test maximum(t_slit) == 18                       # 2*9, nothing shared on the seam

    g_glued = fem2d(k = 2, K = K2, t = t_glued)
    g_slit  = fem2d(k = 2, K = K2, t = t_slit)

    # The default (dedup) constructor cannot tell the slit from the glued mesh —
    # the coordinates are identical — but the topological t can.
    @test same_partition(vec(fem2d(k = 2, K = K2).t), vec(g_glued.t))
    @test !same_partition(vec(g_slit.t), vec(g_glued.t))

    # The seam is interior when glued, boundary (used once per side) when slit.
    @test length(find_boundary(g_slit)) > length(find_boundary(g_glued))

    # And the whole solver runs end-to-end on the slit mesh.
    sol = mgb_solve(assemble(amg(g_slit); p = 1.5); verbose = false, tol = 1e-6)
    @test all(isfinite, sol.z)
end
