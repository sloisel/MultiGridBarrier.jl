
function chebyshev_nodes(k::Int)
    # Chebyshev-Lobatto nodes on [-1, 1]
    # k is the degree, so there are k+1 nodes.
    return [-cos(i * π / k) for i in 0:k]
end

function tensor_product_nodes(nodes_x, nodes_y, nodes_z)
    nx = length(nodes_x)
    ny = length(nodes_y)
    nz = length(nodes_z)
    n = nx * ny * nz
    x = zeros(n, 3)
    idx = 1
    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                x[idx, 1] = nodes_x[i]
                x[idx, 2] = nodes_y[j]
                x[idx, 3] = nodes_z[k]
                idx += 1
            end
        end
    end
    return x
end

function clenshaw_curtis_weights(k::Int)
    # Clenshaw-Curtis weights for k+1 nodes (degree k)
    # Nodes are x_j = -cos(j * pi / k), j = 0..k
    
    if k == 0
        return [2.0]
    end
    
    N = k
    theta = [i * π / N for i in 0:N]
    w = zeros(N + 1)
    
    for i in 0:N
        val = 1.0
        for j in 1:div(N, 2)
            if 2*j == N
                val += 1.0 / (1.0 - 4.0*j^2) * cos(2.0*j*theta[i+1])
            else
                val += 2.0 / (1.0 - 4.0*j^2) * cos(2.0*j*theta[i+1])
            end
        end
        
        if i == 0 || i == N
            w[i+1] = val / N
        else
            w[i+1] = 2.0 * val / N
        end
    end
    
    return w
end

"""
    cube_mesh(k::Int)

Generate a single hexahedral element (cube [-1,1]^3) with Q_k discretization.
Returns the vertices x (flattened) and weights w (reference weights).
"""
function cube_mesh(k::Int)
    nodes = chebyshev_nodes(k)
    weights_1d = clenshaw_curtis_weights(k)
    
    x = tensor_product_nodes(nodes, nodes, nodes)
    
    # Tensor product weights
    # w[idx] = wx[i] * wy[j] * wz[k]
    # Ordering matches tensor_product_nodes: z slow, y medium, x fast
    
    nx = ny = nz = k+1
    n = nx * ny * nz
    w = zeros(n)
    
    idx = 1
    for k_idx in 1:nz
        for j_idx in 1:ny
            for i_idx in 1:nx
                w[idx] = weights_1d[i_idx] * weights_1d[j_idx] * weights_1d[k_idx]
                idx += 1
            end
        end
    end
    
    return x, w
end

"""
    deduplicate_vertices(x::Matrix{T}; tol=1e-12)

Identify unique vertices in the mesh `x` (Nx3).
Returns:
- `unique_x`: Matrix of unique vertices.
- `node_map`: Vector of indices mapping original rows to unique rows.
- `counts`: Multiplicity of each unique vertex.
"""
function deduplicate_vertices(x::Matrix{T}; tol=1e-12) where T
    n = size(x, 1)
    perm = sortperm(eachslice(x, dims=1)) # Lexicographical sort
    
    unique_idxs = Int[]
    node_map = zeros(Int, n)
    
    if n == 0
        return zeros(T, 0, 3), node_map, Int[]
    end
    
    # First element is always unique
    push!(unique_idxs, perm[1])
    node_map[perm[1]] = 1
    
    current_unique_idx = 1
    
    for i in 2:n
        idx = perm[i]
        prev_idx = perm[i-1]
        
        # Check distance
        dist = norm(x[idx, :] - x[prev_idx, :])
        
        if dist > tol
            push!(unique_idxs, idx)
            current_unique_idx += 1
        end
        node_map[idx] = current_unique_idx
    end
    
    unique_x = x[unique_idxs, :]
    
    # Count multiplicity
    counts = zeros(Int, length(unique_idxs))
    for i in 1:n
        counts[node_map[i]] += 1
    end
    
    return unique_x, node_map, counts
end

function lagrange_basis(nodes, x_val)
    k = length(nodes) - 1
    vals = zeros(eltype(nodes), k+1)
    for i in 1:k+1
        num = 1.0
        den = 1.0
        for j in 1:k+1
            if i != j
                num *= (x_val - nodes[j])
                den *= (nodes[i] - nodes[j])
            end
        end
        vals[i] = num / den
    end
    return vals
end

"""
    interpolate_element(parent_nodes, parent_x, child_xi)

Interpolate position in a parent element defined by `parent_nodes` (reference) and `parent_x` (physical)
at reference coordinates `child_xi`.
Assumes tensor product structure.
"""
function interpolate_element(parent_nodes_1d, parent_x_element, child_xi)
    # parent_x_element is (k+1)^3 x 3
    # parent_nodes_1d is vector of length k+1
    # child_xi is (3,) vector
    
    k = length(parent_nodes_1d) - 1
    Lx = lagrange_basis(parent_nodes_1d, child_xi[1])
    Ly = lagrange_basis(parent_nodes_1d, child_xi[2])
    Lz = lagrange_basis(parent_nodes_1d, child_xi[3])
    
    pos = zeros(3)
    idx = 1
    for iz in 1:k+1
        for iy in 1:k+1
            for ix in 1:k+1
                w = Lx[ix] * Ly[iy] * Lz[iz]
                pos += w * parent_x_element[idx, :]
                idx += 1
            end
        end
    end
    return pos
end

function refine_mesh(x_coarse::Matrix{T}, k::Int) where T
    # Split each element into 8 children
    n_nodes_per_elem = (k+1)^3
    n_elems = div(size(x_coarse, 1), n_nodes_per_elem)
    
    nodes_1d = chebyshev_nodes(k)
    
    # Child domains in reference space [-1, 1]
    # 8 children: combinations of [-1, 0] and [0, 1]
    # Map: [-1, 1] -> [-1, 0] is t -> 0.5(t-1)
    # Map: [-1, 1] -> [0, 1]  is t -> 0.5(t+1)
    
    shifts = [-0.5, 0.5]
    scales = 0.5
    
    # Precompute child node locations in parent reference frame
    child_ref_nodes = zeros(8, n_nodes_per_elem, 3)
    
    child_idx = 1
    for sz in shifts
        for sy in shifts
            for sx in shifts
                # For this child, map the standard nodes
                for (i, node_idx) in enumerate(Iterators.product(nodes_1d, nodes_1d, nodes_1d))
                    # node_idx is (x, y, z) from chebyshev_nodes
                    # Map to parent ref frame
                    child_ref_nodes[child_idx, i, 1] = node_idx[1] * scales + sx
                    child_ref_nodes[child_idx, i, 2] = node_idx[2] * scales + sy
                    child_ref_nodes[child_idx, i, 3] = node_idx[3] * scales + sz
                end
                child_idx += 1
            end
        end
    end
    
    x_fine = zeros(T, n_elems * 8 * n_nodes_per_elem, 3)
    
    for e in 1:n_elems
        # Extract parent element vertices
        start_idx = (e-1) * n_nodes_per_elem + 1
        end_idx = e * n_nodes_per_elem
        parent_x = x_coarse[start_idx:end_idx, :]
        
        # For each child
        for c in 1:8
            # For each node in child
            for i in 1:n_nodes_per_elem
                xi = child_ref_nodes[c, i, :]
                pos = interpolate_element(nodes_1d, parent_x, xi)
                
                fine_idx = (e-1)*8*n_nodes_per_elem + (c-1)*n_nodes_per_elem + i
                x_fine[fine_idx, :] = pos
            end
        end
    end
    
    return x_fine
end

"""
    get_boundary_nodes(x::Matrix{T}, k::Int; tol=1e-12)

Identify indices of unique nodes that lie on the boundary of the domain.
Returns:
- `boundary_indices`: Vector of indices into `unique_x` (from deduplicate_vertices).
- `unique_x`: The unique vertices.
- `node_map`: Map from broken to unique.
"""
function get_boundary_nodes(x::Matrix{T}, k::Int; tol=1e-12) where T
    unique_x, node_map, counts = deduplicate_vertices(x; tol=tol)
    
    n_nodes_per_elem = (k+1)^3
    n_elems = div(size(x, 1), n_nodes_per_elem)
    
    # Define face node indices in local element
    # Nodes are ordered by x, then y, then z (tensor product)
    # indices: 1..(k+1) for each dim
    
    # Helper to get linear index from (ix, iy, iz) 1-based
    idx(ix, iy, iz) = (iz-1)*(k+1)^2 + (iy-1)*(k+1) + ix
    
    faces = Vector{Vector{Int}}()
    
    # x = -1 face (ix=1)
    push!(faces, [idx(1, iy, iz) for iy in 1:k+1, iz in 1:k+1][:])
    # x = 1 face (ix=k+1)
    push!(faces, [idx(k+1, iy, iz) for iy in 1:k+1, iz in 1:k+1][:])
    
    # y = -1 face (iy=1)
    push!(faces, [idx(ix, 1, iz) for ix in 1:k+1, iz in 1:k+1][:])
    # y = 1 face (iy=k+1)
    push!(faces, [idx(ix, k+1, iz) for ix in 1:k+1, iz in 1:k+1][:])
    
    # z = -1 face (iz=1)
    push!(faces, [idx(ix, iy, 1) for ix in 1:k+1, iy in 1:k+1][:])
    # z = 1 face (iz=k+1)
    push!(faces, [idx(ix, iy, k+1) for ix in 1:k+1, iy in 1:k+1][:])
    
    face_counts = Dict{Vector{Int}, Int}()
    
    for e in 1:n_elems
        base_idx = (e-1) * n_nodes_per_elem
        
        for face_local_indices in faces
            # Get unique indices for this face
            face_unique_indices = sort([node_map[base_idx + li] for li in face_local_indices])
            
            if haskey(face_counts, face_unique_indices)
                face_counts[face_unique_indices] += 1
            else
                face_counts[face_unique_indices] = 1
            end
        end
    end
    
    boundary_unique_indices = Set{Int}()
    
    for (face_sig, count) in face_counts
        if count == 1
            # Boundary face
            for ui in face_sig
                push!(boundary_unique_indices, ui)
            end
        end
    end
    
    return sort(collect(boundary_unique_indices)), unique_x, node_map
end



"""
    promote_to_Qk(K_q1::Matrix{T}, k::Int)

Promote a Q1 mesh `K_q1` (N x 3, where N is a multiple of 8) to a Q_k mesh.
Each group of 8 rows in `K_q1` represents one linear hexahedron.
Returns a new mesh (M x 3) where each element is replaced by (k+1)^3 nodes
located at the mapped Chebyshev-Lobatto points.
"""
function promote_to_Qk(K_q1::Matrix{T}, k::Int) where T
    n_verts = size(K_q1, 1)
    n_elems = div(n_verts, 8)
    
    if n_verts % 8 != 0
        error("Input mesh size must be a multiple of 8 (8 vertices per Q1 element)")
    end
    
    n_nodes_per_elem = (k+1)^3
    x_Qk = zeros(T, n_elems * n_nodes_per_elem, 3)
    
    # Reference nodes for Qk element [-1, 1]^3
    nodes_1d = chebyshev_nodes(k)
    
    # Reference nodes for Q1 element (corners of [-1, 1]^3)
    # Ordering must match the input K_q1 ordering.
    # We assume standard tensor product ordering for Q1:
    # (-1,-1,-1), (1,-1,-1), (-1,1,-1), (1,1,-1), (-1,-1,1), ...
    # This corresponds to tensor_product_nodes([-1,1], [-1,1], [-1,1])
    q1_ref_nodes = tensor_product_nodes([-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0])
    
    # Precompute Qk nodes in reference space
    qk_ref_nodes = tensor_product_nodes(nodes_1d, nodes_1d, nodes_1d)
    
    for e in 1:n_elems
        # Get Q1 element vertices
        q1_idx_start = (e-1)*8 + 1
        q1_idx_end = e*8
        q1_verts = K_q1[q1_idx_start:q1_idx_end, :]
        
        # Map each Qk reference node to physical space using Q1 shape functions
        # x(xi) = sum N_i(xi) * x_i
        
        qk_idx_start = (e-1)*n_nodes_per_elem + 1
        
        for i in 1:n_nodes_per_elem
            xi = qk_ref_nodes[i, :]
            
            # Trilinear interpolation
            # We can reuse interpolate_element if we pass the Q1 reference nodes
            # interpolate_element(parent_nodes_1d, parent_x_element, child_xi)
            # But interpolate_element assumes parent is high order too?
            # Actually interpolate_element takes parent_nodes_1d.
            # For Q1, parent_nodes_1d is [-1, 1].
            
            pos = interpolate_element([-1.0, 1.0], q1_verts, xi)
            
            x_Qk[qk_idx_start + i - 1, :] = pos
        end
    end
    
    return x_Qk
end
