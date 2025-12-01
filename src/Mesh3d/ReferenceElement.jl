struct ReferenceElement{T}
    k::Int
    nodes_1d::Vector{T}
    weights_1d::Vector{T}
    D_1d::Matrix{T}
    I_1d::Diagonal{T, Vector{T}}
    
    # Tensor product operators
    D_xi_local::Matrix{T}
    D_eta_local::Matrix{T}
    D_zeta_local::Matrix{T}
    
    # Tensor product nodes and weights
    nodes_ref::Matrix{T}
    weights_ref::Vector{T}
end

function ReferenceElement(k::Int, ::Type{T}=Float64) where T
    nodes_1d = T.(chebyshev_nodes(k))
    weights_1d = T.(clenshaw_curtis_weights(k))
    D_1d = T.(derivative_matrix_1d(nodes_1d))
    I_1d = Diagonal(ones(T, k+1))
    
    # Tensor product operators
    # D_xi = I (x) I (x) D
    # D_eta = I (x) D (x) I
    # D_zeta = D (x) I (x) I
    
    D_xi_local = kron(I_1d, kron(I_1d, D_1d))
    D_eta_local = kron(I_1d, kron(D_1d, I_1d))
    D_zeta_local = kron(D_1d, kron(I_1d, I_1d))
    
    nodes_ref = T.(tensor_product_nodes(nodes_1d, nodes_1d, nodes_1d))
    
    # Tensor product weights
    nx = ny = nz = k+1
    n = nx * ny * nz
    weights_ref = zeros(T, n)
    
    idx = 1
    for k_idx in 1:nz
        for j_idx in 1:ny
            for i_idx in 1:nx
                weights_ref[idx] = weights_1d[i_idx] * weights_1d[j_idx] * weights_1d[k_idx]
                idx += 1
            end
        end
    end
    
    return ReferenceElement{T}(
        k, nodes_1d, weights_1d, D_1d, I_1d,
        D_xi_local, D_eta_local, D_zeta_local,
        nodes_ref, weights_ref
    )
end
