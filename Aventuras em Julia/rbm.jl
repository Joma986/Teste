mutable struct RBM1
    W::Matrix{Float64}
    a::Vector{Float64}
    b::Vector{Float64}
    n_visible::Int
    n_hidden::Int
end

function RBM(n_visible,n_hidden)
    W = randn(n_visible,n_hidden)
    a = zeros(n_visual)
    b = zeros(n_hidden)
    return RBM1(W, a, b, n_visible,n_hidden)
end



