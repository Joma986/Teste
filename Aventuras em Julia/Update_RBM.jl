using Kronecker
include("rbm.jl")
function update_rbm(
                    rbm::RBM1, 
                    v_data::Vector{Float64}, 
                    h_data::Vector{Float64}, 
                    v_model::Vector{Float64}, 
                    h_model::Vector{Float64}, 
                    learning_rate::Float64
                    )

    rbm.W += learning_rate.*(v_data * h_data' .- v_model * h_model')
    rbm.a += learning_rate.*(v_data - v_model)
    rbm.b += learning_rate.*(h_data - h_model)
end



