include("contrastive_divergence.jl")
include("Update_RBM.jl")
using Kronecker
function train!(rbm::RBM1, epochs::Int64, X, step::Int,learning_rate = 0.1)
    for _ in 1:epochs
        loss = 0
        v_data, h_data, v_model, h_model = contrastive_divergence(rbm::RBM1,X,step::Int)
        update_rbm(rbm,v_data,Int.(h_data),Int.(v_model),Int.(h_model),learning_rate)
        v_prob = v_dado_h(rbm,h_model)
        v_reconstructed = sample_bernoulli(v_prob)
        loss += sum(((v_data .- v_reconstructed)/(length(X))).^2)
        println(loss)
    end
end