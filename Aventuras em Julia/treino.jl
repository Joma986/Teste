include("contrastive_divergence.jl")
include("Update_RBM.jl")
using Kronecker
function train!(rbm::RBM1, epochs::Int64, X, step::Int,learning_rate = 0.1)
    loss_epoch = []
    for i in 1:epochs
        loss = 0
        v_data, h_data, v_model, h_model, loss = contrastive_divergence(rbm::RBM1,X,step::Int,learning_rate)
        println("Erro da Epoch ",i," ==========> ",loss)
        push!(loss_epoch,loss)
    end
    return loss_epoch
end