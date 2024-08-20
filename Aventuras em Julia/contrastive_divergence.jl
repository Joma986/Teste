include("rbm.jl")

function contrastive_divergence(rbm::RBM1,X,step::Int)
    v_data = nothing
    h_data = nothing
    v_model = nothing
    h_model = nothing
    for x in X
        # Fase Positiva
        v_data = x
        h_data = h_dado_v(rbm,v_data)
        h_data = sample_bernoulli(h_data)
        
        #Fase Negativa
        v_model = gibbs(rbm,v_data,step)
        v_model = sample_bernoulli(v_model)
        h_model = h_dado_v(rbm,v_model) 
        h_model = sample_bernoulli(h_model)
    end
    return (v_data, h_data, v_model, h_model)
end

function h_dado_v(rbm::RBM1, v_data)
    return sigmoid.(rbm.b .+ rbm.W' * v_data)
end

function v_dado_h(rbm::RBM1, h_data)
    return sigmoid.(rbm.a .+ rbm.W * h_data)
end


function sigmoid(x)
    return 1 / (1 + exp(-x))
end

function sample_bernoulli(probabilities)
    return rand(length(probabilities)) .< probabilities
end

function gibbs(rbm::RBM1,v_data,n_inter::Int)
    v_model = v_data

    for _ in 1:n_inter
        h_prob = h_dado_v(rbm,v_model)
        h_model = sample_bernoulli(h_prob)

        v_prob = v_dado_h(rbm,h_model)
        v_model = sample_bernoulli(v_prob)
    end
    return v_model
end