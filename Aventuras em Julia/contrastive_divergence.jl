function contrastive_divergence(rbm::RBM,X,step::Int,learning_rate::Float64=0.1)
    loss_epoch = 0
    for x in X
        # Fase Positiva
        v_data = x
        h_data = h_dado_p(rbm,v_data)
        #Fase Negativa
        v_model = gibbs(rbm,v_data,step)
        h_model = h_dado_p(rbm,v_sample) 
    end
end

function h_dado_v(rbm::RBM, v_data)
    return sigmoid.(rbm.b .+ rbm.W' * v_data)
end

function h_dado_v(rbm::RBM, v_data)
    return sigmoid.(rbm.b .+ rbm.W' * v_data)
end


function sigmoid(x)
    return 1 / (1 + exp(-x))
end

function sample_bernoulli(probabilities)
    return rand(length(probabilities)) .< probabilities
end

function gibbs(rbm::RBM,v_data,n_inter::Int)
    v_model = v_data

    for _ in 1:inter
        h_prob = h_dado_v(rbm,v_model)
        h_model = sample_bernoulli(h_prob)

        v_prob = v_dado_h(rbm,h_model)
        v_model = sample_bernoulli(v_prob)
    end
    return v_model
end