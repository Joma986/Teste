
function reconstruct(rbm::RBM1,X,step)
    v_data = nothing
    h_data = nothing
    v_model = nothing
    h_model = nothing
    v_reconstructed = []
    # Fase Positiva
    v_data = X
    h_data = h_dado_v(rbm,v_data)
    #h_data = sample_bernoulli(h_data)
    v_prob = v_dado_h(rbm,h_data)

    v_reconstructed = v_prob
    return (v_reconstructed)
end