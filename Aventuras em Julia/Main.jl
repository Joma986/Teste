include("rbm.jl")
include("contrastive_divergence.jl")
include("Update_RBM.jl")
include("treino.jl")
include("reconstruct.jl")
using Kronecker
using Pkg
using MLDatasets
Pkg.add("Plots")
Pkg.instantiate()
Pkg.update()
using Images


train_x, train_y = MNIST.traindata()


x_bin = [vec(round.(Int, train_x[:,:,i])) for i in 1:60000]
rbm = RBM(784,500)

loss = train!(rbm, 100, x_bin[1:10000], 3)

x_test,y_test = MNIST.testdata()

x_bin_test = [vec(round.(Int, x_test[:,:,i])) for i in 1:10000]
v_reconstructed = reconstruct(rbm,x_bin_test[5000],1)


original_vec = x_test[:, :, 5000] 

original_vec = vec(original_vec) 

original_img = reshape(original_vec, 28, 28)

img = Gray.(original_img)

reconstructed_img = 1.0 .* reshape(v_reconstructed, (28,28))
Gray.(reconstructed_img)


using Plots
using Colors

using Plots
using Colors

# Número de imagens que você quer plotar
num_images = 60

# Criando um layout para 2 colunas e 'num_images' linhas
n_rows = num_images
layout = @layout [grid(n_rows, 2)]

# Inicializando uma lista para armazenar os plots
plots = []

for i in 1:num_images
    # Seleciona as imagens originais e reconstruídas para o índice i
    img = Gray.(reshape(vec(x_test[:, :, i]), 28, 28))
    reconstructed_img = Gray.(reshape(reconstruct(rbm, x_bin_test[i], 1), 28, 28))
    
    # Cria os plots
    p1 = heatmap(img, title="Original $i", color=:grays, axis=false)
    p2 = heatmap(reconstructed_img, title="Reconstructed $i", color=:grays, axis=false)
    
    # Adiciona os plots à lista
    push!(plots, p1)
    push!(plots, p2)
end

# Combine os plots lado a lado em um layout de grid
plot(plots..., layout=layout, size=(800, 400*n_rows))


cut = 784
corrupted_vec = vcat(original_vec[1:cut], zeros(784-cut))
corrupted_img = 1.0 .* reshape(corrupted_vec, (28,28))
colorview(Gray, corrupted_img)

plot()