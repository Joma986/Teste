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

loss = train!(rbm, 50, x_bin[1:5000], 3)

x_test,y_test = MNIST.testdata()

x_bin_test = [vec(round.(Int, x_test[:,:,i])) for i in 1:10000]
v_reconstructed = reconstruct(rbm,x_bin_test[2],1)


original_vec = x_test[:, :, 1] 

original_vec = vec(original_vec) 

original_img = reshape(original_vec, 28, 28)

img = Gray.(original_img)

reconstructed_img = 1.0 .* reshape(v_reconstructed, (28,28))
Gray.(reconstructed_img)