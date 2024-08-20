include("rbm.jl")
include("contrastive_divergence.jl")
include("Update_RBM.jl")
include("treino.jl")
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

train!(rbm, 10, x_bin, 1)

