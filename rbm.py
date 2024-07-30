import torch
from torch.utils.data import Dataset, DataLoader

class RBM():

    def __init__(self, num_visible, num_hidden, k, learning_rate=1e-3, momentum_coefficient=0.5, weight_decay=1e-4, use_cuda=True):
        '''Inicia os parâmetros da classe.'''
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda

        '''Inicia os pesos da RBM'''
        self.weights = torch.randn(num_visible, num_hidden) * 0.01
        self.visible_bias = torch.zeros(num_visible)
        self.hidden_bias = torch.zeros(num_hidden)

        '''Inicia os parametros de momento.'''
        self.weights_momentum = torch.zeros(num_visible, num_hidden)
        self.visible_bias_momentum = torch.zeros(num_visible)
        self.hidden_bias_momentum = torch.zeros(num_hidden)

        if self.use_cuda:
            '''Passa os Tensores para GPU.'''
            self.weights = self.weights.cuda()
            self.visible_bias = self.visible_bias.cuda()
            self.hidden_bias = self.hidden_bias.cuda()

            self.weights_momentum = self.weights_momentum.cuda()
            self.visible_bias_momentum = self.visible_bias_momentum.cuda()
            self.hidden_bias_momentum = self.hidden_bias_momentum.cuda()

    def sample_hidden(self, visible_probabilities):
        '''Método utilizado para calcular as ativações das camadas ocultas através das visiveis.
           Sigmoid aplicada para calcular P(hi = 1|v)'''
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
        hidden_probabilities = self._sigmoid(hidden_activations)
        return hidden_probabilities

    def sample_visible(self, hidden_probabilities):
        '''Método utilizado para calcular as ativações das camadas visiveis através das ocultas.
           Sigmoid aplicada para calcular P(vi = 1|h)'''
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias
        visible_probabilities = self._sigmoid(visible_activations)
        return visible_probabilities

    def contrastive_divergence(self, input_data):
        '''Aplicação da Contrastive Divergence'''
        # Positive phase
        positive_hidden_probabilities = self.sample_hidden(input_data)
        positive_hidden_activations = (positive_hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()
        positive_associations = torch.matmul(input_data.t(), positive_hidden_activations)

        # Negative phase
        hidden_activations = positive_hidden_activations

        for step in range(self.k):
            '''Aplicação da amostragem de Gibs k vezes'''
            visible_probabilities = self.sample_visible(hidden_activations)
            hidden_probabilities = self.sample_hidden(visible_probabilities)
            hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()

        negative_visible_probabilities = visible_probabilities
        negative_hidden_probabilities = hidden_probabilities

        negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)

        # Update parameters
        self.weights_momentum *= self.momentum_coefficient
        self.weights_momentum += (positive_associations - negative_associations)

        self.visible_bias_momentum *= self.momentum_coefficient
        self.visible_bias_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

        self.hidden_bias_momentum *= self.momentum_coefficient
        self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

        batch_size = input_data.size(0)

        self.weights += self.weights_momentum * self.learning_rate / batch_size
        self.visible_bias += self.visible_bias_momentum * self.learning_rate / batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / batch_size

        self.weights -= self.weights * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_probabilities)**2) / batch_size

        return error

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)

        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()

        return random_probabilities
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {'input': self.X[idx], 'target': self.y[idx]}
        return sample
    
class CustomDatasetNS(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {'input': self.X[idx]}
        return sample

class TestDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {'input': self.X[idx]}
        return sample

def train_rbm(train_loader, rbm, epochs, batch_size, visible_units, cuda):
    for epoch in range(epochs):
        epoch_error = 0.0
        for batch in train_loader:
            inputs, _ = batch['input'], batch['target']
            inputs = inputs.view(len(inputs), visible_units)

            if cuda:
                inputs = inputs.to(device)

            batch_error = rbm.contrastive_divergence(inputs)
            epoch_error += batch_error

        print(f'Epoch Error (epoch={epoch}): {epoch_error.mean():.4f}')

def extract_features(loader, dataset, rbm, hidden_units, batch_size, visible_units, cuda):
    features = np.zeros((len(dataset), hidden_units))
    labels = np.zeros(len(dataset))

    for i, batch in enumerate(loader):
        inputs, labels_batch = batch['input'], batch['target']
        inputs = inputs.view(len(inputs), visible_units)

        if cuda:
            inputs = inputs.to(device)

        features[i*batch_size:i*batch_size+len(inputs)] = rbm.sample_hidden(inputs).cpu().numpy()
        labels[i*batch_size:i*batch_size+len(inputs)] = labels_batch.numpy()

    return features, labels