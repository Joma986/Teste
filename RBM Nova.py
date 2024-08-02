import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from category_encoders import BinaryEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import PrecisionRecallDisplay
import xgboost as xgb
from xgboost import XGBClassifier, QuantileDMatrix
from sklearn.metrics import roc_curve, roc_auc_score
from catboost import CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
import torch
import torch.nn as nn
import torch.optim as optim
from PreProcesament import PreProcessamento
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PreProcesament import PreProcessamento

# Definição da classe RBM
class RBM(nn.Module):
    def __init__(self, n_vis, n_hin, k):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.rand(n_hin, n_vis) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))
        self.k = k

    def sample_from_p(self, p):
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()).to(p.device))))

    def v_to_h(self, v):
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v

    def forward(self, v):
        pre_h1, h1 = self.v_to_h(v)
        h_ = h1
        for _ in range(self.k):
            pre_v_, v_ = self.h_to_v(h_)
            pre_h_, h_ = self.v_to_h(v_)
        return v, v_

    def free_energy(self, v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()
    
    def extract_features(self, v):
        p_h, sample_h = self.v_to_h(v)
        return sample_h 

# Dataset customizado
class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        sample = {'input': self.X[idx], 'target': self.Y[idx]}
        return sample

# Carregando dados
df = pd.read_csv('treino.csv', sep="|")
teste = pd.read_csv('teste.csv', sep="|")

X, Y = PreProcessamento(df)

# Divisão dos dados em treino e teste.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Instanciando as RBMs
rbm = RBM(k = 1, n_vis = 43, n_hin = 21)
rbm2 = RBM(k = 1, n_vis = 21, n_hin = 21)

train_op = optim.SGD(rbm.parameters(), 0.001)
train_op2 = optim.SGD(rbm2.parameters(), 0.001)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Criando os Dataloaders 
train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Dispositivo utilizado para Treinamento: {torch.cuda.get_device_name()}')
rbm.to(device)
rbm2.to(device)

print('\nTreinando RBM...')
# Loop de treino da primeira RBM
epochs = 4
for epoch in range(epochs):
    loss_ = []
    for batch in train_loader:
        inputs, _ = batch['input'], batch['target']
        inputs = inputs.to(device).float()
        sample_inputs = inputs.bernoulli().to(device)

        v, v1 = rbm(sample_inputs)
        loss = rbm.free_energy(v) - rbm.free_energy(v1)
        loss = loss.to(device)
        loss_.append(loss.data.cpu())
        train_op.zero_grad()
        loss.backward()
        train_op.step()
    print(f'Época: {epoch + 1} | Loss ==> {np.mean(loss_)}')

print('\nExtraindo Features....')

# Extraindo features para os dados de treino
with torch.no_grad():
    train_features = []
    for batch in train_loader:
        inputs = batch['input'].to(device).float()
        hidden_features = rbm.extract_features(inputs)
        train_features.append(hidden_features.cpu().numpy())

with torch.no_grad():
    test_features = []
    for batch in test_loader:
        inputs = batch['input'].to(device).float()
        hidden_features = rbm.extract_features(inputs)
        test_features.append(hidden_features.cpu().numpy())

# Convertendo features de treino para tensor e criando novo DataLoader
train_features = np.concatenate(train_features, axis=0)
train_features_tensor = torch.tensor(train_features, dtype=torch.float32).to(device)
train_dataset = CustomDataset(train_features_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)


test_features = np.concatenate(test_features, axis=0)
test_features_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)
test_dataset = CustomDataset(test_features_tensor, y_train_tensor)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
print("Treinando a Segunda RBM....")

# Loop de treino da segunda RBM
for epoch in range(epochs):
    loss_ = []
    for batch in train_loader:
        inputs, _ = batch['input'], batch['target']
        inputs = inputs.to(device).float()
        sample_inputs = inputs.bernoulli().to(device)

        v, v1 = rbm2(sample_inputs)
        loss = rbm2.free_energy(v) - rbm2.free_energy(v1)
        loss = loss.to(device)
        loss_.append(loss.data.cpu())
        train_op2.zero_grad()
        loss.backward()
        train_op2.step()
    print(f'Época: {epoch + 1} | Loss ==> {np.mean(loss_)}')


with torch.no_grad():
    train_features_ = []
    for batch in train_loader:
        inputs = batch['input'].to(device).float()
        hidden_features = rbm2.extract_features(inputs)
        train_features_.append(hidden_features.cpu().numpy())

with torch.no_grad():
    test_features_ = []
    for batch in test_loader:
        inputs = batch['input'].to(device).float()
        hidden_features = rbm2.extract_features(inputs)
        test_features_.append(hidden_features.cpu().numpy())

print('Montando o Random Forest...')

# Convertendo y_train_tensor e y_test_tensor de volta para numpy
y_train = y_train_tensor.cpu().numpy()
y_test = y_test_tensor.cpu().numpy()

print('Treinando...')
clf = LogisticRegression()
clf.fit(train_features, y_train)

print('Testando...')
predict_proba = clf.predict_proba(test_features)[:,1]
pre = []
rec = []
for threshold in np.linspace(0,1,100):
    aux = (predict_proba >= threshold).astype('int64')
    precision = precision_score(y_pred=aux, y_true=y_test)
    recall = recall_score(y_pred=aux, y_true=y_test)
    pre.append(precision)
    rec.append(recall)

plt.plot(np.linspace(0,1,100), pre, label='Precision')
plt.plot(np.linspace(0,1,100), rec, label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.show()
