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
from rbm import RBM
from PreProcesament import PreProcessamento
from torch.utils.data import Dataset, DataLoader
from imblearn.under_sampling import TomekLinks

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {'input': self.X[idx], 'target': self.y[idx]}
        return sample

class TestDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {'input': self.X[idx]}
        return sample
    

BATCH_SIZE = 32
VISIBLE_UNITS = 43
HIDDEN_UNITS = 21
CD_K = 1
EPOCHS = 10

CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")

df = pd.read_csv('treino.csv', sep="|")
teste = pd.read_csv('teste.csv', sep="|")

X, Y = PreProcessamento(df)
smote = SMOTE(random_state=42)
X, Y = smote.fit_resample(X, Y)
tomek = TomekLinks()

X, Y = tomek.fit_resample(X,Y)
print(f'Nome da GPU: {torch.cuda.get_device_name() if CUDA else "CPU"}')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print('Training RBM...')

rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, use_cuda=CUDA)

for epoch in range(EPOCHS):
    epoch_error = 0.0
    for batch in train_loader:
        inputs, _ = batch['input'], batch['target']
        inputs = inputs.view(len(inputs), VISIBLE_UNITS)  # flatten input data

        if CUDA:
            inputs = inputs.to(device)

        batch_error = rbm.contrastive_divergence(inputs)
        epoch_error += batch_error

    print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error.mean()))


########## EXTRACT FEATURES ##########
print('Extracting features...')

train_features = np.zeros((len(train_dataset), HIDDEN_UNITS))
train_labels = np.zeros(len(train_dataset))
test_features = np.zeros((len(test_dataset), HIDDEN_UNITS))
test_labels = np.zeros(len(test_dataset))

for i, batch in enumerate(train_loader):
    inputs, labels = batch['input'], batch['target']
    inputs = inputs.view(len(inputs), VISIBLE_UNITS)  # flatten input data

    if CUDA:
        inputs = inputs.to(device)

    train_features[i*BATCH_SIZE:i*BATCH_SIZE+len(inputs)] = rbm.sample_hidden(inputs).cpu().numpy()
    train_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(inputs)] = labels.numpy()

for i, batch in enumerate(test_loader):
    inputs, labels = batch['input'], batch['target']
    inputs = inputs.view(len(inputs), VISIBLE_UNITS)  # flatten input data

    if CUDA:
        inputs = inputs.to(device)

    test_features[i*BATCH_SIZE:i*BATCH_SIZE+len(inputs)] = rbm.sample_hidden(inputs).cpu().numpy()
    test_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(inputs)] = labels.numpy()


########## CLASSIFICATION ##########
print('Classifying...')

clf = RandomForestClassifier()
clf.fit(train_features, train_labels)
predictions = clf.predict(test_features)
Precision = precision_score(y_test, predictions)
Recall = recall_score(y_test, predictions)
print('Result: %d/%d' % (sum(predictions == test_labels), test_labels.shape[0]))
print(f'Precision: {Precision}')
print(f'Recall: {Recall}')
print(classification_report(test_labels, predictions))

Teste,NumTransacoes = PreProcessamento(teste)

X_novo_tensor = torch.tensor(Teste.values, dtype=torch.float32)
novo_dataset = TestDataset(X_novo_tensor)
novo_loader = DataLoader(novo_dataset, batch_size=BATCH_SIZE, shuffle=False)

novo_features = np.zeros((len(novo_dataset), HIDDEN_UNITS))

for i, batch in enumerate(novo_loader):
    inputs = batch['input']
    inputs = inputs.view(len(inputs), VISIBLE_UNITS)  # flatten input data

    if CUDA:
        inputs = inputs.to(device)

    novo_features[i*BATCH_SIZE:i*BATCH_SIZE+len(inputs)] = rbm.sample_hidden(inputs).cpu().numpy()

# Classificar usando o modelo treinado
predictions = clf.predict_proba(novo_features) # TESTAR PREDICT PROBABILISTICO
print('Predictions for the New Test Dataset:')
print(predictions.sum())

pre = pd.DataFrame(predictions,columns=["Fraude"])

submission = pd.concat([NumTransacoes,pre],axis = 1)

submission.to_csv('submission.csv',index=False)

print("Fim")
