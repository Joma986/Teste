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
from sklearn.metrics import precision_recall_curve
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

# Parâmetros
BATCH_SIZE = 32
VISIBLE_UNITS = 43
HIDDEN_UNITS = 21
CD_K = 1
EPOCHS = 20
LEARNING_RATE = 1e-2
MOMENTUM = 0.7

# Inicialização do dispositivo
CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")

# Carregar e pré-processar dados
df = pd.read_csv('treino.csv', sep="|")
teste = pd.read_csv('teste.csv', sep="|")

X, Y = PreProcessamento(df)
# smote = SMOTE(random_state=42)
# X_smote, Y_smote = smote.fit_resample(X, Y)
# tomek = TomekLinks()
# X_smote, Y_smote = tomek.fit_resample(X_smote, Y_smote)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_smote_train, X_smote_test, y_smote_train, y_smote_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

X_smote_train_tensor = torch.tensor(X_smote_train.values, dtype=torch.float32)
X_smote_test_tensor = torch.tensor(X_smote_test.values, dtype=torch.float32)
y_smote_train_tensor = torch.tensor(y_smote_train.values, dtype=torch.float32)
y_smote_test_tensor = torch.tensor(y_smote_test.values, dtype=torch.float32)

train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

smote_train_dataset = CustomDataset(X_smote_train_tensor, y_smote_train_tensor)
smote_train_loader = DataLoader(smote_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
smote_test_dataset = CustomDataset(X_smote_test_tensor, y_smote_test_tensor)
smote_test_loader = DataLoader(smote_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Treinamento do RBM
print('Training RBM...')
rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, learning_rate=LEARNING_RATE, momentum_coefficient=MOMENTUM, use_cuda=CUDA)

train_rbm(smote_train_loader, rbm, EPOCHS, BATCH_SIZE, VISIBLE_UNITS, CUDA)

# Extração de características
print('Extracting features...')
train_features, train_labels = extract_features(smote_train_loader, smote_train_dataset, rbm, HIDDEN_UNITS, BATCH_SIZE, VISIBLE_UNITS, CUDA)
test_features, test_labels = extract_features(smote_test_loader, smote_test_dataset, rbm, HIDDEN_UNITS, BATCH_SIZE, VISIBLE_UNITS, CUDA)

# Classificação com RandomForest
print('Classifying...')
clf = RandomForestClassifier()
clf.fit(train_features, train_labels)
predictions = clf.predict(test_features)
proba_predictions = clf.predict_proba(test_features)[:, 1]
pre = []
rec = []
# Avaliação

aux = (proba_predictions >= 0.17).astype('int64')
Precision = precision_score(y_test, aux)
Recall = recall_score(y_test, aux)
pre.append(Precision)
rec.append(Recall)
roc_auc = roc_auc_score(test_labels, proba_predictions)

print(f'Precision: {Precision:.4f}')
print(f'Recall: {Recall:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')
print(classification_report(test_labels, predictions))

# Plots de avaliação
fpr, tpr, _ = roc_curve(test_labels, proba_predictions)
precision, recall, _ = precision_recall_curve(test_labels, proba_predictions)

plt.figure()
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

plt.figure()
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Pre-processamento e extração de características para novo conjunto de dados
Teste, NumTransacoes = PreProcessamento(teste)
X_novo_tensor = torch.tensor(Teste.values, dtype=torch.float32)
novo_dataset = TestDataset(X_novo_tensor)
novo_loader = DataLoader(novo_dataset, batch_size=BATCH_SIZE, shuffle=False)

novo_features = np.zeros((len(novo_dataset), HIDDEN_UNITS))

for i, batch in enumerate(novo_loader):
    inputs = batch['input']
    inputs = inputs.view(len(inputs), VISIBLE_UNITS)

    if CUDA:
        inputs = inputs.to(device)

    novo_features[i*BATCH_SIZE:i*BATCH_SIZE+len(inputs)] = rbm.sample_hidden(inputs).cpu().numpy()

# Classificação usando o modelo treinado
novo_predictions = clf.predict(novo_features)

print('Predictions for the New Test Dataset:')
print(novo_predictions.sum())

pre = pd.DataFrame(novo_predictions, columns=["Fraude"])
submission = pd.concat([NumTransacoes, pre], axis=1)
submission.to_csv('submission.csv', index=False)
print("Fim")
