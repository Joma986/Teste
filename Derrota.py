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
from Binarizator import converte_binario

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

def train_rbm(train_loader, rbm, epochs, batch_size, visible_units, cuda, Threshold = 0):
    error = []
    i = 0
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
        error.append(epoch_error.mean())
        MenorErro = min(error)
        if i == 0:
            MelhoresPesos = rbm.weights.clone()
            MelhoresViesesVisiveis = rbm.visible_bias.clone()
            MelhoresViesesOcultos = rbm.hidden_bias.clone()
        if error[epoch] > MenorErro and Threshold != 0:
            i += 1
            print(f'EarlyStopping {i}/{Threshold}')
            if i == Threshold:
                rbm.weights = MelhoresPesos
                rbm.visible_bias = MelhoresViesesVisiveis
                rbm.hidden_bias = MelhoresViesesOcultos
                print(f'EarlyStopping atuou. Pesos salvos da Época {epoch - i}') 
                break
        else:
            i = 0

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
BATCH_SIZE = 256
VISIBLE_UNITS = 43
HIDDEN_UNITS = 128
CD_K = 1
EPOCHS = 200
LEARNING_RATE = 5e-3
MOMENTUM = 0.7

# Inicialização do dispositivo
CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")

# Carregar e pré-processar dados
df = pd.read_csv('treino.csv', sep="|")
teste = pd.read_csv('teste.csv', sep="|")

X, Y = PreProcessamento(df)

TESTE = X.select_dtypes('float64')



# X[X.select_dtypes('float64').columns] = X.select_dtypes('float64').applymap(converte_binario)
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


train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Treinamento do RBM
print('Training RBM...')
rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, learning_rate=LEARNING_RATE, momentum_coefficient=MOMENTUM, use_cuda=CUDA)

train_rbm(train_loader, rbm, EPOCHS, BATCH_SIZE, VISIBLE_UNITS, CUDA, Threshold = 5)

# Extração de características
print('Extracting features...')
train_features, train_labels = extract_features(train_loader, train_dataset, rbm, HIDDEN_UNITS, BATCH_SIZE, VISIBLE_UNITS, CUDA)
test_features, test_labels = extract_features(test_loader, test_dataset, rbm, HIDDEN_UNITS, BATCH_SIZE, VISIBLE_UNITS, CUDA)

# Classificação com RandomForest
print('Classifying...')

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

clf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

grid_search.fit(train_features, train_labels)

print(f'Melhores parâmetros: {grid_search.best_params_}')

clf.fit(train_features, train_labels)
predictions = clf.predict(test_features)
proba_predictions = clf.predict_proba(test_features)[:,1]
pre = []
rec = []
# Avaliação
for Threshold in np.linspace(0,1,100):
    aux = (proba_predictions >= Threshold).astype('int64')
    Precision = precision_score(y_test, aux)
    Recall = recall_score(y_test, aux)
    pre.append(Precision)
    rec.append(Recall)
plt.show()
plt.plot(np.linspace(0,1,100),rec)
plt.plot(np.linspace(0,1,100),pre)
plt.show()

print(f'Precision: {Precision:.4f}')
print(f'Recall: {Recall:.4f}')
# print(f'ROC AUC: {roc_auc:.4f}')

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
novo_predictions = clf.predict_proba(novo_features)[:,1]

novo_predictions = (novo_predictions >= 0.5).astype('int64')

print('Predictions for the New Test Dataset:')
print('Fraudes: ',novo_predictions.sum())

pre = pd.DataFrame(novo_predictions, columns=["Fraude"])
submission = pd.concat([NumTransacoes, pre], axis=1)
submission.to_csv('submission.csv', index=False)
print("Fim")
