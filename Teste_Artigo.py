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
from rbm import RBM,CustomDataset,TestDataset,train_rbm,extract_features
from PreProcesament import PreProcessamento
from torch.utils.data import Dataset, DataLoader
from imblearn.under_sampling import TomekLinks

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

X_Fraudes = X[Y == 1]
X_NaoFraudes = X[Y == 0]

X_Fraudes_tensor = torch.tensor(X_Fraudes.values, dtype=torch.float32)
X_NaoFraudes_tensor = torch.tensor(X_NaoFraudes.values, dtype=torch.float32)

data_set_Fraudes = CustomDatasetNS(X_Fraudes)
data_set_NaoFraudes = CustomDatasetNS(X_NaoFraudes)

print('Training RBM...')
rbm_Fraudes = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, learning_rate=LEARNING_RATE, momentum_coefficient=MOMENTUM, use_cuda=CUDA)
rbm_NaoFraudes = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, learning_rate=LEARNING_RATE, momentum_coefficient=MOMENTUM, use_cuda=CUDA)
