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

BATCH_SIZE = 64
VISIBLE_UNITS = 43
HIDDEN_UNITS = 21
CD_K = 2
EPOCHS = 10

DATA_FOLDER = 'data/mnist'

CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0

if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)

df = pd.read_csv('treino.csv',sep="|")
teste = pd.read_csv('teste.csv',sep="|")
nome = torch.cuda.get_device_name() 
params = dict()
params["device"] = "cuda"
params["tree_method"] = "hist"

X,Y = PreProcessamento(df)

