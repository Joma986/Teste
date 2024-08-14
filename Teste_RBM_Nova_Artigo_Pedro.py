import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn
from rbm import RBM
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import NearMiss
from sklearn import preprocessing
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.special import expit
import torch
import torch.nn as nn
import torch.nn.functional as F
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Subtrai o máximo para estabilidade numérica
    return exp_z / np.sum(exp_z)

class BernuliRBM_Classification(nn.Module):
    def __init__(self, n_visible, n_hidden, n_classes, learning_rate=0.01, class_weights=None):
        super(BernuliRBM_Classification, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible))
        self.U = nn.Parameter(torch.randn(n_hidden, n_classes))
        self.b = nn.Parameter(torch.zeros(n_visible))
        self.c = nn.Parameter(torch.zeros(n_hidden))
        self.d = nn.Parameter(torch.zeros(n_classes))
        self.learning_rate = learning_rate
        self.class_weights = class_weights if class_weights is not None else torch.ones(n_classes)

    def sigmoid(self, x):
        return torch.sigmoid(x)

    def forward(self, x, y):
        # Certifique-se de que x e y são tensores PyTorch
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)

        # Fase Positiva
        h = self.sigmoid(self.c.unsqueeze(1) + torch.matmul(self.W, x.T) + torch.matmul(self.U, y.T))
        return h

    def train_model(self, X, Y, batch_size=32, epochs=100):
        n_samples = X.shape[0]
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        for epoch in range(epochs):
            epoch_loss = 0

            # Shuffle data
            indices = torch.randperm(n_samples)
            X = X[indices]
            Y = Y[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i+batch_size]
                Y_batch = Y[i:i+batch_size]

                # Certifique-se de que X_batch e Y_batch são tensores PyTorch
                if isinstance(X_batch, np.ndarray):
                    X_batch = torch.tensor(X_batch, dtype=torch.float32)
                if isinstance(Y_batch, np.ndarray):
                    Y_batch = torch.tensor(Y_batch, dtype=torch.float32)

                # Fase Positiva
                h = self.sigmoid(self.c.unsqueeze(1) + torch.matmul(self.W, X_batch.T) + torch.matmul(self.U, Y_batch.T))

                # Fase Negativa
                x_sample = self.sigmoid(self.b.unsqueeze(0) + torch.matmul(h.T, self.W))
                y_sample = F.softmax(self.d.unsqueeze(0) + torch.matmul(h.T, self.U), dim=-1)

                h_sample = self.sigmoid(self.c.unsqueeze(1) + torch.matmul(self.W, x_sample.T) + torch.matmul(self.U, y_sample.T))

                # Atualização de Gradiente
                W_grad = torch.matmul(h, X_batch) - torch.matmul(h_sample, x_sample)
                U_grad = torch.matmul(h, Y_batch) - torch.matmul(h_sample, y_sample)
                b_grad = torch.sum(X_batch - x_sample, dim=0)
                c_grad = torch.sum(h - h_sample, dim=1)
                d_grad = torch.sum(Y_batch - y_sample, dim=0)

                self.W.grad = -W_grad / batch_size
                self.U.grad = -U_grad / batch_size
                self.b.grad = -b_grad / batch_size
                self.c.grad = -c_grad / batch_size
                self.d.grad = -d_grad / batch_size

                optimizer.step()

                # Cálculo do y_pred e loss
                y_pred = F.softmax(torch.matmul(h.T, self.U) + self.d, dim=-1)
                loss = criterion(y_pred, Y_batch.argmax(dim=1))
                epoch_loss += loss.item()

            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/n_samples}')

    def predict(self, X):
        with torch.no_grad():
            # Calcular as ativações ocultas h
            h = self.sigmoid(torch.matmul(self.W, X.T) + self.c.unsqueeze(1))
            
            # Calcular as probabilidades preditas y_pred
            logits = torch.matmul(h.T, self.U) + self.d  # Aqui d já está broadcastable
            y_pred = F.softmax(logits, dim=-1)
            
            # Retornar a classe com maior probabilidade
            return y_pred.argmax(dim=1)


BATCH_SIZE = 1024
VISIBLE_UNITS = 893
HIDDEN_UNITS1 = 512
HIDDEN_UNITS2 = 128
HIDDEN_UNITS3 = 64
CD_K = 1
EPOCHS = 200
LEARNING_RATE = 9e-2
MOMENTUM = 0.9

print(f'Parametros \n Batch Size: {BATCH_SIZE} \n Visible Units: {VISIBLE_UNITS} \n Hidden Units: {HIDDEN_UNITS1} \n Contrastive Divergence: {CD_K} \n Epochs: {EPOCHS} \n Learning Rate: {LEARNING_RATE} \n Momentum: {MOMENTUM}')

"""# Carregamento de Arquivos"""

filename = 'treino.csv'
df = pd.read_csv(filename,sep = "|")


filename = 'teste.csv'
teste = pd.read_csv(filename,sep = "|")


"""# Tratando o Set de Treinamento

"""

print("Porcentagem de Fraudes:",str(df['is_fraud'].sum()*100/(abs((df['is_fraud']-1).sum()))) + ' %')
print('Numero muito baixo de Fraudes, Dataset Imbalanceado')

#----------------------------------------------Treino------------------------------------------------------------#
Ndob = df['dob'].str.split('-').str
MesL = Ndob[0].to_list()
Mes = [int(x) for x in MesL]
Ano = pd.DataFrame(Mes)
df['dob'] = Ano
#----------------------------------------------Teste-------------------------------------------------------------#
Ndob = teste['dob'].str.split('-').str
MesL = Ndob[0].to_list()
Mes = [int(x) for x in MesL]
Ano = pd.DataFrame(Mes)
teste['dob'] = Ano

#----------------------------------------------Treino------------------------------------------------------------#
df['name'] = df["first"] + " " + df["last"]
df['adr'] = df['street'] + ' ' + df['city'] + ' ' + df['state']
df.fillna(0,inplace=True)
#----------------------------------------------Teste-------------------------------------------------------------#
teste['name'] = teste["first"] + " " + teste["last"]  # Juntando Nome em uma coluna só
teste['adr'] = teste['street'] + ' ' + teste['city'] + ' ' + teste['state'] # Juntando CEP em uma coluna só
teste.fillna(0,inplace=True)

#----------------------------------------------Treino------------------------------------------------------------#
x = df[['cc_num','amt','lat','long','city_pop','acct_num','unix_time','merch_lat','merch_long']].values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df[['cc_num','amt','lat','long','city_pop','acct_num','unix_time','merch_lat','merch_long']] = x_scaled
#----------------------------------------------Teste-------------------------------------------------------------#
x = teste[['cc_num','amt','lat','long','city_pop','acct_num','unix_time','merch_lat','merch_long']].values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
teste[['cc_num','amt','lat','long','city_pop','acct_num','unix_time','merch_lat','merch_long']] = x_scaled

"""# Organizando em Grupos e Matriz de Correlação

Dividir em Clusters por idade e valor gasto??
"""

#----------------------------------------------Treino------------------------------------------------------------#
dfnovo = df
#----------------------------------------------Teste-------------------------------------------------------------#
testenovo = teste

"""Dividindo Tempo em 4 grupos -> Manhã (06-11), Tarde (12-17), Noite (18-23), Madrugada (00-05)"""

#----------------------------------------------Treino------------------------------------------------------------#
df['trans_time'] = df['trans_time'].fillna('00').astype(str)
HoraL = df['trans_time'].str[0:2].astype(int)
#----------------------------------------------Teste-------------------------------------------------------------#
teste['trans_time'] = teste['trans_time'].fillna('00').astype(str)
HoraL = teste['trans_time'].str[0:2].astype(int)

buckets = [0, 6, 12, 18,23]
buckets_name = ['Madrugada','Manhã','Tarde','Noite']
#----------------------------------------------Treino------------------------------------------------------------#
dfnovo['trans_time'] = pd.cut(HoraL, buckets , labels = buckets_name)
#----------------------------------------------Teste-------------------------------------------------------------#
testenovo['trans_time'] = pd.cut(HoraL, buckets , labels = buckets_name)

"""Dividindo Category em Grupos"""

#----------------------------------------------Treino------------------------------------------------------------#
df['category'] = df['category'].fillna('Desconhecido').astype(str)
dfnovo['category'] = df['category'].str.split('_').str[0]
#----------------------------------------------Teste-------------------------------------------------------------#
teste['category'] = teste['category'].fillna('Desconhecido').astype(str)
testenovo['category'] = teste['category'].str.split('_').str[0]

#----------------------------------------------Treino------------------------------------------------------------#
dfnovo['profile'] = df['profile'].str.split('.').str[0].str.split('_').str[1]
#----------------------------------------------Teste-------------------------------------------------------------#
testenovo['profile'] = teste['profile'].str.split('.').str[0].str.split('_').str[1]

# MatrizCorrelação = abs(dfnovo.corr())
# sn.heatmap(MatrizCorrelação)
# dfnovonum = dfnovo[MatrizCorrelação['is_fraud'].index]

# Listadecolunas = abs(MatrizCorrelação) > 0.1
#dfnovonum = dfnovonum.loc[:,Listadecolunas]

#dfnovo.drop(dfnovonum.columns,axis = 1,inplace = True)
#dfnovo.iloc[0]

#----------------------------------------------Treino------------------------------------------------------------#
dfnovo.drop(['acct_num','zip','merch_long','cc_num'], axis = 1, inplace = True)
#----------------------------------------------Teste-------------------------------------------------------------#
testenovo.drop(['acct_num','zip','merch_long','cc_num'], axis = 1, inplace = True)

#----------------------------------------------Treino------------------------------------------------------------#
# dfnovo['job'] = (dfnovo['job'].str.split(',').str[0]).apply(hash)
# dfnovo['merchant'] = (dfnovo['merchant']).apply(hash)
# #----------------------------------------------Teste-------------------------------------------------------------#
# testenovo['job'] = (testenovo['job'].str.split(',').str[0]).apply(hash)
# testenovo['merchant'] = (testenovo['merchant']).apply(hash)

#----------------------------------------------Treino------------------------------------------------------------#
dfnovo.drop(['first','last','street','state','trans_num','trans_date','ssn','city','name','adr'],inplace = True, axis = 1)
#----------------------------------------------Teste-------------------------------------------------------------#
testenovo.drop(['first','last','street','state','trans_num','trans_date','ssn','city','name','adr'],inplace = True, axis = 1)

#----------------------------------------------Treino------------------------------------------------------------#
x = dfnovo[['dob','merch_lat']].values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
dfnovo[['dob','merch_lat']] = x_scaled
#----------------------------------------------Teste-------------------------------------------------------------#
x = testenovo[['dob','merch_lat']].values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
testenovo[['dob','merch_lat']] = x_scaled

"""Que Deus me ajude"""

#----------------------------------------------Treino------------------------------------------------------------#
encoded_data_Treino = pd.get_dummies(dfnovo[['profile','trans_time','category','gender','job','merchant']],dtype = int)

len(encoded_data_Treino)

#----------------------------------------------Teste-------------------------------------------------------------#
encoded_data_Teste = pd.get_dummies(testenovo[['profile','trans_time','category','gender','job','merchant']],dtype = int)

len(encoded_data_Teste)

#----------------------------------------------Treino------------------------------------------------------------#
dfnovoOHE = dfnovo.drop(['profile','trans_time','category','gender','job','merchant'],axis = 1)
#----------------------------------------------Teste-------------------------------------------------------------#
testenovoOHE = testenovo.drop(['profile','trans_time','category','gender','job','merchant'],axis = 1)

#----------------------------------------------Treino------------------------------------------------------------#
dfnovoConcat = pd.concat([dfnovoOHE,encoded_data_Treino],axis = 1)
#----------------------------------------------Teste-------------------------------------------------------------#
dftestenovoConcat = pd.concat([testenovoOHE,encoded_data_Teste],axis = 1)

dftestenovoConcat[list(set(dfnovoConcat) - set(dftestenovoConcat))] = 0

"""Matriz de Correlação

"""

# MatrizCorrelação = abs(dfnovoConcat.corr())
# sn.heatmap(MatrizCorrelação)

try:
  dfnovoConcat.drop('0',axis = 1,inplace = True)
  dftestenovoConcat.drop('0',axis = 1,inplace = True)
except:
  print('Correto')
#----------------------------------------------Treino------------------------------------------------------------#
X = dfnovoConcat.drop('is_fraud',axis = 1)
Yh = dfnovoConcat['is_fraud']
#----------------------------------------------Teste-------------------------------------------------------------#
P = dftestenovoConcat.drop('is_fraud',axis = 1)
Y = dftestenovoConcat['is_fraud']

X_train,X_test,y_train,y_test = train_test_split(X,Yh,random_state = 42)
y_train = y_train.to_numpy()
y_train_one_hot = np.zeros((y_train.size, 2))  # 2 é o número de classes

for elem in range(len(y_train)):
    if y_train[elem] == 0:
        y_train_one_hot[elem] = np.array([1, 0])
    else: 
        y_train_one_hot[elem] = np.array([0, 1])

print(y_train_one_hot)
class_weights = np.array([1,1000])
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
rbm = BernuliRBM_Classification(VISIBLE_UNITS, HIDDEN_UNITS1, 2, learning_rate=0.01, class_weights = class_weights_tensor)

rbm.train_model(X_train.to_numpy(),y_train_one_hot,32,30)

X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)

rbm.predict(X_test)