import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn
from rbm import RBM
import torch.nn.functional as F
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import BinaryCrossentropy
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import NearMiss
from sklearn import preprocessing
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.special import expit


"""# Definição de Funções"""

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

def train_rbm(train_loader, rbm, epochs, batch_size, visible_units, cuda, Threshold=0):
    error = []
    i = 0
    for epoch in range(epochs):
        epoch_error = 0.0 # Inicializa o erro como zero em cada iteração.
        '''--------------------------------Incialização de parâmetros da RBM-----------------------------------'''
        for batch in train_loader:
            '''Cada Batch é um Dicionário com as chaves Input e Target, aqui o código 
               extrai apenas o target dentre os dois, para realizar o treino.'''  
            inputs = batch['input'] # Cada Batch é um Dicionário com as chaves.
            inputs = inputs.float() # Converte o Tensor para a dimensão desejada (Geralmente para imagem, precisa pra esse caso?).

            if cuda:
                inputs = inputs.to(device) # Tranfere os inputs para a GPU caso esteja disponivel

            '''---------------------------------Calculo de Erro da RBM-----------------------------------------'''
            batch_error = rbm.contrastive_divergence(inputs) # Calcula os erros do batch através do contrastive divergence
            epoch_error += batch_error # Atualiza os erros da epoca.
        print(f'Epoch Error (epoch={epoch}): {epoch_error.mean():.4f}')
        error.append(epoch_error.mean()) # Atualiza o vetor de erros para o EarlyStopping.
        MenorErro = min(error) # Erro minimo do processo
        
        '''----------------------------------Processo de EarlyStopping-----------------------------------------'''
        if (abs(error[epoch] - MenorErro) > 0.001) and (abs(error[epoch] - MenorErro) != 0) and (Threshold != 0):
            i += 1
            print(f'EarlyStopping {i}/{Threshold}')
            if i == Threshold:
                rbm.weights = MelhoresPesos
                rbm.visible_bias = MelhoresViesesVisiveis
                rbm.hidden_bias = MelhoresViesesOcultos
                print(f'EarlyStopping atuou. Pesos salvos da Época {melhorepoch}') 
                break
        else:
            if MenorErro < 1e-4:
                melhorepoch = epoch
                MelhoresPesos = rbm.weights
                MelhoresViesesVisiveis = rbm.visible_bias
                MelhoresViesesOcultos = rbm.hidden_bias 
                break
            melhorepoch = epoch
            MelhoresPesos = rbm.weights
            MelhoresViesesVisiveis = rbm.visible_bias
            MelhoresViesesOcultos = rbm.hidden_bias 
            i = 0

def extract_features(loader, rbm, hidden_units, visible_units, cuda):
    features = []
    labels = []
    
    for batch in loader:
        inputs = batch['input']
        inputs = inputs.view(len(inputs), visible_units).float()

        if cuda:
            inputs = inputs.to(device)

        hidden_features = rbm.sample_hidden(inputs).cpu().detach().numpy()
        features.append(hidden_features)
        if 'target' in batch:
            labels.append(batch['target'].numpy())

    features = np.vstack(features)
    if labels:
        labels = np.concatenate(labels)
        return features, labels
    return features

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



#X.drop(['merchant','job'],inplace = True, axis = 1)
st = SMOTEENN()
X,Yh = st.fit_resample(X, Yh)
X_train,X_test,Y_train,Y_test = train_test_split(X,Yh,test_size=0.2,random_state=42)

X_F = X_train.loc[Yh == 1]
X_NF = X_train.loc[Yh == 0]

Y_F = Y_train.loc[Yh == 1]
Y_NF = Y_train.loc[Yh == 1]

X_FT = torch.tensor(X_F.values, dtype=torch.float32)
X_NFT = torch.tensor(X_NF.values, dtype=torch.float32)


X_FTD = TestDataset(X_FT)
X_FTDD = DataLoader(X_FTD, batch_size=BATCH_SIZE, shuffle=True)

X_NFTD = TestDataset(X_NFT)
X_NFTDD = DataLoader(X_NFTD, batch_size=BATCH_SIZE, shuffle=True)

# DBN NÃO FRAUDES:
CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")

print('Trainando a DBM para Não Fraudes...')
rbm1 = RBM(VISIBLE_UNITS, HIDDEN_UNITS1, CD_K, learning_rate=LEARNING_RATE, momentum_coefficient=MOMENTUM, use_cuda=CUDA)
rbm2 = RBM(HIDDEN_UNITS1, HIDDEN_UNITS2, CD_K, learning_rate=LEARNING_RATE, momentum_coefficient=MOMENTUM, use_cuda=CUDA)
rbm3 = RBM(HIDDEN_UNITS2, HIDDEN_UNITS3, CD_K, learning_rate=LEARNING_RATE, momentum_coefficient=MOMENTUM, use_cuda=CUDA)


train_rbm(X_NFTDD, rbm1, EPOCHS, BATCH_SIZE, VISIBLE_UNITS, CUDA, Threshold = 5)
X_NF = extract_features(X_NFTDD, rbm1, HIDDEN_UNITS1, VISIBLE_UNITS, CUDA)
X_NFT = torch.tensor(X_NF, dtype=torch.float32)
X_NFT = F.relu(X_NFT)
X_NFTD = TestDataset(X_NFT)
X_NFTDD = DataLoader(X_NFTD, batch_size=BATCH_SIZE, shuffle=True)

train_rbm(X_NFTDD, rbm2, EPOCHS, BATCH_SIZE, HIDDEN_UNITS1, CUDA, Threshold = 5)

X_NF = extract_features(X_NFTDD, rbm2, HIDDEN_UNITS2, HIDDEN_UNITS1, CUDA)
X_NFT = torch.tensor(X_NF, dtype=torch.float32)
X_NFT = F.relu(X_NFT)

# Ajuste dos tensores para a segunda RBM
# X_NFT = torch.tensor(X_NF, dtype=torch.float32)
# X_NFTD = TestDataset(X_NFT)
# X_NFTDD = DataLoader(X_NFTD, batch_size = BATCH_SIZE, shuffle=True)

# train_rbm(X_NFTDD, rbm3, EPOCHS, BATCH_SIZE, HIDDEN_UNITS3, CUDA, Threshold = 5)

# X_NF = extract_features(X_NFTDD, rbm3, HIDDEN_UNITS3, HIDDEN_UNITS2, CUDA)
# X_NF = expit(X_NF)
print('Treinando a DBN para Fraudes...')
# DBN PARA FRAUDES
print('Training first DBN...')
rbm4 = RBM(VISIBLE_UNITS, HIDDEN_UNITS1, CD_K, learning_rate=LEARNING_RATE, momentum_coefficient=MOMENTUM, use_cuda=CUDA)
rbm5 = RBM(HIDDEN_UNITS1, HIDDEN_UNITS2, CD_K, learning_rate=LEARNING_RATE, momentum_coefficient=MOMENTUM, use_cuda=CUDA)
rbm6 = RBM(HIDDEN_UNITS2, HIDDEN_UNITS3, CD_K, learning_rate=LEARNING_RATE, momentum_coefficient=MOMENTUM, use_cuda=CUDA)


train_rbm(X_FTDD, rbm4, EPOCHS, BATCH_SIZE, VISIBLE_UNITS, CUDA, Threshold = 5)
X_F = extract_features(X_FTDD, rbm4, HIDDEN_UNITS1, VISIBLE_UNITS, CUDA)
X_FT = torch.tensor(X_F, dtype=torch.float32)
X_FT = F.relu(X_FT)

X_FT = torch.tensor(X_F, dtype=torch.float32)
X_FTD = TestDataset(X_FT)
X_FTDD = DataLoader(X_FTD, batch_size = BATCH_SIZE, shuffle=True)

train_rbm(X_FTDD, rbm5, EPOCHS, BATCH_SIZE, HIDDEN_UNITS1, CUDA, Threshold = 5)

X_FT = torch.tensor(X_F, dtype=torch.float32)
X_FT = F.relu(X_FT)

# # Ajuste dos tensores para a segunda RBM
# X_FT = torch.tensor(X_F, dtype=torch.float32)
# X_FTD = TestDataset(X_FT)
# X_FTDD = DataLoader(X_FTD, batch_size = BATCH_SIZE, shuffle=True)

# train_rbm(X_FTDD, rbm6, EPOCHS, BATCH_SIZE, HIDDEN_UNITS3, CUDA, Threshold = 5)

# X_F = extract_features(X_FTDD, rbm6, HIDDEN_UNITS3, HIDDEN_UNITS2, CUDA)
# X_F = expit(X_F)

#----------------------------------------------Treino------------------------------------------------------------#
# X = X.to_numpy()
# Yh = Yh.to_numpy()
# #----------------------------------------------Teste-------------------------------------------------------------#
# P = P.to_numpy()
# Y = Y.to_numpy()

X_T = X_test
Y_T = Y_test
X_novo_tensor = torch.tensor(X_T.values, dtype=torch.float32)
novo_dataset = TestDataset(X_novo_tensor)
novo_loader_orig = DataLoader(novo_dataset, batch_size=BATCH_SIZE, shuffle=False)
print('Testando Não Fraudes:')

novo_features_1 = extract_features(novo_loader_orig, rbm1, HIDDEN_UNITS1, VISIBLE_UNITS, CUDA)
X_novo_tensor = torch.tensor(novo_features_1, dtype=torch.float32)
novo_features_1 = F.relu(X_novo_tensor)
novo_dataset = TestDataset(novo_features_1)
novo_loader = DataLoader(novo_dataset, batch_size=BATCH_SIZE, shuffle=False)
novo_features_2 = extract_features(novo_loader, rbm2, HIDDEN_UNITS2, HIDDEN_UNITS1, CUDA)
X_novo_tensor = torch.tensor(novo_features_2, dtype=torch.float32)
novo_features_2 = F.relu(X_novo_tensor)
# novo_dataset = TestDataset(novo_features_2)
# novo_loader = DataLoader(novo_dataset, batch_size=BATCH_SIZE, shuffle=False)
# novo_features_3 = extract_features(novo_loader, rbm3, HIDDEN_UNITS3, HIDDEN_UNITS2, CUDA)
# novo_features_3 = expit(novo_features_3)

print('Testando Fraudes:')

novo_features_4 = extract_features(novo_loader_orig, rbm4, HIDDEN_UNITS1, VISIBLE_UNITS, CUDA)
X_novo_tensor = torch.tensor(novo_features_4, dtype=torch.float32)
novo_features_4 = F.relu(X_novo_tensor)
novo_dataset = TestDataset(novo_features_4)
novo_loader = DataLoader(novo_dataset, batch_size=BATCH_SIZE, shuffle=False)
novo_features_5 = extract_features(novo_loader, rbm5, HIDDEN_UNITS2, HIDDEN_UNITS1, CUDA)
X_novo_tensor = torch.tensor(novo_features_5, dtype=torch.float32)
novo_features_5 = F.relu(X_novo_tensor)
# novo_dataset = TestDataset(novo_features_5)
# novo_loader = DataLoader(novo_dataset, batch_size=BATCH_SIZE, shuffle=False)
# novo_features_6 = extract_features(novo_loader, rbm6, HIDDEN_UNITS3, HIDDEN_UNITS2, CUDA)
# novo_features_6 = expit(novo_features_6)

concatenated_array = []
for i in range (0,len(novo_features_5)):
    concatenated_array.append(np.concatenate((novo_features_2[i], novo_features_5[i]), axis=0))

X_train,X_test,Y_train,Y_test = train_test_split(concatenated_array,Y_test,test_size=0.2,random_state=42)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

clf.fit(X_train,Y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print(f"Precision: {precision_score(Y_test,y_pred)}")
print(f"Precision: {recall_score(Y_test,y_pred)}")


X_novo_tensor = torch.tensor(P.values, dtype=torch.float32)
novo_dataset = TestDataset(X_novo_tensor)
novo_loader_orig = DataLoader(novo_dataset, batch_size=BATCH_SIZE, shuffle=False)

novo_features_1 = extract_features(novo_loader_orig, rbm1, HIDDEN_UNITS1, VISIBLE_UNITS, CUDA)
X_novo_tensor = torch.tensor(novo_features_1, dtype=torch.float32)
novo_features_1 = F.relu(X_novo_tensor)
novo_dataset = TestDataset(novo_features_1)
novo_loader = DataLoader(novo_dataset, batch_size=BATCH_SIZE, shuffle=False)
novo_features_2 = extract_features(novo_loader, rbm2, HIDDEN_UNITS2, HIDDEN_UNITS1, CUDA)
X_novo_tensor = torch.tensor(novo_features_2, dtype=torch.float32)
novo_features_2 = F.relu(X_novo_tensor)


novo_features_4 = extract_features(novo_loader_orig, rbm4, HIDDEN_UNITS1, VISIBLE_UNITS, CUDA)
X_novo_tensor = torch.tensor(novo_features_4, dtype=torch.float32)
novo_features_4 = F.relu(X_novo_tensor)
novo_dataset = TestDataset(novo_features_4)
novo_loader = DataLoader(novo_dataset, batch_size=BATCH_SIZE, shuffle=False)
novo_features_5 = extract_features(novo_loader, rbm5, HIDDEN_UNITS2, HIDDEN_UNITS1, CUDA)
X_novo_tensor = torch.tensor(novo_features_5, dtype=torch.float32)
novo_features_5 = F.relu(X_novo_tensor)

concatenated_array_test = []
for i in range (0,len(novo_features_5)):
    concatenated_array_test.append(np.concatenate((novo_features_2[i], novo_features_5[i]), axis=0))


y_pred = clf.predict_proba(concatenated_array_test)[:,1]

print('Fim')