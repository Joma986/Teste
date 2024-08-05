import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_curve, precision_recall_curve, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from rbm import RBM
from sklearn.linear_model import LogisticRegression
from PreProcesament import PreProcessamento
from catboost import CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
from imblearn.over_sampling import SMOTE
from sklearn.cluster import DBSCAN, KMeans
from scipy.special import expit
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
        if (abs(error[epoch] - MenorErro) > 0.1) and (abs(error[epoch] - MenorErro) != 0) and (Threshold != 0):
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

# Parâmetros
BATCH_SIZE = 1024
VISIBLE_UNITS = 43
HIDDEN_UNITS1 = 256
HIDDEN_UNITS2 = 25
HIDDEN_UNITS3 = 12
CD_K = 1
EPOCHS = 200
LEARNING_RATE = 9e-2
MOMENTUM = 0.9

print(f'Parametros \n Batch Size: {BATCH_SIZE} \n Visible Units: {VISIBLE_UNITS} \n Hidden Units: {HIDDEN_UNITS1} \n Contrastive Divergence: {CD_K} \n Epochs: {EPOCHS} \n Learning Rate: {LEARNING_RATE} \n Momentum: {MOMENTUM}')

# Inicialização do dispositivo -> Escolher trabalhar com a GPU ao invês da CPU.
CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")

# Carregar e pré-processar dados.
df = pd.read_csv('treino.csv', sep="|")
teste = pd.read_csv('teste.csv', sep="|")
X, Y = PreProcessamento(df)
smt = SMOTE()
X, Y = smt.fit_resample(X, Y)
X_fraudes = (X.loc[X['is_fraud'] == True]).drop('is_fraud',axis = 1)
X_Nao_fraudes = (X.loc[X['is_fraud'] == False]).drop('is_fraud',axis = 1)


# X, Y = PreProcessamento(df)
#Divisão dos dados em treino e teste.




# train_pool = Pool(X_train, y_train, feature_names=X.columns.to_list())
# test_pool = Pool(X_test, y_test, feature_names=X.columns.to_list())
# model = CatBoostClassifier(iterations=1000, random_seed=0,verbose= 10)
# summary = model.select_features(
# train_pool,
# eval_set=test_pool,
# features_for_select='0-42',
# num_features_to_select=25,
# steps=3,
# algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
# shap_calc_type=EShapCalcType.Regular,
# train_final_model=True,
# plot=True
# )

# # selected_features_indices = summary['selected_features'] 
#  # Indices das features selecionadas
# colunas = X_train.columns[selected_features_indices]
# X_train = X_train[colunas]
# X_test = X_test[colunas]

# Conversão para tensores -> Pytorch
X_fraudes_tensor = torch.tensor(X_fraudes.values, dtype=torch.float32)
# X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
X_Nao_fraudes_tensor = torch.tensor(X_Nao_fraudes.values, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Criação dos DataLoader -> Pytorch
X_fraudes_dataset = TestDataset(X_fraudes_tensor)
Fraudes_loader = DataLoader(X_fraudes_dataset, batch_size=BATCH_SIZE, shuffle=True)
X_Nao_fraudes_dataset = TestDataset(X_Nao_fraudes_tensor)
Nao_fraudes_loader = DataLoader(X_Nao_fraudes_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Treinamento da primeira DBN para dados Não Fraudes
print('Training first DBN...')
rbm1 = RBM(VISIBLE_UNITS, HIDDEN_UNITS1, CD_K, learning_rate=LEARNING_RATE, momentum_coefficient=MOMENTUM, use_cuda=CUDA)
rbm2 = RBM(HIDDEN_UNITS1, HIDDEN_UNITS2, CD_K, learning_rate=LEARNING_RATE, momentum_coefficient=MOMENTUM, use_cuda=CUDA)
rbm3 = RBM(HIDDEN_UNITS2, HIDDEN_UNITS3, CD_K, learning_rate=LEARNING_RATE, momentum_coefficient=MOMENTUM, use_cuda=CUDA)
train_rbm(Fraudes_loader, rbm1, EPOCHS, BATCH_SIZE, VISIBLE_UNITS, CUDA, Threshold = 10)

# Extração de características da primeira RBM
print('Extracting features from first RBM...')
X_fraudes_array = extract_features(Fraudes_loader, rbm1, HIDDEN_UNITS1, VISIBLE_UNITS, CUDA)
X_fraudes_array = expit(X_fraudes_array)

# Ajuste dos tensores para a segunda RBM
X_fraudes_tensor = torch.tensor(X_fraudes_array, dtype=torch.float32)
Fraudes_dataset = TestDataset(X_fraudes_tensor)
Fraudes_loader = DataLoader(Fraudes_dataset, batch_size = BATCH_SIZE, shuffle=True)
train_rbm(Fraudes_loader, rbm2, EPOCHS, BATCH_SIZE, HIDDEN_UNITS1, CUDA, Threshold = 10)

X_fraudes_array = extract_features(Fraudes_loader, rbm2, HIDDEN_UNITS2, HIDDEN_UNITS1, CUDA)
X_fraudes_array = expit(X_fraudes_array)


# Ajuste dos tensores para a segunda RBM
X_fraudes_tensor = torch.tensor(X_fraudes_array, dtype=torch.float32)
Fraudes_dataset = TestDataset(X_fraudes_tensor)
Fraudes_loader = DataLoader(Fraudes_dataset, batch_size = BATCH_SIZE, shuffle=True)
train_rbm(Fraudes_loader, rbm3, EPOCHS, BATCH_SIZE, HIDDEN_UNITS3, CUDA, Threshold = 10)

print('Training second DBN...')
rbm4 = RBM(VISIBLE_UNITS, HIDDEN_UNITS1, CD_K, learning_rate=LEARNING_RATE, momentum_coefficient=MOMENTUM, use_cuda=CUDA)
rbm5 = RBM(HIDDEN_UNITS1, HIDDEN_UNITS2, CD_K, learning_rate=LEARNING_RATE, momentum_coefficient=MOMENTUM, use_cuda=CUDA)
rbm6 = RBM(HIDDEN_UNITS2, HIDDEN_UNITS3, CD_K, learning_rate=LEARNING_RATE, momentum_coefficient=MOMENTUM, use_cuda=CUDA)

train_rbm(Nao_fraudes_loader, rbm4, EPOCHS, BATCH_SIZE, VISIBLE_UNITS, CUDA, Threshold = 10)
print('Extracting features from first RBM...')
Nao_fraudes_array = extract_features(Nao_fraudes_loader, rbm4, HIDDEN_UNITS1, VISIBLE_UNITS, CUDA)
Nao_fraudes_array = expit(Nao_fraudes_array)



# Ajuste dos tensores para a segunda RBM
X_Nao_fraudes_tensor = torch.tensor(Nao_fraudes_array, dtype=torch.float32)
Nao_Fraudes_dataset = TestDataset(X_Nao_fraudes_tensor)
Nao_Fraudes_loader = DataLoader(Nao_Fraudes_dataset, batch_size = BATCH_SIZE, shuffle=True)
train_rbm(Nao_Fraudes_loader, rbm5, EPOCHS, BATCH_SIZE, HIDDEN_UNITS1, CUDA, Threshold = 10)

Nao_fraudes_array = extract_features(Nao_Fraudes_loader, rbm5, HIDDEN_UNITS2, HIDDEN_UNITS1, CUDA)
Nao_fraudes_array = expit(Nao_fraudes_array)

# Ajuste dos tensores para a segunda RBM
X_Nao_fraudes_tensor = torch.tensor(Nao_fraudes_array, dtype=torch.float32)
Nao_Fraudes_dataset = TestDataset(X_Nao_fraudes_tensor)
Nao_Fraudes_loader = DataLoader(Nao_Fraudes_dataset, batch_size = BATCH_SIZE, shuffle=True)
train_rbm(Nao_Fraudes_loader, rbm6, EPOCHS, BATCH_SIZE, HIDDEN_UNITS2, CUDA, Threshold = 10)



# test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
# test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False)

# # Treinamento da segunda RBM
# print('Training second RBM...')
# rbm2 = RBM(HIDDEN_UNITS1, HIDDEN_UNITS2, CD_K, learning_rate=5e-2, momentum_coefficient=MOMENTUM, use_cuda=CUDA)
# train_rbm(train_loader, rbm2, EPOCHS, BATCH_SIZE, HIDDEN_UNITS1, CUDA, Threshold = 10)

# # Extração de características da segunda RBM
# print('Extracting features from second RBM...')
# train_features, train_labels = extract_features(train_loader, rbm2, HIDDEN_UNITS2, HIDDEN_UNITS1, CUDA)
# test_features, test_labels = extract_features(test_loader, rbm2, HIDDEN_UNITS2, HIDDEN_UNITS1, CUDA)

# Classificação com RandomForest
# print('Classifying with Random Forest...')
# clf = RandomForestClassifier()
# print('Fitting...')
# clf.fit(train_features, train_labels)
# # predictions = clf.predict(test_features)
# proba_predictions = clf.predict_proba(test_features)[:, 1]
# pre = []
# rec = []
# for Threshold in np.linspace(0,1,100):
#     aux = (proba_predictions >= Threshold).astype('int64')
#     precision = precision_score(test_labels, aux)
#     recall = recall_score(test_labels, aux)
#     pre.append(precision)
#     rec.append(recall)
# plt.plot(np.linspace(0,1,100),pre)
# plt.plot(np.linspace(0,1,100),rec)
# plt.show()
# # Avaliação
# aux = (proba_predictions >= 0.139).astype('int64')
# precision = precision_score(y_test, aux)
# recall = recall_score(y_test, aux)
# MatrziDeConfusão = confusion_matrix(y_test,aux)
# print(f'Precision: {precision:.4f}')
# print(f'Recall: {recall:.4f}')

# # Plots de avaliação
# fpr, tpr, _ = roc_curve(test_labels, proba_predictions)
# precision, recall, _ = precision_recall_curve(test_labels, proba_predictions)

# plt.figure()
# plt.plot(fpr, tpr, label='ROC Curve')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(recall, precision, label='Precision-Recall Curve')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend()
# plt.show()

# Pre-processamento e extração de características para novo conjunto de dados
print('Processing new test data...')
Teste, NumTransacoes = PreProcessamento(teste)
X_novo_tensor = torch.tensor(Teste.values, dtype=torch.float32)
novo_dataset = TestDataset(X_novo_tensor)
novo_loader_orig = DataLoader(novo_dataset, batch_size=BATCH_SIZE, shuffle=False)

novo_features_1 = extract_features(novo_loader_orig, rbm1, HIDDEN_UNITS1, VISIBLE_UNITS, CUDA)
novo_dataset = TestDataset(novo_features_1)
novo_loader = DataLoader(novo_dataset, batch_size=BATCH_SIZE, shuffle=False)
novo_features_2 = extract_features(novo_loader, rbm2, HIDDEN_UNITS2, HIDDEN_UNITS1, CUDA)
novo_dataset = TestDataset(novo_features_2)
novo_loader = DataLoader(novo_dataset, batch_size=BATCH_SIZE, shuffle=False)
novo_features_3 = extract_features(novo_loader, rbm3, HIDDEN_UNITS3, HIDDEN_UNITS2, CUDA)



novo_features_4 = extract_features(novo_loader_orig, rbm4, HIDDEN_UNITS1, VISIBLE_UNITS, CUDA)
novo_dataset = TestDataset(novo_features_4)
novo_loader = DataLoader(novo_dataset, batch_size=BATCH_SIZE, shuffle=False)
novo_features_5 = extract_features(novo_loader, rbm5, HIDDEN_UNITS2, HIDDEN_UNITS1, CUDA)
novo_dataset = TestDataset(novo_features_5)
novo_loader = DataLoader(novo_dataset, batch_size=BATCH_SIZE, shuffle=False)
novo_features_6 = extract_features(novo_loader, rbm6, HIDDEN_UNITS3, HIDDEN_UNITS2, CUDA)

concatenated_array = []
for i in range (0,len(novo_features_6)):
    concatenated_array.append(np.concatenate((novo_features_3[i], novo_features_6[i]), axis=0))

# Classificação usando o modelo treinado
eps = 0.5 
min_samples = 5  
dbz = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbz.fit_predict(concatenated_array)
plt.hist(labels)
plt.show()
print('Predictions for the New Test Dataset:')
print('Fraudes:', labels.sum())

pre = pd.DataFrame(labels, columns=["Fraude"])
submission = pd.concat([NumTransacoes, pre], axis=1)
submission.to_csv('submission.csv', index=False)
print("Fim")
