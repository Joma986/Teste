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


class RBM(nn.Module): # Criação da RBM do Pytorch (Aceita inputs numéricos de 0 a 1 e Binários).
    # Criação da classe, defini-se o quais atributos a classe tem.
    def __init__(self, n_visible, n_hidden): 
        super(RBM, self).__init__() # Inicia a Classe.
        self.n_visible = n_visible # Classe RBM tem um atributo Visible Layers.
        self.n_hidden = n_hidden # Classe RBM tem um atributo  Hidden Layers.
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01) # Atributo Matriz de Peso W.
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))  # Classe tem um atributo para camada de Bias Visible.
        self.v_bias = nn.Parameter(torch.zeros(n_visible))  # Classe tem um atributo para camada de Bias Hidden.

        '''-------------Aqui definimos os métodos, ou seja, as funções associada a classe RBM --------------------'''

    def sample_from_p(self, p):
        """O seguinte método tem a função de receber um tensor de porbabilidade p, 
           ou seja, um tensor onde todos seus atributos são números entre 0 e 1. 
           Os valores são correspondentes a cada variavel visivel e cada variavel oculta. 
           Ou seja, cada valor de p representa a ativação de uma camada específica.

           O método torch.bernoulli, faz uma amostragem de Bernoulli para cada entrada do tensor p, 
           retornando um valor binário com uma probabilidade. Ou seja, se o tensor de entrada é [0.8],
           existe 80% de chance do retorno da função ser 1 e 20% de ser 0.
         """
        return torch.bernoulli(p)  

    def v_to_h(self, v):
        """O presente método é utilizado para calcular a probabilidade de ativação de uma camada 
           oculta dada a ativação de uma camada visivel. Em termos mais técnicos, P(hi = 1|v).

           A linha 57 monta o valor da probabilidade condicional em relação a entrada v, 
           a linha 58 utiliza do método sample_from_p para transformar as probabilidades em valores binários. 
        """
        p_h_given_v = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        sample_h = self.sample_from_p(p_h_given_v)
        return p_h_given_v, sample_h

    def h_to_v(self, h):
        """Mesma coisa do método anterior, porém com a probabilidade de P(vi = 1|h)"""
        p_v_given_h = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        sample_v = self.sample_from_p(p_v_given_h)
        return p_v_given_h, sample_v

    def forward(self, v):
        """O presente método é utilizado para fazer a atualição feed-foward na RBM. 
           O método self.v_to_h atualiza os parâmetros da camada Hidden(h) e acordo com a 
           probabilidade condicional de P(vi = 1|h). Já o método self.h_to_v atualiza
           os parâmetros das camadas visíveis de acordo com a probabilidade condicional 
           P(hi = 1|v).
        """
        p_h_given_v, h = self.v_to_h(v)
        p_v_given_h, v = self.h_to_v(h)
        return v

    def free_energy(self, v):
        """O presente método é utilizado para fazer calcular a atual energia livre da RBM. A energia livre
           pode se ser entendida como a ""Função Custo"" da RBM. Note que, ""Função Custo"" está entre muitas
           aspas, isso se dá, pois RBM é um método NÃO supervisionado, a ideia de função custo é apenas uma alegoria
           para melhor entedimento. A energia livre é utilizada para saber o quão bem uma RBM consegue reconstruir 
           dados de entrada.   
        """
        vbias_term = v.mv(self.v_bias) # Produto matricial entre v e a.
        wx_b = torch.matmul(v, self.W.t()) + self.h_bias # Produto matricial v.W^t + b
        hidden_term = wx_b.exp().add(1).log().sum(1) # Calcula o termo exponencial da energia.
        return (-hidden_term - vbias_term).mean() #Retorna a média da energia por Mini-Batch.

def train_rbm(rbm, train_loader, lr=0.01, k=1, epochs=10, device='cpu',plot = 0): # Função de treino da RBM, através do Contrastive Divergence. Aqui a magia acontece (Ou deveria).
    plots = []
    rbm.to(device) # Faz com que o treino seja realizado no dispositivo escolhido (CPU ou GPU).
    optimizer = optim.SGD(rbm.parameters(), lr=lr) # Inicia o Stogastic Gradient Descent para otimizar a função.
    for epoch in range(epochs): # Loop de treino.
        epoch_loss = 0 # Variável para acumular os valores de perda na epoch do treino.
        for _, (data, _) in enumerate(train_loader): # Loop do treino | obs: _ é apenas um place holder para uma variável que não será usada
            data = data.view(-1, rbm.n_visible).to(device) # Redimensiona o Tensor data e garante que ele seja movido para o device certo.
            sample_data = data.bernoulli() # Realiza a amostragem dos dados de entrada. 

            v = sample_data #Inicializa v com as amostras binárias
            for _ in range(k):
                _, h = rbm.v_to_h(v) # Dadas as unidades visiveis, calcula as ocultas.
                _, v = rbm.h_to_v(h) # Dadas as unidades ocultas calcula as visiveis.
            
            v_ = v.detach()  # Para evitar a atualização dos gradientes durante o Backpropagation
            
            # Free energy
            loss = rbm.free_energy(sample_data) - rbm.free_energy(v_) # Calcula diferença entre o valor calculado e o valor real
            epoch_loss += loss.item() # Adiciona o valor da perda da Mini-Batch
            
            optimizer.zero_grad() # Zera o Gradiente.
            loss.backward(retain_graph=True) # Atualiza os parametros da perda.
            optimizer.step() # Atualiza os parametros do modelo.
        if plot == 1: 
            plots.append(epoch_loss/len(train_loader.dataset))
        print(f'Epoch {epoch+1}: Loss: {epoch_loss/len(train_loader.dataset)}')
    if plot == 1:
        plt.plot(plots)
        plt.savefig(f'RBM_{epochs}_Epochs.png')
        plt.close()

def separaidades(string):
    if type(string) != str:
      print("Função só funciona com String")
      print("\n")
      print(string)
      return
    else: 
       classe = string.split("_")[0]
       idade = string.split("_")[1]
       if idade == "50up":
          return "Idoso"
       else:
          if classe == "adults":
            return "Adulto"
          if classe == "young":
            return "Jovem"

def PreProcessamento(df):
    NumTransacoes = df['trans_num']
    df['adr'] = df['state']
    
    # Definindo bins e labels para mês e hora
    binsMes = np.arange(1, 14)
    labelsMes = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    binsHora = [-1, 6, 12, 18, 24]
    labelsHora = ['Madrugada', 'Manhã', 'Tarde', 'Noite']
    
    # Criando colunas de mês e período do dia
    df['mes'] = pd.to_datetime(df['trans_date'], format='%Y-%m-%d').dt.month
    df['month_category'] = pd.cut(df['mes'], bins=binsMes, labels=labelsMes, right=False, include_lowest=True)
    df['Hora'] = pd.to_datetime(df['trans_time'], format='%H:%M:%S').dt.hour
    df['time_period'] = pd.cut(df['Hora'], bins=binsHora, labels=labelsHora, right=False, include_lowest=True)
    
    # Removendo colunas desnecessárias
    df.drop(['cc_num', 'ssn', 'first', 'last', 'acct_num', 'trans_num', 'merch_lat', 'merch_long', 'zip', 'street', 'city', 'state', 'dob', 'unix_time', 'mes', 'Hora', 'trans_time', 'trans_date'], axis=1, inplace=True)
    df.dropna(inplace=True)
    
    # Mapeando a coluna 'profile' (certifique-se que 'separaidades' está definido)
    df['profile'] = df['profile'].map(separaidades)
    
    # Aplicando o StandardScaler
    scalerMinMax = MinMaxScaler()
    df[['lat', 'long', 'city_pop', 'amt']] = scalerMinMax.fit_transform(df[['lat', 'long', 'city_pop', 'amt']])
    
    # Codificando colunas categóricas
    enc = BinaryEncoder(cols=['job', 'gender', 'profile', 'category', 'merchant', 'adr', 'month_category', 'time_period']).fit(df)
    numeric_dataset = enc.transform(df)
    
    try:
        X = numeric_dataset.drop('is_fraud', axis=1)
        Y = numeric_dataset['is_fraud']
        return X, Y
    except KeyError:
        Teste = numeric_dataset
        return Teste,NumTransacoes

df = pd.read_csv('treino.csv',sep="|")
teste = pd.read_csv('teste.csv',sep="|")
nome = torch.cuda.get_device_name() 
params = dict()
params["device"] = "cuda"
params["tree_method"] = "hist"

X,Y = PreProcessamento(df)
smote = SMOTE(random_state=42)
X,Y = smote.fit_resample(X,Y)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state = 42)


# Definir o device (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)


n_visible = X_train.shape[1]
n_hidden = 83

n_visible_rbm2 = n_hidden
n_hidden_rbm2 = 50

n_visible_rbm3 = n_hidden_rbm2
n_hidden_rbm3 = 25

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)

print(f"Nome do Dispositivo: {nome}")
rbm = RBM(n_visible=n_visible, n_hidden=n_hidden).to(device)
rbm2 = RBM(n_visible=n_visible_rbm2, n_hidden=n_hidden_rbm2).to(device)
rbm3 = RBM(n_visible=n_visible_rbm3, n_hidden=n_hidden_rbm3).to(device)
train_rbm(rbm, train_loader, lr = 0.0001, k = 1, epochs = 10, device = device)
_, X_train_features_rbm1 = rbm.v_to_h(X_train_tensor)
_, X_test_features_rbm1 = rbm.v_to_h(X_test_tensor)
X_test_recon = rbm(X_test_tensor)
X_test_recon_np = X_test_recon.detach().cpu().numpy()


train_loader_rbm2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_features_rbm1, y_train_tensor), batch_size=64, shuffle=True)
train_rbm(rbm2, train_loader_rbm2, lr=0.0001, k=1, epochs=10, device=device)

_, X_train_features_rbm2 = rbm2.v_to_h(X_train_features_rbm1)
_, X_test_features_rbm2 = rbm2.v_to_h(X_test_features_rbm1)


# clf = LogisticRegression()
# clf.fit(X_train_features_rbm2.cpu().detach().numpy(), y_train_tensor.cpu().detach().numpy())
# y_pred = clf.predict(X_test_features_rbm2.cpu().detach().numpy())

Numero_de_Anomalias = (reconstruction_error>=0.6).astype(int).sum()
print(f"Erro de reconstrução: {Numero_de_Anomalias}")



X_train_features_np = X_train_features.detach().cpu().numpy()
X_test_features_np = X_test_features.detach().cpu().numpy()
pre = []
rec = []  

# for i in np.linspace(0, 1, 100):
#     XGB = XGBClassifier(colsample_bytree = 1.0, learning_rate = 0.2, max_depth = 5, n_estimators = 300, subsample =  0.8)

#     XGB.fit(X_train_features_np, y_train)

#     y_pred = XGB.predict_proba(X_test_features_np)[:,1]

#     y_pred = (y_pred >= i).astype(int)

#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     print(f"Precision: {precision}")
#     print(f"Recall: {recall}")
#     pre.append(precision)
#     rec.append(recall)

# plt.plot(list(np.linspace(0, 1, 100)), pre, 'b-', label='Precision')
# plt.plot(list(np.linspace(0, 1, 100)), rec, 'r-', label='Recall')
# plt.xlabel('Threshold')
# plt.ylabel('Score')
# plt.title('Precision e Recall por Threshold')
# plt.legend()
# plt.grid(True)
# plt.savefig(f'XGBoost_RBM_Threshold_{i}_Features.png')
# plt.close()
# print("Classification Report:\n", classification_report(y_test, y_pred))


# XGB = XGBClassifier(colsample_bytree = 1.0, learning_rate = 0.02, max_depth = 5, n_estimators = 300, subsample =  0.8)
# XGB.fit(X_train_features_np, y_train)
# y_pred = XGB.predict_proba(X_test_features_np)[:,1]
# y_pred = (y_pred >= 0.2).astype(int)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)


# Teste,NumTransacoes = PreProcessamento(teste)

# Teste_tensor = torch.tensor(Teste.values, dtype=torch.float32).to(device)

# _, Teste = rbm.v_to_h(Teste_tensor)
# Testes_np = Teste.detach().cpu().numpy()
# y_pred = XGB.predict_proba(Testes_np)[:,1]
# hist = []

# aux = (y_pred >= 0.56).astype(int)
# # hist.append(aux.sum())
# # plt.plot(hist)
# # plt.show()
# pre = pd.DataFrame(aux,columns=["Fraude"])

# submission = pd.concat([NumTransacoes,pre],axis = 1)

# print(submission)

# submission.to_csv('submission.csv',index=False)

# print(aux.sum())

# print("Fim")
