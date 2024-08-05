import numpy as np
import pandas as pd
from category_encoders import BinaryEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
        X = numeric_dataset
        Y = numeric_dataset['is_fraud']
        return X, Y
    except KeyError:
        Teste = numeric_dataset
        return Teste,NumTransacoes
    

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
