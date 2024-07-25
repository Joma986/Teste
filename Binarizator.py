import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import os
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier,Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks


def converte_binario(value):
    if pd.isna(value):
        return ''
    if isinstance(value, str):
        return ' '.join(format(ord(char), '08b' ) for char in value) #Convete o texto em ASCII e depois o número do ASCII em binário
    if isinstance(value, int):
        return format(value, 'b') #Convete o valores inteiros em binário
    if isinstance(value, float):
        return ''.join(format(ord(char), '08b') for char in str(value))  #Convete o valores floats em binário
    return value

# def Tratamento(df):
#     df['name'] = df["first"] + " " + df["last"]
#     df['adr'] = df['street'] + ' ' + df['city'] + ' ' + df['state']
#     x = df[['cc_num','amt','lat','long','city_pop','acct_num','unix_time','merch_lat','merch_long']].values
#     min_max_scaler = preprocessing.MinMaxScaler()
#     x_scaled = min_max_scaler.fit_transform(x)
#     df[['cc_num','amt','lat','long','city_pop','acct_num','unix_time','merch_lat','merch_long']] = x_scaled
#     dfnovo = df.drop('dob',axis = 1)
#     df['trans_time'] = df['trans_time'].fillna('00').astype(str)
#     HoraL = df['trans_time'].str[0:2].astype(int)
#     buckets = [-1, 6, 12, 18,24]
#     buckets_name = ['Madrugada','Manhã','Tarde','Noite']
#     dfnovo['trans_time'] = pd.cut(HoraL, buckets , labels = buckets_name)
#     df['category'] = df['category'].fillna('Desconhecido').astype(str)
#     dfnovo['category'] = df['category'].str.split('_').str[0]
#     dfnovo['profile'] = df['profile'].str.split('.').str[0]
#     x = dfnovo[['merch_lat','merch_long']].values
#     min_max_scaler = preprocessing.MinMaxScaler()
#     x_scaled = min_max_scaler.fit_transform(x)
#     dfnovo[['merch_lat','merch_long']] = x_scaled
#     dfnovo.drop(['ssn','cc_num','first','last','street','city','state','zip','lat','long','acct_num','trans_num','unix_time','merch_long','merch_lat'],axis = 1,inplace=True)
#     categorias = dfnovo.select_dtypes(include=['object', 'category']).columns.tolist()
#     dfnovo.dropna(inplace=True)
#     return dfnovo,categorias


# df_Treino = pd.read_csv("treino.csv", delimiter='|')
# df_Teste = pd.read_csv("teste.csv", delimiter='|')
# df_Treino_Tratado ,categorias = Tratamento(df_Treino)[0],Tratamento(df_Treino)[1]
# X = df_Treino_Tratado.drop('is_fraud',axis = 1)
# y = df_Treino_Tratado['is_fraud']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# train_pool = Pool(X_train, y_train, cat_features=categorias)
# test_pool = Pool(X_test, y_test, cat_features=categorias)
# model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=3, verbose=True)
# model.fit(train_pool)
# feature_importances = model.get_feature_importance(train_pool)
# feature_names = X_train.columns
# importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
# importance_df = importance_df.sort_values(by='Importance', ascending=False)
# print(importance_df)
# important_features = importance_df[importance_df['Importance'] >= 10]['Feature'].tolist()
# X_filtered = X_train[important_features]
# X_test_Filtered = X_test[important_features]

# encoded_data_Treino = pd.get_dummies(X_filtered[['trans_time']],dtype = int)
# dfnovoOHE = X_filtered.drop(['trans_time'],axis = 1)


# encoded_data_Teste= pd.get_dummies(X_test_Filtered[['trans_time']],dtype = int)
# dfnovoOHE_Teste = X_test_Filtered.drop(['trans_time'],axis = 1)

# X_test = pd.concat([dfnovoOHE_Teste,encoded_data_Teste],axis = 1)


# X_filtered = pd.concat([dfnovoOHE,encoded_data_Treino],axis = 1)

# smote = SMOTE(random_state=42)
# X_filtered, y_train = smote.fit_resample(X_filtered, y_train)

# tomek = TomekLinks()
# X_filtered, y_train = tomek.fit_resample(X_filtered, y_train)
# RDF = RandomForestClassifier(random_state=42)
# model = RandomForestClassifier(n_estimators=100, random_state=42,verbose=1)

# # Treinar o modelo
# model.fit(X_filtered, y_train)

# # Fazer previsões
# y_pred = model.predict(X_test)

# # Calcular as métricas de recall e precisão
# recall = recall_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# accuracy = accuracy_score(y_test, y_pred)

# print(f'Accuracy: {accuracy}')
# print(f'Recall: {recall}')
# print(f'Precision: {precision}')





# dfbinario = df_Treino.map(converte_binario)
# dfbinario.head(30).to_csv('binarizado.csv', index=False)
