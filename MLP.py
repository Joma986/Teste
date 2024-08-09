import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm



# Carregando os dados
treino = pd.read_csv('Treino_MLP.csv')
Y = pd.read_csv('Y_MLP.csv')
teste = pd.read_csv('Teste_MLP.csv')
TransNum = pd.read_csv('teste.csv',sep = "|")['trans_num']

# Dividindo os dados em treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(treino, Y, test_size=0.2, random_state=42)

# train_pool = Pool(X_train, Y_train, feature_names=X_train.columns.to_list())
# test_pool = Pool(X_test, Y_test, feature_names=X_train.columns.to_list())

# model = CatBoostClassifier(iterations=1000, random_seed=0,verbose=10)
# summary = model.select_features(
#     train_pool,
#     eval_set=test_pool,
#     features_for_select='0-255',
#     num_features_to_select=50,
#     steps=3,
#     algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
#     shap_calc_type=EShapCalcType.Regular,
#     train_final_model=True,
#     plot=True
# )

# colunas = summary['selected_features_names']

# Convertendo para tensores do TensorFlow
X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
Y_train_tensor = tf.convert_to_tensor(Y_train, dtype=tf.float32)
X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
Y_test_tensor = tf.convert_to_tensor(Y_test, dtype=tf.float32)
teste = tf.convert_to_tensor(teste, dtype=tf.float32)

# Criando o modelo de regularização
normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
normalizer.adapt(X_train_tensor)  # Ajusta o normalizador aos dados de treinamento

# Criando o modelo com a camada de normalização
model = tf.keras.Sequential([
    normalizer,  # Adicionando a camada de normalização
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Última camada para classificação binária
])

# Compilando o modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0009),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Configurando o callback de Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  patience=5,  # Número de épocas sem melhoria antes de parar
                                                  restore_best_weights=True)

# Treinando o modelo com Early Stopping
history = model.fit(X_train_tensor, Y_train_tensor, 
                    epochs=100,  # Número máximo de épocas
                    batch_size=16, 
                    validation_data=(X_test_tensor, Y_test_tensor),
                    callbacks=[early_stopping])  # Adicionando o Early Stopping

# Avaliando o modelo no conjunto de teste
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test_tensor, Y_test_tensor)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.2f}')
print(f'Test Precision: {test_precision:.2f}')
print(f'Test Recall: {test_recall:.2f}')


predictions = model.predict(teste)
pd.concat([TransNum,pd.DataFrame(predictions, columns = ['is_fraud'])],axis = 1).to_csv('submission.csv',index = False)

print("Fim")