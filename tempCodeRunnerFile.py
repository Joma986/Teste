# Pre-processamento e extração de características para novo conjunto de dados
# Teste, NumTransacoes = PreProcessamento(teste)
# X_novo_tensor = torch.tensor(Teste.values, dtype=torch.float32)
# novo_dataset = CustomDataset(X_novo_tensor)
# novo_loader = DataLoader(novo_dataset, batch_size=BATCH_SIZE, shuffle=False)

# # Calcular erro de reconstrução para o novo conjunto de dados
# novo_reconstruction_errors = calculate_reconstruction_error(novo_loader, rbm, VISIBLE_UNITS, CUDA)
# novo_anomalies = novo_reconstruction_errors > threshold

# print('Anomalies in the New Test Dataset:')
# print(novo_anomalies.sum())

# pre = pd.DataFrame(novo_anomalies.astype(int), columns=["Fraude"])
# submission = pd.concat([NumTransacoes, pre], axis=1)
# submission.to_csv('submission.csv', index=False)