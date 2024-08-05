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
import customtkinter
from customtkinter import CTkTextbox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import sys
import mplcyberpunk
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm

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

class RedirectText(object):
    def __init__(self, textbox):
        self.textbox = textbox

    def write(self, string):
        self.textbox.insert('end', string)
        self.textbox.yview('end')  # Scroll para o final a cada nova linha

    def flush(self):
        pass

def train_rbm(train_loader, rbm, epochs, batch_size, visible_units, cuda, Threshold=0):
    error = []
    epoch_ = []
    i = 0
    for epoch in range(epochs):
        epoch_error = 0.0
        for batch in train_loader:
            inputs, _ = batch['input'], batch['target']
            inputs = inputs.float()

            if cuda:
                inputs = inputs.to(device)

            batch_error = rbm.contrastive_divergence(inputs)
            epoch_error += batch_error

        epoch_error_cpu = epoch_error.cpu().detach().numpy() if cuda else epoch_error.detach().numpy()
        print(f'Epoch Error (epoch={epoch}): {epoch_error_cpu.mean():.4f}')

        error.append(epoch_error_cpu.mean())
        MenorErro = min(error)
        
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
            melhorepoch = epoch
            MelhoresPesos = rbm.weights
            MelhoresViesesVisiveis = rbm.visible_bias
            MelhoresViesesOcultos = rbm.hidden_bias 
            i = 0

        epoch_.append(epoch)
        update_graph(epoch_, error)
        root.update()

def update_graph(x_data, y_data):
    ax.clear()
    ax.plot(x_data, y_data)
    ax.set_title('Erro por Época')
    ax.set_xlabel('Época')
    ax.set_ylabel('Erro')
    mplcyberpunk.add_glow_effects(ax=ax)
    canvas.draw()

def ClassificaçãoGraficos():
    HIDDEN_UNITS1 = int(entry1.get())
    VISIBLE_UNITS = int(entry2.get())
    CD_K = int(entry3.get())
    EPOCHS = int(entry4.get())
    LEARNING_RATE = float(entry5.get())
    MOMENTUM = float(entry6.get())
    BATCH_SIZE = int(entry7.get())
    plt.close()
    print('Classifying with Random Forest...')
    clf = RandomForestClassifier(verbose=2)
    print('Fitting...')
    clf.fit(train_features, train_labels)
    print("Done!")
    proba_predictions = clf.predict_proba(test_features)[:, 1]

    # Gráfico: Precision e Recall vs Threshold
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    thresholds = np.linspace(0, 1, 100)
    precisions = [precision_score(test_labels, (proba_predictions >= t).astype(int)) for t in thresholds]
    recalls = [recall_score(test_labels, (proba_predictions >= t).astype(int)) for t in thresholds]

    ax.plot(thresholds, precisions, label='Precision', color='orange')
    ax.plot(thresholds, recalls, label='Recall', color='blue')
    ax.set_title('Precision and Recall vs Threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.legend(loc='best')
    ax.grid(True)
    canvas = FigureCanvasTkAgg(fig, master=sub_frame1)
    canvas.get_tk_widget().grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
    canvas.draw()

    # Gráfico 2: Precision-Recall Curve
    fig4, ax4 = plt.subplots(figsize=(5, 4), dpi=100)
    precision, recall, _ = precision_recall_curve(test_labels, proba_predictions)
    ax4.plot(recall, precision)
    ax4.set_title('Precision-Recall Curve')
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    canvas4 = FigureCanvasTkAgg(fig4, master=sub_frame2)
    canvas4.get_tk_widget().grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
    canvas4.draw()

    print('Processing new test data...')
    Teste, NumTransacoes = PreProcessamento(teste)
    Teste = Teste[colunas]
    X_novo_tensor = torch.tensor(Teste.values, dtype=torch.float32)
    novo_dataset = TestDataset(X_novo_tensor)
    novo_loader = DataLoader(novo_dataset, batch_size=BATCH_SIZE, shuffle=False)

    novo_features = extract_features(novo_loader, rbm1, HIDDEN_UNITS1, VISIBLE_UNITS, CUDA)

    # Classificação usando o modelo treinado
    novo_predictions = clf.predict_proba(novo_features)[:, 1]
    novo_predictions = (novo_predictions >= 0.99).astype('int64')

    print('Predictions for the New Test Dataset:')
    print('Fraudes:', novo_predictions.sum())

    pre = pd.DataFrame(novo_predictions, columns=["Fraude"])
    submission = pd.concat([NumTransacoes, pre], axis=1)
    submission.to_csv('submission.csv', index=False)
    print("Fim")


def execute_random_forest_thread():
    threading.Thread(target=ClassificaçãoGraficos).start()

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

customtkinter.set_appearance_mode('dark')  # Modo escuro para o Tkinter
customtkinter.set_default_color_theme('dark-blue')

root = customtkinter.CTk()
root.geometry('1200x800')

def LoopRBM():
    HIDDEN_UNITS1 = int(entry1.get())
    VISIBLE_UNITS = int(entry2.get())
    CD_K = int(entry3.get())
    EPOCHS = int(entry4.get())
    LEARNING_RATE = float(entry5.get())
    MOMENTUM = float(entry6.get())
    BATCH_SIZE = int(entry7.get())
    print(f"Hidden Units: {HIDDEN_UNITS1}")
    print(f"Visible Units: {VISIBLE_UNITS}")
    print(f"Contrastive Divergence: {CD_K}")    
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Momentum: {MOMENTUM}")
    print(f"Batch Size: {BATCH_SIZE}")
    global device,teste,df,CUDA
    CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if CUDA else "cpu")

    df = pd.read_csv('treino.csv', sep="|")
    teste = pd.read_csv('teste.csv', sep="|")

    X, Y = PreProcessamento(df)
    smt = SMOTE()
    X,Y = smt.fit_resample(X,Y)
    global X_train, X_test, y_train, y_test, colunas
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_pool = Pool(X_train, y_train, feature_names=X.columns.to_list())
    test_pool = Pool(X_test, y_test, feature_names=X.columns.to_list())
    model = CatBoostClassifier(iterations=1000, random_seed=0,verbose= 10)
    summary = model.select_features(
    train_pool,
    eval_set=test_pool,
    features_for_select='0-42',
    num_features_to_select=int(entry2.get()),
    steps=3,
    algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
    shap_calc_type=EShapCalcType.Regular,
    train_final_model=True,
    plot=True
    )

    selected_features_indices = summary['selected_features'] 
     # Indices das features selecionadas
    colunas = X_train.columns[selected_features_indices]
    X_train = X_train[colunas]
    X_test = X_test[colunas]

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print('Training first RBM...')
    global rbm1
    rbm1 = RBM(VISIBLE_UNITS, HIDDEN_UNITS1, CD_K, learning_rate=LEARNING_RATE, momentum_coefficient=MOMENTUM, use_cuda=CUDA)
    train_rbm(train_loader, rbm1, EPOCHS, BATCH_SIZE, VISIBLE_UNITS, CUDA, Threshold = 10)

    print('Extracting features from first RBM...')
    global train_features, test_features, train_labels, test_labels
    train_features, train_labels = extract_features(train_loader, rbm1, HIDDEN_UNITS1, VISIBLE_UNITS, CUDA)
    test_features, test_labels = extract_features(test_loader, rbm1, HIDDEN_UNITS1, VISIBLE_UNITS, CUDA)

    X_train_tensor = torch.tensor(train_features, dtype=torch.float32)
    X_test_tensor = torch.tensor(test_features, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_labels, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_labels, dtype=torch.float32)

    train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False)
    print('Done!')

def on_closing():
    root.quit()
    root.destroy()

# Configuração da interface com customtkinter
frame_login = customtkinter.CTkFrame(master=root)
frame_login.grid(row=0, column=0, rowspan=2, padx=20, pady=20, sticky="nsew")

label = customtkinter.CTkLabel(master=frame_login, text='Interface RBM')
label.grid(row=0, column=0, pady=12, padx=10)

# Frame para o formulário de entrada
frame_login = customtkinter.CTkFrame(master=root)
frame_login.grid(row=0, column=0, rowspan=2, padx=20, pady=20, sticky="nsew")

label_title = customtkinter.CTkLabel(master=frame_login, text='Interface RBM')
label_title.grid(row=0, column=0, pady=12, padx=10, columnspan=2)

# Rótulos e campos de entrada
label1 = customtkinter.CTkLabel(master=frame_login, text="Hidden Units:")
label1.grid(row=1, column=0, pady=12, padx=10, sticky="e")
entry1 = customtkinter.CTkEntry(master=frame_login)
entry1.grid(row=1, column=1, pady=12, padx=10)
entry1.insert(0, "43")

label2 = customtkinter.CTkLabel(master=frame_login, text="Visible Units:")
label2.grid(row=2, column=0, pady=12, padx=10, sticky="e")
entry2 = customtkinter.CTkEntry(master=frame_login)
entry2.grid(row=2, column=1, pady=12, padx=10)
entry2.insert(0, "21")

label3 = customtkinter.CTkLabel(master=frame_login, text="Contrastive Divergence:")
label3.grid(row=3, column=0, pady=12, padx=10, sticky="e")
entry3 = customtkinter.CTkEntry(master=frame_login)
entry3.grid(row=3, column=1, pady=12, padx=10)
entry3.insert(0, "1")

label4 = customtkinter.CTkLabel(master=frame_login, text="Epochs:")
label4.grid(row=4, column=0, pady=12, padx=10, sticky="e")
entry4 = customtkinter.CTkEntry(master=frame_login)
entry4.grid(row=4, column=1, pady=12, padx=10)
entry4.insert(0, "200")

label5 = customtkinter.CTkLabel(master=frame_login, text="Learning Rate:")
label5.grid(row=5, column=0, pady=12, padx=10, sticky="e")
entry5 = customtkinter.CTkEntry(master=frame_login)
entry5.grid(row=5, column=1, pady=12, padx=10)
entry5.insert(0, "0.09")

label6 = customtkinter.CTkLabel(master=frame_login, text="Momentum:")
label6.grid(row=6, column=0, pady=12, padx=10, sticky="e")
entry6 = customtkinter.CTkEntry(master=frame_login)
entry6.grid(row=6, column=1, pady=12, padx=10)
entry6.insert(0, "0.9")

label7 = customtkinter.CTkLabel(master=frame_login, text="Batch Size:")
label7.grid(row=7, column=0, pady=12, padx=10, sticky="e")
entry7 = customtkinter.CTkEntry(master=frame_login)
entry7.grid(row=7, column=1, pady=12, padx=10)
entry7.insert(0, "256")

button_login = customtkinter.CTkButton(master=frame_login, text="Executar RBM", command=lambda: threading.Thread(target=LoopRBM).start())
button_login.grid(row=8, column=1, pady=12, padx=10)

button_graph = customtkinter.CTkButton(master=frame_login, text="Executar Random Forest", command=execute_random_forest_thread)
button_graph.grid(row=9, column=1, pady=12, padx=10)

# Frame para o gráfico ao lado
frame_graph = customtkinter.CTkFrame(master=root)
frame_graph.grid(row=0, column=1, rowspan=2, padx=20, pady=20, sticky="nsew")

# Subframes para os gráficos
sub_frame1 = customtkinter.CTkFrame(master=frame_graph)
sub_frame1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

sub_frame2 = customtkinter.CTkFrame(master=frame_graph)
sub_frame2.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

sub_frame3 = customtkinter.CTkFrame(master=frame_graph)
sub_frame3.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

sub_frame4 = customtkinter.CTkFrame(master=frame_graph)
sub_frame4.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

# Configuração inicial do gráfico
plt.style.use('cyberpunk')  # Define o estilo cyberpunk para o matplotlib
fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=sub_frame1)
canvas.get_tk_widget().grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

# Adicionando o Textbox para o log
log_frame = customtkinter.CTkFrame(master=root)
log_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")
log_textbox = CTkTextbox(master=log_frame, width=1200, height=100)
log_textbox.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")

# Redirecionando a saída padrão e de erro para o Textbox
sys.stdout = RedirectText(log_textbox)
sys.stderr = RedirectText(log_textbox)

# Configurando o redimensionamento dos frames
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=3)
root.grid_rowconfigure(0, weight=2)
root.grid_rowconfigure(1, weight=2)
root.grid_rowconfigure(2, weight=1)

# Inicia a interface
root.mainloop()
