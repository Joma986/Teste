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
    plt.close()
    print('Classifying with Random Forest...')
    clf = RandomForestClassifier(verbose=2)
    print('Fitting...')
    clf.fit(train_features, train_labels)
    print("Done!")
    proba_predictions = clf.predict_proba(test_features)[:, 1]

    # Gráfico 1: Precision vs Threshold
    fig1, ax1 = plt.subplots(figsize=(5, 4), dpi=100)
    pre = []
    for Threshold in np.linspace(0,1,100):
        aux = (proba_predictions >= Threshold).astype('int64')
        precision = precision_score(test_labels, aux)
        pre.append(precision)
    ax1.plot(np.linspace(0,1,100), pre)
    ax1.set_title('Precision vs Threshold')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Precision')
    canvas1 = FigureCanvasTkAgg(fig1, master=sub_frame1)
    canvas1.get_tk_widget().grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
    canvas1.draw()

    # Gráfico 2: Recall vs Threshold
    fig2, ax2 = plt.subplots(figsize=(5, 4), dpi=100)
    rec = []
    for Threshold in np.linspace(0,1,100):
        aux = (proba_predictions >= Threshold).astype('int64')
        recall = recall_score(test_labels, aux)
        rec.append(recall)
    ax2.plot(np.linspace(0,1,100), rec)
    ax2.set_title('Recall vs Threshold')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Recall')
    canvas2 = FigureCanvasTkAgg(fig2, master=sub_frame2)
    canvas2.get_tk_widget().grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
    canvas2.draw()

    # Gráfico 3: Precision-Recall Curve
    fig3, ax3 = plt.subplots(figsize=(5, 4), dpi=100)
    precision, recall, _ = precision_recall_curve(test_labels, proba_predictions)
    ax3.plot(recall, precision)
    ax3.set_title('Precision-Recall Curve')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    canvas3 = FigureCanvasTkAgg(fig3, master=sub_frame3)
    canvas3.get_tk_widget().grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
    canvas3.draw()

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

    CUDA = torch.cuda.is_available()
    global device
    device = torch.device("cuda" if CUDA else "cpu")

    df = pd.read_csv('treino.csv', sep="|")
    teste = pd.read_csv('teste.csv', sep="|")

    X, Y = PreProcessamento(df)

    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print('Training first RBM...')
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

entry1 = customtkinter.CTkEntry(master=frame_login, placeholder_text="Hidden Units")
entry1.grid(row=1, column=0, pady=12, padx=10)

entry2 = customtkinter.CTkEntry(master=frame_login, placeholder_text="Visible Units")
entry2.grid(row=2, column=0, pady=12, padx=10)

entry3 = customtkinter.CTkEntry(master=frame_login, placeholder_text="Contrastive Divergence")
entry3.grid(row=3, column=0, pady=12, padx=10)

entry4 = customtkinter.CTkEntry(master=frame_login, placeholder_text="Epochs")
entry4.grid(row=4, column=0, pady=12, padx=10)

entry5 = customtkinter.CTkEntry(master=frame_login, placeholder_text="Learning Rate")
entry5.grid(row=5, column=0, pady=12, padx=10)

entry6 = customtkinter.CTkEntry(master=frame_login, placeholder_text="Momentum")
entry6.grid(row=6, column=0, pady=12, padx=10)

entry7 = customtkinter.CTkEntry(master=frame_login, placeholder_text="Batch Size")
entry7.grid(row=7, column=0, pady=12, padx=10)

button_login = customtkinter.CTkButton(master=frame_login, text="Executar RBM", command=lambda: threading.Thread(target=LoopRBM).start())
button_login.grid(row=8, column=0, pady=12, padx=10)

button_graph = customtkinter.CTkButton(master=frame_login, text="Executar Random Forest", command=execute_random_forest_thread)
button_graph.grid(row=9, column=0, pady=12, padx=10)

# Frame para o gráfico ao lado
frame_graph = customtkinter.CTkFrame(master=root)
frame_graph.grid(row=0, column=1, rowspan=2, padx=20, pady=20, sticky="nsew")

# Subframes para os gráficos
sub_frame1 = customtkinter.CTkFrame(master=frame_graph)
sub_frame1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

sub_frame2 = customtkinter.CTkFrame(master=frame_graph)
sub_frame2.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

sub_frame3 = customtkinter.CTkFrame(master=frame_graph)
sub_frame3.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")

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
