
# THIS FILE IS USED TO PERFORM HYPERPARAMETER SWEEPING USING WEIGHTS & BIASES ON DIFFERENT TYPES OF ADVERSARIAL ATTACKS, EACH WITH CONFIGURABLE HYPERPARAMETERS FOR EACH SWEEP. 
# IN ADDITION, ADVERSARIAL EXAMPLES ARE GENERATED, A TRAINING PROCESS WITH CROSS-VALIDATION IS IMPLEMENTED, AND VARIOUS METRICS, SUCH AS AUROC AND ATTACK SUCCESS RATIO, ARE CALCULATED TO EVALUATE THE ROBUSTNESS AND EFFECTIVENESS OF THE MODEL IN A COMPARATIVE MANNER.


import numpy as np
import pandas as pd
import os
import re
import os.path
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import torch
from tqdm.notebook import tqdm
from torch import nn, optim
import torch.nn as nn
from torcheval.metrics import BinaryAUROC
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import  balanced_accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import torchattacks
import shap
from datetime import datetime

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

print("torchattacks %s"%(torchattacks.__version__))
pd.set_option('display.max_columns', 500)
shap.initjs()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
#wandb.login(key='')

DRIAMS_B = 'DRIAMS-B'
DRIAMS_C = 'DRIAMS-C'
DRIAMS_D = 'DRIAMS-D'
datasets = [DRIAMS_B, DRIAMS_C, DRIAMS_D]
dataset = datasets[2]
organism = 'Klebsiella pneumoniae'
antimicrobials = ['Cefepime'] #, 'Tobramycin']
antimicrobial = antimicrobials[0]


# Cargar datasets
df = pd.read_csv(f'dataframes/dataframe_{dataset}_{organism}_{antimicrobial}.csv')
df_test = pd.read_csv('dataframes/test.csv')
df_test_driams = pd.read_csv(f'dataframes/dataframe_{DRIAMS_C}_{organism}_{antimicrobial}.csv')
df_test_gm = pd.read_csv('dataframes/gm_data.csv')
df_test_ryc = pd.read_csv('dataframes/ryc_data.csv')

def string_a_lista(cadena):
    cadena_sin_corchetes = re.sub(r'\[|\]', '', cadena)
    lista_de_floats = [float(x) for x in cadena_sin_corchetes.split(',')]
    return lista_de_floats

# Convertir cada string de la columna 'lista_strings' a una lista de floats
df['MALDI_binned'] = df['MALDI_binned'].apply(string_a_lista)

# GM + RyC en conjunto (lo utilizo para que la metrica a monitorizar en el sweep del wandb sea la auc conjunta de ambos datasets)
df_test['MALDI_binned'] = df_test['MALDI_binned'].apply(string_a_lista)
# DRIAMS TEST
df_test_driams['MALDI_binned'] = df_test_driams['MALDI_binned'].apply(string_a_lista)
# GM TEST
df_test_gm['MALDI_binned'] = df_test_gm['MALDI_binned'].apply(string_a_lista)
# RyC TEST
df_test_ryc['MALDI_binned'] = df_test_ryc['MALDI_binned'].apply(string_a_lista)

max_abs = df['MALDI_binned'].apply(max).max()
min_abs = df['MALDI_binned'].apply(min).min()

df['MALDI_binned'] = df['MALDI_binned'].apply(lambda x: [(val - min_abs) / (max_abs - min_abs) for val in x])
df_train, df_test_intrahospital = train_test_split(df, test_size=0.2, random_state=3, stratify = df[antimicrobial])

df_test['MALDI_binned'] = df_test['MALDI_binned'].apply(lambda x: [(val - min_abs) / (max_abs - min_abs) for val in x])
df_test_driams['MALDI_binned'] = df_test_driams['MALDI_binned'].apply(lambda x: [(val - min_abs) / (max_abs - min_abs) for val in x])
df_test_gm['MALDI_binned'] = df_test_gm['MALDI_binned'].apply(lambda x: [(val - min_abs) / (max_abs - min_abs) for val in x])
df_test_ryc['MALDI_binned'] = df_test_ryc['MALDI_binned'].apply(lambda x: [(val - min_abs) / (max_abs - min_abs) for val in x])

# Hago un shuffle de los conjuntos de test (por si acaso)
df_test = df_test.sample(frac=1).reset_index(drop=True)
df_test_driams = df_test_driams.sample(frac=1).reset_index(drop=True)
df_test_gm = df_test_gm.sample(frac=1).reset_index(drop=True)
df_test_ryc = df_test_ryc.sample(frac=1).reset_index(drop=True)

def normalize(value, max, min):
  return ((value - min) / (max - min))

def invertir_normalizacion(normalizado, min_abs, max_abs):
    original = normalizado * (max_abs - min_abs) + min_abs
    return original

def generate_adversarial_examples(data, labels, adversary_attack):
    data.requires_grad = True  # Habilitar el gradiente para los datos de entrada
    adv_data = adversary_attack(data, labels)
    return adv_data

def generate_adversarial_dataloader(dataloader, adversary_attack):
  # Generar ejemplos adversariales para el conjunto de prueba
  adversarial_examples = []
  adversarial_labels = []
  batch_size = dataloader.batch_size

  for data, labels in dataloader:
      data, labels = data.to(device), labels.to(device)
      adv_data = generate_adversarial_examples(data, labels, adversary_attack)
      adversarial_examples.append(adv_data.cpu().detach().numpy())
      adversarial_labels.append(labels.cpu().detach().numpy())

  # Convertir la lista de ejemplos adversariales y etiquetas en tensores
  adversarial_examples = torch.tensor(np.concatenate(adversarial_examples, axis=0))
  adversarial_labels = torch.tensor(np.concatenate(adversarial_labels, axis=0))

  # Crear un nuevo conjunto de datos adversariales
  adversarial_dataset = TensorDataset(adversarial_examples, adversarial_labels)

  # Crear un DataLoader para el conjunto de datos adversariales
  adversarial_dataloader = DataLoader(adversarial_dataset, batch_size=batch_size)

  return adversarial_dataloader

# Early Stopping class, obtained from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

class EarlyStopping:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.early_stop = False

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# BINARY AUROC
def auroc_mlp(model, validloader):
  model.eval()
  target = torch.tensor([]).to(device)
  predicted = torch.tensor([]).to(device)

  metric = BinaryAUROC(num_tasks=1).to(device)
  with torch.no_grad():
    for data, labels in validloader:
      data, labels = data.to(device), labels.to(device)
      out = model.forward(data.view(data.shape[0], -1))
      predicted = torch.cat((predicted, out), 0)
      target = torch.cat((target, labels), 0)
        
  predicted = predicted.view(predicted.shape[0])
  target = target.view(target.shape[0])
  metric.update(predicted, target)
  auc = metric.compute()
  model.train()
  return auc.cpu()

# BINARY AUROC
def adversarial_auroc_mlp(model, validloader, attack):
  target = torch.tensor([]).to(device)
  predicted = torch.tensor([]).to(device)

  metric = BinaryAUROC(num_tasks=1)
  for data, labels in validloader:
      data, labels = data.to(device), labels.to(device)
      out = model.forward(data.view(data.shape[0], -1))
      predicted = torch.cat((predicted, out), 0)
      target = torch.cat((target, labels), 0)
  predicted = predicted.view(predicted.shape[0])
  target = target.view(target.shape[0])
  metric.update(predicted, target)
  auc = metric.compute()
  return auc

class MLP(nn.Module):
    def __init__(self, dimx, num_labels, hidden_layers, dropout=None, batch_norm=False):
      super().__init__()

      self.num_hidden_layers = len(hidden_layers) + 1
      self.layers = [dimx] + hidden_layers + [num_labels]

      self.outputs = nn.ModuleList()
      for i in range(0, self.num_hidden_layers):
        self.outputs.append(nn.Linear(self.layers[i], self.layers[i+1]))

      self.relu = nn.ReLU()

      self.batch_norm = batch_norm
      self.batch_norm_layers = nn.ModuleList()
      if self.batch_norm:
        for b in range(0, self.num_hidden_layers):
          self.batch_norm_layers.append(nn.BatchNorm1d(self.layers[b + 1]))

      self.dropouts = nn.ModuleList()
      if dropout != None:
        for d in range(0, len(dropout)):
          self.dropouts.append(nn.Dropout(p=dropout[d]))

      self.sigmoid = nn.Sigmoid()

    def forward(self, x):

      start_dropout = None
      if len(self.dropouts) != 0:
        start_dropout = self.num_hidden_layers - 1 - len(self.dropouts)

      for l in range(0, self.num_hidden_layers - 1):
        x = self.outputs[l](x)
        if self.batch_norm:
          x = self.batch_norm_layers[l](x)
        x = self.relu(x)
        if start_dropout != None and l >= start_dropout:
          x = self.dropouts[l - start_dropout](x)

      x = self.outputs[-1](x)
      x = self.sigmoid(x)
      return x.view(-1)
    
class MLP_extended(MLP):
  def __init__(self, hidden_layers, dimx=6000, num_labels=1, dropout=None, batch_norm=False, lr=0.001, n_epochs=20, attack_name = None):

    super().__init__(dimx, num_labels, hidden_layers, dropout, batch_norm)
    self.lr = lr
    self.epochs = n_epochs
    self.optim = optim.Adam(self.parameters(), self.lr, weight_decay=1e-4)

    self.criterion = nn.BCELoss()

    # Crear nombre de la carpeta basado en la hora fecha actual y el ataque realizado
    current_date = datetime.now().strftime('%Y-%m-%d')
    folder_name = f"model_saves_complete_{attack_name}_{current_date}"
    self.save_folder = os.path.join(os.getcwd(), folder_name)

    # Crear la carpeta si no existe
    os.makedirs(self.save_folder, exist_ok=True)

    # LOSS EVOLUTION
    self.loss_during_training = []
    self.adv_loss_during_training = []
    self.valid_loss_during_training = []
    self.adv_valid_loss_during_training = []

  def train_valid_loop(self, trainloader, valloader):

    best_val_loss = float('inf')
    best_model_path = os.path.join(self.save_folder, 'best_model.pth')
    self.to(device)

    for e in range(int(self.epochs)):
      running_loss = 0.
      for data, labels in trainloader:
        data, labels = data.to(device), labels.to(device)
        self.optim.zero_grad()
        out = self.forward(data.view(data.shape[0], -1))
        loss = self.criterion(out.view(-1), labels)
        running_loss += loss.item()
        loss.backward()
        self.optim.step()

      self.loss_during_training.append(running_loss/len(trainloader))
      wandb.log({"training_loss":running_loss/len(trainloader)})

      torch.save(self.state_dict(), os.path.join(self.save_folder, f'model_weights_{e}.pth'))

      with torch.no_grad():
        self.eval()
        val_run_loss = 0.
        for validdata, validlabels in valloader:
          validdata, validlabels = validdata.to(device), validlabels.to(device)
          validout = self.forward(validdata.view(validdata.shape[0], -1))
          validloss = self.criterion(validout.view(-1), validlabels)
          val_run_loss += validloss.item()

        self.valid_loss_during_training.append(val_run_loss/len(valloader))

        wandb.log({"validation_loss":val_run_loss/len(valloader)})

      if self.valid_loss_during_training[-1] < best_val_loss:
        best_val_loss = self.valid_loss_during_training[-1]
        torch.save(self.state_dict(), best_model_path)
        print(f'Guardado el mejor modelo en la época {e} con pérdida de validación {best_val_loss:.4f}')

      self.train()

      # if e % 1 == 0:
      #   print("training loss after %d epochs: %f" %(e, self.loss_during_training[-1]))
      #   print("validation loss after %d epochs: %f\n" %(e, self.valid_loss_during_training[-1]))

    if os.path.exists(best_model_path):
      self.load_state_dict(torch.load(best_model_path))
      print('Loaded best model from epoch with lowest validation loss at the end of training.')

  def train_valid_loop_with_adversarial(self, trainloader, valloader, adversary_attack, adversary_partition):
    # patience = 250
    best_val_loss = float('inf')
    best_model_path = os.path.join(self.save_folder, 'best_model.pth')
    self.to(device)

    for e in range(int(self.epochs)):
      running_loss = 0.
      adv_running_loss = 0.
        
      # Generate a subset of 20% for adversarial training
      all_indices = list(range(len(trainloader.dataset)))
      adv_indices = np.random.choice(all_indices, size=int(adversary_partition * len(all_indices)), replace=False)
      adv_sampler = Subset(trainloader.dataset, adv_indices)
      dataloader_to_adv = DataLoader(adv_sampler, batch_size=trainloader.batch_size, shuffle=True)

      # Training with benign examples
      for data, labels in trainloader:
          data, labels = data.to(device), labels.to(device)
          self.optim.zero_grad()
          out = self.forward(data.view(data.shape[0], -1))
          loss = self.criterion(out.view(-1), labels)
          running_loss += loss.item()
          loss.backward()
          self.optim.step()

      # Generate adversarial examples and train with them
      #print("Adversarial training currently undergoing...")
      for data, labels in dataloader_to_adv:
          data, labels = data.to(device), labels.to(device)
          adv_data = generate_adversarial_examples(data, labels, adversary_attack)
          self.optim.zero_grad()
          adv_out = self.forward(adv_data.view(adv_data.shape[0], -1))
          loss = self.criterion(adv_out.view(-1), labels)
          adv_running_loss += loss.item()
          loss.backward()
          self.optim.step()

      #print("Adversarial training epoch finished succesfully.")
      self.loss_during_training.append(running_loss / len(trainloader))
      self.adv_loss_during_training.append(adv_running_loss / len(trainloader))

      #wandb.log({"training_loss":running_loss/len(trainloader)})
      #wandb.log({"adv_training_loss":adv_running_loss/len(trainloader)})

      #torch.save(self.state_dict(), os.path.join(self.save_folder, f'model_weights_{e}.pth'))

      with torch.no_grad():
        self.eval()
        val_run_loss = 0.
        adv_val_run_loss = 0.
          
        for validdata, validlabels in valloader:
          validdata, validlabels = validdata.to(device), validlabels.to(device)
          validout = self.forward(validdata.view(validdata.shape[0], -1))
          validloss = self.criterion(validout.view(-1), validlabels)
          val_run_loss += validloss.item()

      for validdata, validlabels in valloader:
        validdata, validlabels = validdata.to(device), validlabels.to(device)
        adv_validdata = generate_adversarial_examples(validdata, validlabels, adversary_attack)
        adv_validout = self.forward(adv_validdata.view(adv_validdata.shape[0], -1))
        adv_validloss = self.criterion(adv_validout.view(-1), validlabels)
        adv_val_run_loss += adv_validloss.item()

      self.valid_loss_during_training.append(val_run_loss/len(valloader))
      self.adv_valid_loss_during_training.append(adv_val_run_loss/len(valloader))

      #wandb.log({"validation_loss":val_run_loss/len(valloader)})
      #wandb.log({"adv_validation_loss":adv_val_run_loss/len(valloader)})

      if self.valid_loss_during_training[-1] < best_val_loss:
        best_val_loss = self.valid_loss_during_training[-1]
        torch.save(self.state_dict(), best_model_path)
        print(f'Guardado el mejor modelo en la época {e} con pérdida de validación {best_val_loss:.4f}')

      self.train()

      # if e % 1 == 0:
      #     print("training loss after %d epochs: %f" %(e, self.loss_during_training[-1]))
      #     print("validation loss after %d epochs: %f\n" %(e, self.valid_loss_during_training[-1]))


    if os.path.exists(best_model_path):
      self.load_state_dict(torch.load(best_model_path))
      print('Loaded best model from epoch with lowest validation loss at the end of training.')

  def accuracy(self, dataloader, types="Test"):
    target = []
    predicted = []
    self.to(device)

    with torch.no_grad():
      self.eval()
      for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        out = self.forward(data.view(data.shape[0], -1))
        #out = self.softmax(out)
        target.extend(labels.cpu().numpy())
        predicted.extend(out.cpu().numpy())
      target = np.array(target)
      predicted = np.where(np.array(predicted) < 0.5, 0, 1)
      print("TARGET", target)
      print("PREDICTED", predicted)
      # Convertir numpy arrays a tensores
      target_tensor = torch.tensor(target)
      predicted_tensor = torch.tensor(predicted)
      print("DIFFERENCES: ", torch.sum(target_tensor != predicted_tensor))
      accuracy = balanced_accuracy_score(target, predicted)

      self.train()
      return accuracy, target, predicted

  def attack_success_ratio(self, adversarial_dataloader):
    attack_success_count = 0
    total_samples = 0

    with torch.no_grad():
      self.eval()
      for inputs, labels in adversarial_dataloader:
          inputs, labels = inputs.to(device), labels.to(device)
          # Calcular las probabilidades de pertenencia a la clase 1
          probabilities = self.forward(inputs).squeeze()  # Obtener las probabilidades (salida única)

          # Convertir las probabilidades en predicciones (0 o 1)
          predicted = (probabilities >= 0.5).int()  # Clasificar como clase 1 si la probabilidad >= 0.5

          # Contar ejemplos clasificados incorrectamente como resultado del ataque adversarial
          attack_success_count += (predicted != labels).sum().item()
          total_samples += labels.size(0)

      attack_success_rate = attack_success_count / total_samples
      self.train()  # Volver al modo de entrenamiento
      return attack_success_rate

  def fooling_ratio(self, dataloader, adversarial_dataloader):
      num_samples = len(dataloader.dataset)
      num_fooled = 0
      with torch.no_grad():
          self.eval()
          for original_batch, adversarial_batch in zip(dataloader, adversarial_dataloader):
              original_inputs, _ = original_batch
              adversarial_inputs, _ = adversarial_batch

              # Move tensors to device
              original_inputs = original_inputs.to(device)
              adversarial_inputs = adversarial_inputs.to(device)

              # Predictions on original and adversarial inputs
              original_probs = self.forward(original_inputs).squeeze()
              adversarial_probs = self.forward(adversarial_inputs).squeeze()

              # Convert probabilities to predictions (assuming threshold of 0.5)
              original_predicted = (original_probs >= 0.5).int()
              adversarial_predicted = (adversarial_probs >= 0.5).int()

              # Calculate number of differences in predictions
              num_fooled += torch.sum(original_predicted != adversarial_predicted).item()

              print("ORIGINAL PREDS:", original_predicted)
              print("ADVERSARIAL PREDS:", adversarial_predicted)
              print("DIFFERENCES: ", torch.sum(original_predicted != adversarial_predicted).item())

      self.train()  # Volver al modo de entrenamiento
      return num_fooled / num_samples
  

sweep_config = {
    'method': 'grid'
    }

metric = {
    'name': 'auc_test',
    'goal': 'maximize'
    }

sweep_config['metric'] = metric


parameters_dict = {
    'hidden_layers': {
        'values': [[128, 128, 128]] 
        },
    'dropout': {
          'values': [[0.1, 0.1]]
        },
    'epochs':{
          'value':350
    },
    'lr': {
        'values': [1e-3]
      },
    'batch_norm': {
        'values': [False]
    },
    'antimicrobial': {
        'value': 'Cefepime'
      },
    'smote': {
        'values': [False]
      },
    'attack': {
        'values': ['PGD_Linf'] 
      },    
    'epsilon': {
        'values': [0.0001, 0.00025, 0.0005, 0.00075, 0.001] 
      },
    'alpha': {
        'values': [0.000175, 0.000375, 0.000625, 0.000875, 0.001]  
      },    
    'steps': {
        'values': [10, 20]
      },
    'random_start': {
        'values': [True, False]
      },
    'adversary_partition': {
        'values': [0.2, 1]
      }
}
    
sweep_config['parameters'] = parameters_dict

def do_sweep_mlp(config=None):

  with wandb.init(config=config, project = 'malditof', anonymous="allow"):

      torch.manual_seed(42)
      torch.cuda.manual_seed_all(42)
      np.random.seed(42)
      torch.backends.cudnn.deterministic = True

      config = wandb.config

      antimicrobial = config.antimicrobial
      lr = config.lr
      epochs = config.epochs
      dropout = config.dropout
      hidden_layers = config.hidden_layers
      batch_norm = config.batch_norm

      attack = config.attack
      adversary_partition = config.adversary_partition

      if attack != 'GN':
        epsilon = config.epsilon
        epsilon = normalize(epsilon, max_abs, min_abs)

        if attack == 'FAB':
          alpha_max = config.alpha_max
          alpha_max = normalize(alpha_max, max_abs, min_abs)

        else:
            if attack != 'FGSM':
              alpha = config.alpha
              alpha = normalize(alpha, max_abs, min_abs)
      else:
        std = config.std

      if attack != 'GN' and attack != 'FGSM' and attack != 'FFGSM':
        steps = config.steps

      if attack == 'EOTPGD':
        eot_iter = config.eot_iter

      elif attack == 'FAB':
        n_restarts = config.n_restarts
        eta = config.eta
        beta = config.beta
        norm = config.norm
        iterations = config.iterations
        seed = 42
        n_classes = 2

      elif attack == 'PGD_Linf':
        random_start = config.random_start

      use_smote = config.smote
      batch_size = 64

      # Inicializa la validación cruzada
      k = 5
      kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

      # Lista para almacenar los resultados de la métrica de interés en cada iteración de validación cruzada

      # Sobre conjunto benigno

      auc_scores_train = []
      auc_scores_val = []
      auc_scores_test = []
      auc_scores_test_intrahospital = []
      auc_scores_test_driams = []
      auc_scores_test_gm = []
      auc_scores_test_ryc = []

      accuracy_scores_train = []
      accuracy_scores_val = []
      accuracy_scores_test = []
      accuracy_scores_test_intrahospital = []
      accuracy_scores_test_driams = []
      accuracy_scores_test_gm = []
      accuracy_scores_test_ryc = []

      # Sobre conjunto adversario

      adv_auc_scores_train = []
      adv_auc_scores_val = []
      adv_auc_scores_test = []
      adv_auc_scores_test_intrahospital = []
      adv_auc_scores_test_driams = []
      adv_auc_scores_test_gm = []
      adv_auc_scores_test_ryc = []

      adv_accuracy_scores_train = []
      adv_accuracy_scores_val = []
      adv_accuracy_scores_test = []
      adv_accuracy_scores_test_intrahospital = []
      adv_accuracy_scores_test_driams = []
      adv_accuracy_scores_test_gm = []
      adv_accuracy_scores_test_ryc = []

      # Para monitorizar el Attack Succes Ratio: 
      # Se calcula sobre el adversarial_dataloader porque ahi tienes las perturbaciones ya generadas con las salidas correctas, y al hacer forward de las perturbaciones puedes comparar con las salidas correctas
      # porque ya las tienes en el dataloader
      attack_success_ratio_scores_train = []
      attack_success_ratio_scores_val = []
      attack_success_ratio_scores_test = []
      attack_success_ratio_scores_test_intrahospital = []
      attack_success_ratio_scores_test_driams = []
      attack_success_ratio_scores_test_gm = []
      attack_success_ratio_scores_test_ryc = []

      # Para monitorizar el Fooling Ratio:
      fooling_ratio_scores_train = []
      fooling_ratio_scores_val = []
      fooling_ratio_scores_test = []
      fooling_ratio_scores_test_intrahospital = []
      fooling_ratio_scores_test_driams = []
      fooling_ratio_scores_test_gm = []
      fooling_ratio_scores_test_ryc = []

      # Generamos el testloader (para luego poder evaluar los resultados en el crossval). Puesto que no es necesario volverlo a generar en cada iteración (como sí ocurre con train y val), se hace fuera

      # TEST SET (GENERAMOS UN CONJUNTO DE TEST PARA DRIAMS (DIFERENTE HOSPITAL MISMO PAÍS) Y OTROS DOS SEPARADOS PARA GM Y RyC)

      # TEST (GM + RyC)

      X_test = df_test['MALDI_binned'].values.tolist()

      y_test = df_test[antimicrobial].values.astype('float32')

      tensor_X_test = torch.tensor(X_test, dtype=torch.float32)
      tensor_X_test = tensor_X_test.view(tensor_X_test.shape[0], -1)

      tensor_y_test = torch.tensor(y_test, dtype=torch.float32)

      dataset = TensorDataset(tensor_X_test, tensor_y_test)
      testloader = DataLoader(dataset, batch_size=batch_size)

      # INTRA-HOSPITAL
      
      X_test = df_test_intrahospital['MALDI_binned'].values.tolist()

      y_test = df_test_intrahospital[antimicrobial].values.astype('float32')

      tensor_X_test = torch.tensor(X_test, dtype=torch.float32)
      tensor_X_test = tensor_X_test.view(tensor_X_test.shape[0], -1)

      tensor_y_test = torch.tensor(y_test, dtype=torch.float32)

      dataset_intrahospital = TensorDataset(tensor_X_test, tensor_y_test)
      testloader_intrahospital = DataLoader(dataset_intrahospital, batch_size=batch_size)

      # DRIAMS
      X_test = df_test_driams['MALDI_binned'].values.tolist()

      y_test = df_test_driams[antimicrobial].values.astype('float32')

      tensor_X_test = torch.tensor(X_test, dtype=torch.float32)
      tensor_X_test = tensor_X_test.view(tensor_X_test.shape[0], -1)

      tensor_y_test = torch.tensor(y_test, dtype=torch.float32)

      dataset_driams = TensorDataset(tensor_X_test, tensor_y_test)
      testloader_driams = DataLoader(dataset_driams, batch_size=batch_size)

      # GM
      X_test = df_test_gm['MALDI_binned'].values.tolist()

      y_test = df_test_gm[antimicrobial].values.astype('float32')

      tensor_X_test = torch.tensor(X_test, dtype=torch.float32)
      tensor_X_test = tensor_X_test.view(tensor_X_test.shape[0], -1)

      tensor_y_test = torch.tensor(y_test, dtype=torch.float32)

      dataset_gm = TensorDataset(tensor_X_test, tensor_y_test)
      testloader_gm = DataLoader(dataset_gm, batch_size=batch_size)

      # RyC
      X_test = df_test_ryc['MALDI_binned'].values.tolist()

      y_test = df_test_ryc[antimicrobial].values.astype('float32')

      tensor_X_test = torch.tensor(X_test, dtype=torch.float32)
      tensor_X_test = tensor_X_test.view(tensor_X_test.shape[0], -1)

      tensor_y_test = torch.tensor(y_test, dtype=torch.float32)

      dataset_ryc = TensorDataset(tensor_X_test, tensor_y_test)
      testloader_ryc = DataLoader(dataset_ryc, batch_size=batch_size)

      # Itera sobre cada fold
      for fold_idx, (train_index, valid_index) in enumerate(kf.split(df_train, df_train[antimicrobial])):
        print(f"Fold {fold_idx + 1}/{k}")

        df_train_iter = df_train.iloc[train_index]
        df_valid = df_train.iloc[valid_index]

        if use_smote:
          # Inicialización de SMOTE
          smote = SMOTE(random_state = 42)

        # TRAINING SET
        X = df_train_iter.drop(antimicrobials, axis=1).values.tolist()
        y = df_train_iter[antimicrobial].values.astype('float32')

        # Generación de tensores
        tensor_X_train = torch.tensor(X, dtype=torch.float32)
        tensor_X_train = tensor_X_train.view(tensor_X_train.shape[0], -1)

        tensor_y_train = torch.tensor(y, dtype=torch.float32)

        if use_smote:
          # Aplicación de SMOTE al TRAINING SET
          X_train_np, y_train_np = smote.fit_resample(tensor_X_train, tensor_y_train)

          tensor_X_train = torch.tensor(X_train_np, dtype=torch.float32)
          tensor_y_train = torch.tensor(y_train_np, dtype=torch.float32)

        # Generación de Dataset de PyTorch para TRAINING SET
        dataset = TensorDataset(tensor_X_train, tensor_y_train)

        # Generación de DataLoader sobre TRAINING SET
        trainloader = DataLoader(dataset, batch_size=batch_size)

        # Generación de SHAP-DataLoader para estimar SHAP VALUES
        trainloader_shap = DataLoader(dataset, batch_size=1)

        # VALIDATION SET
        X = df_valid.drop(antimicrobials, axis=1).values.tolist()
        y = df_valid[antimicrobial].values.astype('float32')

        # Generación de tensores
        tensor_X_val = torch.tensor(X, dtype=torch.float32)
        tensor_X_val = tensor_X_val.view(tensor_X_val.shape[0], -1)

        tensor_y_val = torch.tensor(y, dtype=torch.float32)

        # Generación de Dataset de PyTorch para VALIDATION SET
        dataset = TensorDataset(tensor_X_val, tensor_y_val)

        # Generación de DataLoader sobre VALIDATION SET
        validloader = DataLoader(dataset, batch_size=batch_size)

        # Inicialización del MLP
        multi_layer = MLP_extended(hidden_layers, n_epochs=epochs, lr=lr, dropout = dropout, batch_norm = batch_norm, attack_name=attack)
        multi_layer.to(device)

        # Inicialización del Ataque Adversario según la configuración

        if attack == 'PGD_Linf':
          adversary_attack = torchattacks.PGD(multi_layer, eps=epsilon, alpha=alpha, steps=steps, random_start=random_start)

        elif attack == 'FGSM':
          adversary_attack = torchattacks.FGSM(multi_layer, eps=epsilon)

        elif attack == 'FFGSM':
          adversary_attack = torchattacks.FFGSM(multi_layer, eps=epsilon, alpha=alpha)

        elif attack == 'RFGSM':
          adversary_attack = torchattacks.RFGSM(multi_layer, eps=epsilon, alpha=alpha, steps=steps)

        elif attack == 'BIM':
          adversary_attack = torchattacks.BIM(multi_layer, eps=epsilon, alpha=alpha, steps=steps)

        elif attack == 'TPGD':
          adversary_attack = torchattacks.TPGD(multi_layer, eps=epsilon, alpha=alpha, steps=steps)

        elif attack == 'EOTPGD':
          adversary_attack = torchattacks.EOTPGD(multi_layer, eps=epsilon, alpha=alpha, steps=steps, eot_iter=eot_iter)

        elif attack == 'FAB':
          adversary_attack = torchattacks.FAB(multi_layer, norm = norm, eps = epsilon, steps = iterations, n_restarts = n_restarts, alpha_max = alpha_max, eta = eta, beta = beta, seed=seed, n_classes=n_classes)

        elif attack == 'GN':
          adversary_attack = torchattacks.GN(multi_layer, std=std)

        elif attack == 'Multi-Attack':
          atk1 = None
          atk2 = None
          adversary_attack = torchattacks.MultiAttack([atk1, atk2])

        else:
          print("No Adversarial Attack has been defined. Stopping execution.")
          break

        # Se entrena el modelo con el conjunto de datos de siempre de MALDI-TOF
        multi_layer.train_valid_loop_with_adversarial(trainloader, validloader, adversary_attack, adversary_partition=adversary_partition)

        # Loggear métricas de balanced accuracy y binary AUC (TRAIN, VAL Y TEST BENIGNOS)
        acc_train,_,_ = multi_layer.accuracy(trainloader, "Train")
        acc_val,_,_ = multi_layer.accuracy(validloader, "Validation")

        # GM + RyC
        acc_test, true_labels, predicted_labels = multi_layer.accuracy(testloader)
        # INTRA-HOSPITAL
        acc_test_intrahospital, _, _ = multi_layer.accuracy(testloader_intrahospital)
        # DRIAMS
        acc_test_driams, _, _ = multi_layer.accuracy(testloader_driams)
        # GM
        acc_test_gm, _, _ = multi_layer.accuracy(testloader_gm)
        # RyC
        acc_test_ryc, _, _ = multi_layer.accuracy(testloader_ryc)

        auc_train = auroc_mlp(multi_layer, trainloader)
        auc_val = auroc_mlp(multi_layer, validloader)

        # GM + RyC
        auc_test = auroc_mlp(multi_layer, testloader)
        # INTRAHOSPITAL
        auc_test_intrahospital = auroc_mlp(multi_layer, testloader_intrahospital)
        # DRIAMS
        auc_test_driams = auroc_mlp(multi_layer, testloader_driams)
        # GM
        auc_test_gm = auroc_mlp(multi_layer, testloader_gm)
        # RyC
        auc_test_ryc = auroc_mlp(multi_layer, testloader_ryc)

        accuracy_scores_train.append(acc_train)
        auc_scores_train.append(auc_train)

        accuracy_scores_val.append(acc_val)
        auc_scores_val.append(auc_val)

        accuracy_scores_test.append(acc_test)
        accuracy_scores_test_intrahospital.append(acc_test_intrahospital)
        accuracy_scores_test_driams.append(acc_test_driams)
        accuracy_scores_test_gm.append(acc_test_gm)
        accuracy_scores_test_ryc.append(acc_test_ryc)

        auc_scores_test.append(auc_test)
        auc_scores_test_intrahospital.append(auc_test_intrahospital)
        auc_scores_test_driams.append(auc_test_driams)
        auc_scores_test_gm.append(auc_test_gm)
        auc_scores_test_ryc.append(auc_test_ryc)

  #      wandb.log({"acc_train":acc_train})
  #      wandb.log({"acc_val":acc_val})
  #      wandb.log({"acc_test":acc_test}

  #      wandb.log({"auc_train":auc_train})
  #      wandb.log({"auc_val":auc_val})
  #      wandb.log({"auc_test":auc_test})

        # Generación de dataloaders adversarios (para ver la robustez adversaria del modelo)
        adversarial_trainloader = generate_adversarial_dataloader(trainloader, adversary_attack)
        adversarial_validloader = generate_adversarial_dataloader(validloader, adversary_attack)

        adversarial_testloader = generate_adversarial_dataloader(testloader, adversary_attack)
        adversarial_testloader_intrahospital = generate_adversarial_dataloader(testloader_intrahospital, adversary_attack)
        adversarial_testloader_driams = generate_adversarial_dataloader(testloader_driams, adversary_attack)
        adversarial_testloader_gm = generate_adversarial_dataloader(testloader_gm, adversary_attack)
        adversarial_testloader_ryc = generate_adversarial_dataloader(testloader_ryc, adversary_attack)

        # Loggear métricas de acuracy y AUC (TRAIN Y VAL ADVERSARIOS)

        # Train
        adv_acc_train,_,_ = multi_layer.accuracy(adversarial_trainloader, "Adversarial Train")
        adv_auc_train = auroc_mlp(multi_layer, adversarial_trainloader)
        attack_success_ratio_train = multi_layer.attack_success_ratio(adversarial_trainloader)
        fooling_ratio_train = multi_layer.fooling_ratio(trainloader, adversarial_trainloader)

        # Validation
        adv_acc_val,_,_ = multi_layer.accuracy(adversarial_validloader, "Adversarial Validation")
        adv_auc_val = auroc_mlp(multi_layer, adversarial_validloader)
        attack_success_ratio_val = multi_layer.attack_success_ratio(adversarial_validloader)
        fooling_ratio_val = multi_layer.fooling_ratio(validloader, adversarial_validloader)

        # Test
        # GM + RyC
        adv_acc_test,_,_ = multi_layer.accuracy(adversarial_testloader, "Adversarial Test")
        adv_auc_test = auroc_mlp(multi_layer, adversarial_testloader)
        attack_success_ratio_test = multi_layer.attack_success_ratio(adversarial_testloader)
        fooling_ratio_test = multi_layer.fooling_ratio(testloader, adversarial_testloader)

        # INTRAHOSPITAL
        adv_acc_test_intrahospital,_,_ = multi_layer.accuracy(adversarial_testloader_intrahospital, "Adversarial Test")
        adv_auc_test_intrahospital = auroc_mlp(multi_layer, adversarial_testloader_intrahospital)
        attack_success_ratio_test_intrahospital = multi_layer.attack_success_ratio(adversarial_testloader_intrahospital)
        fooling_ratio_test_intrahospital = multi_layer.fooling_ratio(testloader_intrahospital, adversarial_testloader_intrahospital)      

        # DRIAMS
        adv_acc_test_driams,_,_ = multi_layer.accuracy(adversarial_testloader_driams, "Adversarial Test DRIAMS")
        adv_auc_test_driams = auroc_mlp(multi_layer, adversarial_testloader_driams)
        attack_success_ratio_test_driams = multi_layer.attack_success_ratio(adversarial_testloader_driams)
        fooling_ratio_test_driams = multi_layer.fooling_ratio(testloader_driams, adversarial_testloader_driams)

        # GM
        adv_acc_test_gm,_,_ = multi_layer.accuracy(adversarial_testloader_gm, "Adversarial Test GM")
        adv_auc_test_gm = auroc_mlp(multi_layer, adversarial_testloader_gm)
        attack_success_ratio_test_gm = multi_layer.attack_success_ratio(adversarial_testloader_gm)
        fooling_ratio_test_gm = multi_layer.fooling_ratio(testloader_gm, adversarial_testloader_gm)

        # RyC
        adv_acc_test_ryc,_,_ = multi_layer.accuracy(adversarial_testloader_ryc, "Adversarial Test RyC")
        adv_auc_test_ryc = auroc_mlp(multi_layer, adversarial_testloader_ryc)
        attack_success_ratio_test_ryc = multi_layer.attack_success_ratio(adversarial_testloader_ryc)
        fooling_ratio_test_ryc = multi_layer.fooling_ratio(testloader_ryc, adversarial_testloader_ryc)

        # Adversarial Train
        adv_accuracy_scores_train.append(adv_acc_train)
        adv_auc_scores_train.append(adv_auc_train)
        attack_success_ratio_scores_train.append(attack_success_ratio_train)
        fooling_ratio_scores_train.append(fooling_ratio_train)

        # Adversarial Validation
        adv_accuracy_scores_val.append(adv_acc_val)
        adv_auc_scores_val.append(adv_auc_val)
        attack_success_ratio_scores_val.append(attack_success_ratio_val)
        fooling_ratio_scores_val.append(fooling_ratio_val)

        # Adversarial Test

        # GM + RyC
        adv_accuracy_scores_test.append(adv_acc_test)
        adv_auc_scores_test.append(adv_auc_test)
        attack_success_ratio_scores_test.append(attack_success_ratio_test)
        fooling_ratio_scores_test.append(fooling_ratio_test)

        # INTRAHOSPITAL
        adv_accuracy_scores_test_intrahospital.append(adv_acc_test_intrahospital)
        adv_auc_scores_test_intrahospital.append(adv_auc_test_intrahospital)
        attack_success_ratio_scores_test_intrahospital.append(attack_success_ratio_test_intrahospital)
        fooling_ratio_scores_test_intrahospital.append(fooling_ratio_test_intrahospital)

        # DRIAMS
        adv_accuracy_scores_test_driams.append(adv_acc_test_driams)
        adv_auc_scores_test_driams.append(adv_auc_test_driams)
        attack_success_ratio_scores_test_driams.append(attack_success_ratio_test_driams)
        fooling_ratio_scores_test_driams.append(fooling_ratio_test_driams)

        # GM
        adv_accuracy_scores_test_gm.append(adv_acc_test_gm)
        adv_auc_scores_test_gm.append(adv_auc_test_gm)
        attack_success_ratio_scores_test_gm.append(attack_success_ratio_test_gm)
        fooling_ratio_scores_test_gm.append(fooling_ratio_test_gm)

        # RyC
        adv_accuracy_scores_test_ryc.append(adv_acc_test_ryc)
        adv_auc_scores_test_ryc.append(adv_auc_test_ryc)
        attack_success_ratio_scores_test_ryc.append(attack_success_ratio_test_ryc)
        fooling_ratio_scores_test_ryc.append(fooling_ratio_test_ryc)

        print("Datos benignos:")
        multi_layer.accuracy(trainloader, "Train")
        multi_layer.accuracy(validloader, "Validation")
        print("Validation set AUC: ", auroc_mlp(multi_layer, validloader))
        
        print("Ejemplos adversarios:")
        
        multi_layer.accuracy(adversarial_trainloader, "Adversarial Train")
        multi_layer.accuracy(adversarial_validloader, "Adversarial Validation")
        
        print("Adversarial Validation set AUC: ", auroc_mlp(multi_layer, adversarial_validloader))

        print('---------------------------------- EVALUACIÓN CON TEST ----------------------------------')
        
        print("AUC: ", auroc_mlp(multi_layer, testloader))
        # wandb.log({"auc":auroc_mlp(multi_layer, testloader)})
        
        accuracy, true_labels, predicted_labels = multi_layer.accuracy(testloader)
        # wandb.log({"acc":accuracy})
        print("ACC: ", accuracy)
        
        print('MATRIZ DE CONFUSIÓN CON DATOS BENIGNOS')
        # Calcular la matriz de confusión
        print(true_labels)
        print(predicted_labels)
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        print(conf_matrix)
        
        # Plotear resultados durante el entrenamiento
        
        plt.plot(multi_layer.loss_during_training, label='BCE Train')
        plt.plot(multi_layer.adv_loss_during_training, label='Adversarial BCE Train')
        plt.plot(multi_layer.valid_loss_during_training, label='BCE Validation')
        plt.plot(multi_layer.adv_valid_loss_during_training, label='Adversarial BCE Validation')
        
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.grid()
        # plt.savefig('MALDI-TOF/loss_plot.png')
        # wandb.log({"Plot de loss": wandb.Image("MALDI-TOF/loss_plot.png")})
        plt.savefig(f'recursos/loss_plot_{attack}.png')
        plt.close()
        classes = [str(i) for i in range(2)]
        plt.figure(figsize=(17, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        # plt.savefig('MALDI-TOF/confusion_matrix_weighted_loss.png')
        # wandb.log({"Matriz de confusión con Weighted Loss": wandb.Image("MALDI-TOF/confusion_matrix_weighted_loss.png")})
        plt.savefig(f'recursos/confusion_matrix_{attack}.png')
        plt.close()

      print("AUC VAL: ", np.mean(auc_scores_val))
      print("AUC TEST: ", np.mean(auc_scores_test))

      # Loggear resultados de CROSS-VALIDACIÓN

      # Train
      wandb.log({"balanced_acc_train":np.mean(accuracy_scores_train)})
      wandb.log({"auc_train":np.mean(auc_scores_train)})
      wandb.log({"adv_balanced_acc_train":np.mean(adv_accuracy_scores_train)})
      wandb.log({"adv_auc_train":np.mean(adv_auc_scores_train)})
      wandb.log({"attack_success_ratio_scores_train":np.mean(attack_success_ratio_scores_train)})
      wandb.log({"fooling_ratio_scores_train":np.mean(fooling_ratio_scores_train)})

      # Validation
      wandb.log({"balanced_acc_val":np.mean(accuracy_scores_val)})
      wandb.log({"auc_val":np.mean(auc_scores_val)})
      wandb.log({"adv_balanced_acc_val":np.mean(adv_accuracy_scores_val)})
      wandb.log({"adv_auc_val":np.mean(adv_auc_scores_val)})
      wandb.log({"attack_success_ratio_scores_val":np.mean(attack_success_ratio_scores_val)})
      wandb.log({"fooling_ratio_scores_val":np.mean(fooling_ratio_scores_val)})

      # Test

      # GM + RyC
      wandb.log({"balanced_acc_test":np.mean(accuracy_scores_test)})
      wandb.log({"auc_test":np.mean(auc_scores_test)})
      wandb.log({"adv_balanced_acc_test":np.mean(adv_accuracy_scores_test)})
      wandb.log({"adv_auc_test":np.mean(adv_auc_scores_test)})
      wandb.log({"attack_success_ratio_scores_test":np.mean(attack_success_ratio_scores_test)})
      wandb.log({"fooling_ratio_scores_test":np.mean(fooling_ratio_scores_test)})

      # INTRAHOSPITAL
      wandb.log({"balanced_acc_test_intrahospital":np.mean(accuracy_scores_test_intrahospital)})
      wandb.log({"auc_test_intrahospital":np.mean(auc_scores_test_intrahospital)})
      wandb.log({"adv_balanced_acc_test_intrahospital":np.mean(adv_accuracy_scores_test_intrahospital)})
      wandb.log({"adv_auc_test_intrahospital":np.mean(adv_auc_scores_test_intrahospital)})
      wandb.log({"attack_success_ratio_scores_test_intrahospital":np.mean(attack_success_ratio_scores_test_intrahospital)})
      wandb.log({"fooling_ratio_scores_test_intrahospital":np.mean(fooling_ratio_scores_test_intrahospital)})

      # DRIAMS
      wandb.log({"balanced_acc_test_driams":np.mean(accuracy_scores_test_driams)})
      wandb.log({"auc_test_driams":np.mean(auc_scores_test_driams)})
      wandb.log({"adv_balanced_acc_test_driams":np.mean(adv_accuracy_scores_test_driams)})
      wandb.log({"adv_auc_test_driams":np.mean(adv_auc_scores_test_driams)})
      wandb.log({"attack_success_ratio_scores_test_driams":np.mean(attack_success_ratio_scores_test_driams)})
      wandb.log({"fooling_ratio_scores_test_driams":np.mean(fooling_ratio_scores_test_driams)})

      # GM
      wandb.log({"balanced_acc_test_gm":np.mean(accuracy_scores_test_gm)})
      wandb.log({"auc_test_gm":np.mean(auc_scores_test_gm)})
      wandb.log({"adv_balanced_acc_test_gm":np.mean(adv_accuracy_scores_test_gm)})
      wandb.log({"adv_auc_test_gm":np.mean(adv_auc_scores_test_gm)})
      wandb.log({"attack_success_ratio_scores_test_gm":np.mean(attack_success_ratio_scores_test_gm)})
      wandb.log({"fooling_ratio_scores_test_gm":np.mean(fooling_ratio_scores_test_gm)})

      # RyC
      wandb.log({"balanced_acc_test_ryc":np.mean(accuracy_scores_test_ryc)})
      wandb.log({"auc_test_ryc":np.mean(auc_scores_test_ryc)})
      wandb.log({"adv_balanced_acc_test_ryc":np.mean(adv_accuracy_scores_test_ryc)})
      wandb.log({"adv_auc_test_ryc":np.mean(adv_auc_scores_test_ryc)})
      wandb.log({"attack_success_ratio_scores_test_ryc":np.mean(attack_success_ratio_scores_test_ryc)})
      wandb.log({"fooling_ratio_scores_test_ryc":np.mean(fooling_ratio_scores_test_ryc)})

sweep_id = wandb.sweep(sweep_config, project="malditof")
wandb.agent(sweep_id, function = do_sweep_mlp)