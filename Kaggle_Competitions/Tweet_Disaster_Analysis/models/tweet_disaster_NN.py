import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score, \
    classification_report
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from timeit import default_timer as timer

from tweet_disaster import tfid_preprocessing, X, y, X_test, test_df_id, count_preprocessing

X_train, X_val, y_train, y_val, X_test = count_preprocessing(X, X_test, y)

def get_data_loaders(BATCH_SIZES=256):
    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZES, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZES, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZES)
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = get_data_loaders(256)

def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)


def calculate_metrics(all_labels, all_preds):
    metrics = {
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),  # It's fine to leave the matrix as-is
        'precision': float(precision_score(all_labels, all_preds)),
        'recall': float(recall_score(all_labels, all_preds)),
        'f1': float(f1_score(all_labels, all_preds)),
        'macro_precision': float(precision_score(all_labels, all_preds, average='macro')),
        'macro_recall': float(recall_score(all_labels, all_preds, average='macro')),
        'macro_f1': float(f1_score(all_labels, all_preds, average='macro')),
        'micro_precision': float(precision_score(all_labels, all_preds, average='micro')),
        'micro_recall': float(recall_score(all_labels, all_preds, average='micro')),
        'micro_f1': float(f1_score(all_labels, all_preds, average='micro'))
    }

    return metrics, classification_report(all_labels, all_preds, target_names=['ham', 'spam'], digits=6)

class TweetDisasterModel(nn.Module):
    def __init__(self,input_shape,hidden_units1,hidden_units2,out_shape):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=hidden_units1),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(in_features=hidden_units1, out_features=hidden_units2),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units1, out_features=out_shape)
        )

    def forward(self,X):
        return self.layer(X)

class TweetDisasterRNNModel(nn.Module):
    def __init__(self,input_shape,hidden_units,out_shape):
        super().__init__()
        self.rnn = nn.RNN(input_shape, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, out_shape)

    def forward(self,X):
        X = X.unsqueeze(1)
        output, hidden = self.rnn(X)
        return self.fc(hidden[-1])

tweet_model = TweetDisasterModel(X_train.shape[1],16,8,1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(tweet_model.parameters(), lr=0.001, weight_decay=1e-4)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

tweet_RNN_model = TweetDisasterRNNModel(X_train.shape[1],16,1)
criterion_RNN = nn.BCEWithLogitsLoss()
optimizer_RNN = optim.Adam(tweet_model.parameters(), lr=0.001, weight_decay=1e-4)

def train_mode(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for batch, (X, y) in enumerate(data_loader):
        y_preds = model(X)
        loss = loss_fn(y_preds, y.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = torch.sigmoid(y_preds).round()  # Apply sigmoid and threshold at 0.5
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        # running_accuracy +=
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(data_loader.dataset)} samples")
    train_loss = running_loss / len(data_loader)

    return train_loss, calculate_metrics(all_labels, all_preds)


def test_mode(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for batch, (X, y) in enumerate(data_loader):
        y_preds = model(X)
        loss = loss_fn(y_preds, y.unsqueeze(1))
        running_loss += loss.item()
        preds = torch.sigmoid(y_preds).round()  # Apply sigmoid and threshold at 0.5
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(data_loader.dataset)} samples")
    test_loss = running_loss / len(data_loader)

    return test_loss, calculate_metrics(all_labels, all_preds)


def predict_on_test_set(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader):
    model.eval()
    all_preds = []

    with torch.no_grad():  # No need to track gradients during inference
        for batch, X in enumerate(test_loader):
            y_preds = model(X[0])
            preds = torch.sigmoid(y_preds).round()
            all_preds.extend(preds.detach().cpu().numpy())

    return all_preds


def run_model(model, criterion, optimizer, scheduler):
    torch.manual_seed(42)
    train_time_start_on_cpu = timer()
    epochs = 12
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n---------")
        train_loss, (train_metrics, train_classification_report) = train_mode(tweet_model, train_loader, criterion,
                                                                              optimizer)
        print(f"Train loss: {train_loss:.5f}")
        print(train_classification_report)

        test_loss, (test_metrics, test_classification_report) = test_mode(tweet_model, val_loader, criterion)
        print(f"Test loss: {test_loss:.5f}")
        print(test_classification_report)
        if scheduler is not None:
            scheduler.step()
        print("___________________________________")
    train_time_end_on_cpu = timer()
    total_train_time_model = print_train_time(start=train_time_start_on_cpu,
                                              end=train_time_end_on_cpu,
                                              device=str(next(model.parameters()).device))


run_model(model=tweet_model, criterion=criterion, optimizer=optimizer, scheduler=None)

y_pred = predict_on_test_set(tweet_model, test_loader)
y_pred = [int(pred[0]) for pred in y_pred]
output_df = pd.DataFrame({
    'id': test_df_id,
    'target': y_pred
})

# Save the DataFrame to a CSV file
output_df.to_csv(r'D:\Kaggle\disaster tweets\simple_nn.csv', index=False)