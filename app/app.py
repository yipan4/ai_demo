import requests
import numpy as np 
import pandas as pd
import torch 

DATA_URL = "https://example.com/data.csv"

def fetch_data(url: str) -> int:
    r = requests.get(url, timeout=5)
    return r.status_code

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    # Example processing: fill NaNs and normalize a column
    data = data.fillna(0)
    if 'value' in data.columns:
        data['value'] = (data['value'] - data['value'].mean()) / data['value'].std()
    return data

# Simple neural network model
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Train model 
def train_model(data: pd.DataFrame):
    if 'value' not in data.columns:
        raise ValueError("Data must contain 'value' column for training.")
    
    X = data.drop(columns=['value']).values
    y = data['value'].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    model = SimpleNN(input_size=X.shape[1], hidden_size=10, output_size=1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

    return model


def evaluate_model(model: SimpleNN, data: pd.DataFrame):
    if 'value' not in data.columns:
        raise ValueError("Data must contain 'value' column for evaluation.")
    
    X = data.drop(columns=['value']).values
    y = data['value'].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor)
        mse = torch.nn.MSELoss()(predictions, y_tensor).item()
    
    print(f'Evaluation MSE: {mse:.4f}')
    return mse

def run_pipeline():
    status_code = fetch_data(DATA_URL)
    if status_code != 200:
        raise ValueError(f"Failed to fetch data, status code: {status_code}")

    data = pd.read_csv(DATA_URL)
    processed_data = process_data(data)
    model = train_model(processed_data)
    evaluate_model(model, processed_data)

if __name__ == "__main__":
    run_pipeline()
