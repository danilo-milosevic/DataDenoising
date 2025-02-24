import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, l2_reg=0.0):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )
        self.l2_reg = l2_reg
    
    def forward(self, x):
        return self.model(x)
    
    def fit(self, X_train, y_train, epochs=10, batch_size=32, learning_rate=0.001):
        X_train = torch.tensor(np.array(X_train, dtype=np.float32))
        y_train = torch.from_numpy(np.array(y_train))
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss expects raw logits
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=self.l2_reg)
        
        for _ in range(epochs):
            total_loss = 0.0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = self.forward(X_batch)
                loss = criterion(outputs, y_batch.argmax(dim=1))  # Convert one-hot to class index
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(np.array(X, dtype=np.float32))
            outputs = self.forward(X_tensor)
            predicted_labels = torch.argmax(outputs, dim=1)  # Convert logits to class index
        return predicted_labels.numpy()
