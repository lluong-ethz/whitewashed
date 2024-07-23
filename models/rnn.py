import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils import *
from sklearn.metrics import accuracy_score, classification_report

# Class for RNN
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.rnn(x)
        hidden = hidden.squeeze(0)
        output = self.fc(hidden)
        output = self.sigmoid(output)
        return output

# Training RNN
def train_rnn(train_loader, save_model = True):
    vocab_size = 5000
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 1

    model = RNN(vocab_size, embedding_dim, hidden_dim, output_dim)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 25
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    if(save_model):
        torch.save(model, 'rnn.pt')
    return model

# Testing RNN
def test_rnn(test_loader, model):
    model.eval()
    #correct = 0
    #total = 0
    all_preds = []
    all_labels = []  
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            predicted = (outputs > 0.5).float()
            #total += labels.size(0)
            #correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    #accuracy = correct / total
    #print(f'Test Accuracy: {accuracy * 100:.2f}%')
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    print(f"Validation Accuracy: {accuracy}")
    print(report)




