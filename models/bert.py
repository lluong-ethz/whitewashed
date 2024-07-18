import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from sklearn.metrics import accuracy_score, classification_report
from utils import *

def train_bert(train_loader):
    model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")

    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    epochs = 3
    num_training_steps = epochs * len(train_loader)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            #outputs = model(input_ids=input, labels = labels).squeeze()
            outputs = model(input_ids=input_ids, attention_mask = attention_mask, labels = labels)
            loss = outputs.loss
            loss.backward()
            lr_scheduler.step()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
    
    torch.save(model, 'bert.pt')
    return model

def test_bert(val_loader, model):
    model.eval()
    all_preds = []
    all_labels = []     
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids=input_ids, attention_mask = attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels)

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    print(f"Validation Accuracy: {accuracy}")
    print(report)
