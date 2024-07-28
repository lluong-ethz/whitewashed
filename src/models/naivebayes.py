#!/usr/bin/env python3
from sklearn.naive_bayes import MultinomialNB

def train_naive_bayes(X_train, Y_train):
    model = MultinomialNB()
    model.fit(X_train, Y_train)
    return model

def predict_naive_bayes(model, X_val):
    y_pred = model.predict(X_val)
    return y_pred
