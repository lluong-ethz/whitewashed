#!/usr/bin/env python3
import pickle
import numpy as np
from utils import *
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# BoW pipeline
def bow(X_train, X_val, Y_train, Y_val, verbose = True):

    # We only keep the 5000 most frequent words, both to reduce the computational cost and reduce overfitting
    vectorizer = CountVectorizer(max_features=5000)

    # Important: we call fit_transform on the training set, and only transform on the validation set
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)

    model = LogisticRegression(C=1e5, max_iter=100)
    model.fit(X_train, Y_train)

    model_features = model.coef_[0]
    mapping = vectorizer.get_feature_names_out()

    y_pred = model.predict(X_val)

    if(verbose):
        get_top_pos_neg(model_features, mapping, 10)

        accuracy = accuracy_score(Y_val, y_pred) 
        report = classification_report(Y_val, y_pred)
        print(f"Validation Accuracy: {accuracy}")
        print(report)


    return model