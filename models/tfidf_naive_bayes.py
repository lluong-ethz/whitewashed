#!/usr/bin/env python3
import pickle
import numpy as np
from utils import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def tfidf_naive_bayes(tweets, labels):

    X_train, X_val, Y_train, Y_val = split_train_test(tweets, labels, 1)

    # We only keep the 5000 most frequent words, both to reduce the computational cost and reduce overfitting
    vectorizer = TfidfVectorizer(max_features=5000)

    # Important: we call fit_transform on the training set, and only transform on the validation set
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)

    model = MultinomialNB()
    model.fit(X_train, Y_train)

    model_features = model.coef_[0]
    mapping = vectorizer.get_feature_names_out()

    get_top_pos_neg(model_features, mapping, 10)

    y_pred = model.predict(X_val)

    get_basic_metrics(y_pred, Y_val)
