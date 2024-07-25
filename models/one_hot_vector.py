#!/usr/bin/env python3
import numpy as np
import sys
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from utils import load_tweets, get_basic_metrics, build_vocab, split_train_test
from preprocessing import preprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# converts tweet to a one-hot vector based on vocabulary
def convert_to_one_hot_vec(tweet, vocabulary):
    vector = np.zeros(len(vocabulary))
    for word in tweet.split():
        if word in vocabulary:
            index = vocabulary.get(word)
            vector[index] = 1
    return vector


# get top 10 negative and positive features
def get_features_ohv(model, vectorizer):
    model_features = model.coef_[0]
    # increasing order
    sorted_features = np.argsort(model_features)
    # small coefs equivalent to negative
    top_neg = sorted_features[:10]
    # large coefs equivalent to positive
    top_pos = sorted_features[-10:]

    mapping = vectorizer.get_feature_names_out()

    print('---- Top 10 negative words')
    for i in top_neg:
        print(mapping[i], model_features[i])
    print()

    print('---- Top 10 positive words')
    for i in top_pos:
        print(mapping[i], model_features[i])
    print()


def main():
    tweets = []
    labels = []

    # load tweets
    load_tweets('../data/twitter-datasets/train_neg.txt', 0, tweets, labels)
    load_tweets('../data/twitter-datasets/train_pos.txt', 1, tweets, labels)
    # load_tweets('../data/twitter-datasets/train_neg_full.txt', 0, tweets, labels)
    # load_tweets('../data/twitter-datasets/train_pos_full.txt', 1, tweets, labels)
    preprocess(tweets, labels)

    # convert to NumPy array to facilitate indexing
    tweets = np.array(tweets)
    labels = np.array(labels)

    X_train, X_val, Y_train, Y_val = split_train_test(np.array(tweets), np.array(labels), 1)

    # build vocabulary
    vocab = build_vocab(tweets)

    # we only keep the 5000 most frequent words, both to reduce the computational cost and reduce overfitting
    vectorizer = CountVectorizer(vocabulary=vocab)
    vectorizer.fit_transform(tweets)

    # convert tweets to one-hot vectors
    X_train_ohv = []
    X_val_ohv = []
    for tweet in tqdm(X_train, desc="Converting training set"):
        X_train_ohv.append(convert_to_one_hot_vec(tweet, vocab))
    for tweet in tqdm(X_val, desc="Converting validation set"):
        X_val_ohv.append(convert_to_one_hot_vec(tweet, vocab))

    # train logistic regression model
    model = LogisticRegression(C=1e5, max_iter=100)
    model.fit(X_train_ohv, Y_train)

    # print features
    get_features_ohv(model, vectorizer)

    # predict on validation set
    y_pred = model.predict(X_val_ohv)

    # print metrics
    get_basic_metrics(y_pred, Y_val)


if __name__ == '__main__':
    main()
