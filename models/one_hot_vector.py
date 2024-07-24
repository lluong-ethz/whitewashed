#!/usr/bin/env python3
import numpy as np
import sys
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from utils import load_tweets, get_basic_metrics, build_vocab
from preprocessing import preprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# split into training and validation set
def split_train_test(tweets, seed):
    np.random.seed(seed)  # reproducibility
    shuffled_indices = np.random.permutation(len(tweets))
    split_idx = int(0.9 * len(tweets))
    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]
    return train_indices, val_indices


# converts tweet to a one-hot vector based on vocabulary
def convert_to_one_hot_vec(tweet, vocabulary):
    vector = np.zeros(len(vocabulary))
    for word in tweet.split():
        if word in vocabulary:
            index = vocabulary.get(word)
            vector[index] = 1
    return vector


def main():
    tweets = []
    labels = []

    # load tweets
    load_tweets('../data/train_neg.txt', 0, tweets, labels)
    load_tweets('../data/train_pos.txt', 1, tweets, labels)
    # load_tweets('data/train_neg_full.txt', 0)
    # load_tweets('data/train_pos_full.txt', 1)
    preprocess(tweets, labels)

    # convert to NumPy array to facilitate indexing
    tweets = np.array(tweets)
    labels = np.array(labels)

    train_indices, val_indices = split_train_test(tweets, seed=1)

    # build vocabulary
    vocab = build_vocab(tweets)

    # we only keep the 5000 most frequent words, both to reduce the computational cost and reduce overfitting
    vectorizer = CountVectorizer(vocabulary=vocab)
    vectorizer.fit_transform(tweets)

    # convert tweets to one-hot vectors
    X_train = []
    X_val = []
    for tweet in tqdm(tweets[train_indices], desc="Converting training set"):
        X_train.append(convert_to_one_hot_vec(tweet, vocab))
    for tweet in tqdm(tweets[val_indices], desc="Converting validation set"):
        X_val.append(convert_to_one_hot_vec(tweet, vocab))

    Y_train = labels[train_indices]
    Y_val = labels[val_indices]

    # train logistic regression model
    model = LogisticRegression(C=1e5, max_iter=100)
    model.fit(X_train, Y_train)

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

    # predict on validation set
    y_pred = model.predict(X_val)

    # print metrics
    get_basic_metrics(y_pred, Y_val)


if __name__ == '__main__':
    main()
