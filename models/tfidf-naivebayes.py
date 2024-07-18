#!/usr/bin/env python3
import numpy as np
from utils import load_tweets, split_train_test, get_top_pos_neg, get_basic_metrics, build_vocab
from embeddings.tfidf import transform_tweets_to_tfidf
from models.naivebayes import train_naive_bayes, predict_naive_bayes


def main():
    tweets = []
    labels = []

    # Load tweets
    load_tweets('../data/train_neg.txt', 0, tweets, labels)
    load_tweets('../data/train_pos.txt', 1, tweets, labels)
    # load_tweets('../data/train_neg_full.txt', 0, tweets, labels)
    # load_tweets('../data/train_pos_full.txt', 1, tweets, labels)

    # Convert to NumPy array to facilitate indexing
    tweets = np.array(tweets)
    labels = np.array(labels)

    train_indices, val_indices = split_train_test(tweets, seed=1)

    # Build vocabulary
    vocab = build_vocab(tweets)

    # Transform the tweets into TF-IDF features
    X, vectorizer = transform_tweets_to_tfidf(tweets, vocab)

    X_train = X[train_indices]
    X_val = X[val_indices]

    Y_train = labels[train_indices]
    Y_val = labels[val_indices]

    # Train Naive Bayes model
    model = train_naive_bayes(X_train, Y_train)

    # Feature log probabilities
    model_features = model.feature_log_prob_

    # We need to subtract the log probabilities of the negative class from the positive class
    feature_diff = model_features[1] - model_features[0]

    # Get the top positive and negative words
    get_top_pos_neg(feature_diff, vectorizer.get_feature_names_out(), k=10)

    # Predict on validation set
    y_pred = predict_naive_bayes(model, X_val)

    # Print metrics
    get_basic_metrics(y_pred, Y_val)


if __name__ == '__main__':
    main()
