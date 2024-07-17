#!/usr/bin/env python3
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils import load_tweets, split_train_test, get_top_pos_neg, get_basic_metrics


def build_vocab(tweets):
    vocabulary = {}
    tokens = []
    for tweet in tweets:
        for word in tweet.split():
            tokens.append(word)

    # We only keep the 5000 most frequent words, both to reduce the computational cost and reduce overfitting
    counter = Counter(tokens)
    most_common_words = counter.most_common(5000)
    most_common_words_len = len(most_common_words)

    for index in range(most_common_words_len):
        word = most_common_words[index][0]
        vocabulary[word] = index
    return vocabulary


# train machine learning model to predict the label
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
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(tweets)

    X_train = X[train_indices]
    X_val = X[val_indices]

    Y_train = labels[train_indices]
    Y_val = labels[val_indices]

    model = MultinomialNB()
    model.fit(X_train, Y_train)

    # Feature log probabilities
    model_features = model.feature_log_prob_

    # We need to subtract the log probabilities of the negative class from the positive class
    feature_diff = model_features[1] - model_features[0]

    # Get the top positive and negative words
    get_top_pos_neg(feature_diff, vectorizer.get_feature_names_out(), k=10)

    # Predict on validation set
    y_pred = model.predict(X_val)

    # Print metrics
    get_basic_metrics(y_pred, Y_val)


if __name__ == '__main__':
    main()
