#!/usr/bin/env python3
import pickle
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def main():

    tweets = []
    labels = []

    def load_tweets(filename, label):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                tweets.append(line.rstrip())
                labels.append(label)


    load_tweets('twitter-datasets/train_neg.txt', 0)
    load_tweets('twitter-datasets/train_pos.txt', 1)
    #load_tweets('twitter-datasets/train_neg_full.txt', 0)
    #load_tweets('twitter-datasets/train_pos_full.txt', 1)

    # Convert to NumPy array to facilitate indexing
    tweets = np.array(tweets)
    labels = np.array(labels)

    np.random.seed(1) # Reproducibility!

    shuffled_indices = np.random.permutation(len(tweets))
    split_idx = int(0.9 * len(tweets))
    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]

    # We only keep the 5000 most frequent words, both to reduce the computational cost and reduce overfitting
    vectorizer = CountVectorizer(max_features=5000)

    # Important: we call fit_transform on the training set, and only transform on the validation set
    X_train = vectorizer.fit_transform(tweets[train_indices])
    X_val = vectorizer.transform(tweets[val_indices])

    Y_train = labels[train_indices]
    Y_val = labels[val_indices]

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

    y_pred = model.predict(X_val)

    print("ACCURACY: " + str(accuracy_score(Y_val, y_pred)))
    print("RECALL: " + str(recall_score(Y_val, y_pred)))
    print("F1: " + str(f1_score(Y_val, y_pred)))
    print("PRECISION: " + str(precision_score(Y_val, y_pred)))
    

if __name__ == '__main__':
    main()
