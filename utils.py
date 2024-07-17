import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import re
from preprocessing import *
from collections import Counter

# from keras.preprocessing.text import Tokenizer
# from keras.utils import pad_sequences


SMALL_TRAIN_NEG = 'data/twitter-datasets/train_neg.txt'
SMALL_TRAIN_POS = 'data/twitter-datasets/train_pos.txt'
TRAIN_NEG = 'data/twitter-datasets/train_neg_full.txt'
TRAIN_POS = 'data/twitter-datasets/train_pos_full.txt'

GLOVE_WIKI_50D = 'data/glove_wiki/glove.6B.50d.txt'
GLOVE_TWEET_50D = 'data/glove_twitter/glove.twitter.27B.50d.txt'
GLOVE_TWEET_200D = 'data/glove_twitter/glove.twitter.27B.200d.txt'


def load_tweets(filename, label, tweets, labels):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tweets.append(line.rstrip())
            labels.append(label)


def split_train_test(tweets, labels, seed):
    np.random.seed(seed)

    shuffled_indices = np.random.permutation(len(tweets))
    split_idx = int(0.9 * len(tweets))
    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]

    return tweets[train_indices], tweets[val_indices], labels[train_indices], labels[val_indices]


def split_train_test(tweets, seed):
    np.random.seed(seed)

    shuffled_indices = np.random.permutation(len(tweets))
    split_idx = int(0.9 * len(tweets))
    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]

    return train_indices, val_indices


def get_tokens(x_train, x_val):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(x_train)
    tokens_train = tokenizer.texts_to_sequences(x_train)
    tokens_val = tokenizer.texts_to_sequences(x_val)

    # using average length of word in English
    max_sequence_length = 35
    tokens_train = pad_sequences(tokens_train, maxlen=max_sequence_length)
    tokens_val = pad_sequences(tokens_val, maxlen=max_sequence_length)

    return tokens_train, tokens_val


def get_top_pos_neg(model_features, words, k):
    # increasing order
    sorted_features = np.argsort(model_features)
    # small coefs equivalent to negative
    top_neg = sorted_features[:k]
    # large coefs equivalent to positive
    top_pos = sorted_features[-k:]

    print('---- Top 10 negative words')
    for i in top_neg:
        print(words[i], model_features[i])
    print()

    print('---- Top 10 positive words')
    for i in top_pos:
        print(words[i], model_features[i])
    print()


def get_basic_metrics(pred, gt):
    print("ACCURACY: " + str(accuracy_score(gt, pred)))
    print("RECALL: " + str(recall_score(gt, pred)))
    print("F1: " + str(f1_score(gt, pred)))
    print("PRECISION: " + str(precision_score(gt, pred)))


# !/usr/bin/env python3
from collections import Counter


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
