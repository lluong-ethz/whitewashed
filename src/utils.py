import numpy as np
#import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from preprocessing import *
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


# Global variables for tweet paths
SMALL_TRAIN_NEG = 'data/twitter-datasets/train_neg.txt'
SMALL_TRAIN_POS = 'data/twitter-datasets/train_pos.txt'
TRAIN_NEG = 'data/twitter-datasets/train_neg_full.txt'
TRAIN_POS = 'data/twitter-datasets/train_pos_full.txt'
TEST_SET = 'data/twitter-datasets/test_data.txt'

# Global variables for Glove paths
GLOVE_WIKI_50D = 'data/glove_wiki/glove.6B.50d.txt'
GLOVE_WIKI_100D = 'data/glove_wiki/glove.6B.100d.txt'
GLOVE_WIKI_200D = 'data/glove_wiki/glove.6B.200d.txt'
GLOVE_TWEET_50D = 'data/glove_twitter/glove.twitter.27B.50d.txt'
GLOVE_TWEET_100D = 'data/glove_twitter/glove.twitter.27B.100d.txt'
GLOVE_TWEET_200D = 'data/glove_twitter/glove.twitter.27B.200d.txt'


# Method to tokenize a list of tweets given a tokenizer
def tokenize_tweets(tweets, tokenizer):
    return tokenizer(tweets, padding="max_length", truncation=True)  


# Loading tweets from a file path
def load_tweets(filename, label, tweets, labels):
    indices = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # separate indices if using test set
            if(filename == TEST_SET):
                index, tweet = line.split(',', 1)
                indices.append(int(index))
                tweets.append(tweet)
                labels.append(label)
            else:
                tweets.append(line.rstrip())
                labels.append(label)
    return indices

# Splitting the data in train / val
def split_train_test(tweets, labels, seed = 1):
    np.random.seed(seed)

    shuffled_indices = np.random.permutation(len(tweets))
    split_idx = int(0.9 * len(tweets))
    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]

    return tweets[train_indices], tweets[val_indices], labels[train_indices], labels[val_indices]


# Tokenize tweets for the RNN model
def get_tokens_rnn(x_train, x_val):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(x_train)
    tokens_train = tokenizer.texts_to_sequences(x_train)
    tokens_val = tokenizer.texts_to_sequences(x_val)

    # using average length of a word in English
    max_sequence_length = 35
    tokens_train = pad_sequences(tokens_train, maxlen=max_sequence_length)
    tokens_val = pad_sequences(tokens_val, maxlen=max_sequence_length)

    return tokens_train, tokens_val

# If doing logistic regresion, obtain most "positive / negative" words
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

# Basic metrics for classification
def get_basic_metrics(pred, gt):
    print("ACCURACY: " + str(accuracy_score(gt, pred)))
    print("RECALL: " + str(recall_score(gt, pred)))
    print("F1: " + str(f1_score(gt, pred)))
    print("PRECISION: " + str(precision_score(gt, pred)))


# !/usr/bin/env python3
from collections import Counter

# Build vocabulary of tweets
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

def submission(model):
    tweets = []
    dummy = []

    indices = load_tweets(TEST_SET, 0, tweets, dummy)

    # TODO: do some pre-processing?

    predictions = model(tweets)
    labels = [1 if pred > 0.5 else -1 for pred in predictions]

    df = pd.DataFrame({
    'Id': indices,
    'Prediction': labels
    }) 

    df.to_csv("submission.csv", index = False)

