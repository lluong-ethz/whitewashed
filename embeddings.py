#!/usr/bin/env python3
import pickle
import numpy as np
from utils import *
import torch
import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
# import tf.keras.models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


def cbow(full = False):
    tweets = []
    dummy = []

    # shouldn't we load the tweets outside the function?
    if(not full):
        load_tweets(SMALL_TRAIN_POS, 0, tweets, dummy)
        load_tweets(SMALL_TRAIN_NEG, 1, tweets, dummy)
    else:
        load_tweets(TRAIN_POS, 0, tweets, dummy)
        load_tweets(TRAIN_NEG, 0, tweets, dummy)

    # Convert to NumPy array to facilitate indexing
    tweets = np.array(tweets)
    labels = np.array(labels)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)

    vocab_size = len(tokenizer.word_index) + 1
    sequences = tokenizer.texts_to_sequences(tweets)

    window_size = 4

    skip_grams = [skipgrams(seq, vocabulary_size=vocab_size, window_size=window_size, negative_samples = 0) for seq in sequences]
    pairs, _ = skip_grams[0][0], skip_grams[0][1]

    word_target, word_context = zip(*pairs)
    word_target = np.array(word_target)
    word_context = np.array(word_context)

    embedding_size = 100

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=2*window_size))
    model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embedding_size,)))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    labels = to_categorical(labels, vocab_size)

    model.fit([word_context], word_target, epochs=100, verbose=2)

    # Extract the word embeddings
    word_embeddings = model.layers[0].get_weights()[0]

    # Function to get embedding for a word
    def get_embedding(word):
        word_id = tokenizer.word_id[word]
        return word_embeddings[word_id]

# Load GloVe embeddings
def load_glove_embeddings(glove_file):
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# Preprocess tweets (simple example, adapt as needed)
def preprocess_tweet(tweet):
    # Convert to lowercase
    tweet = tweet.lower()
    
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    
    # Remove user mentions
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Remove hashtags
    tweet = re.sub(r'#\w+', '', tweet)
    
    # Remove special characters, numbers, and punctuation
    tweet = re.sub(r'\W', ' ', tweet)
    tweet = re.sub(r'\d', '', tweet)
    
    # Remove extra spaces
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    
    # Tokenize by splitting on whitespace
    words = tweet.split()
    
    return words

# Convert tweets to GloVe vector representations
def tweet_to_glove_vector(tweet, embeddings_index, embedding_dim=100):
    words = preprocess_tweet(tweet)
    vectors = [embeddings_index.get(word, np.zeros(embedding_dim)) for word in words]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(embedding_dim)

glove_embeddings_path = 'glove/glove.6B.100d.txt'
embeddings_index = load_glove_embeddings(glove_embeddings_path)

# Load your data
tweets = []
labels = []
load_tweets(SMALL_TRAIN_POS, 0, tweets, labels)
load_tweets(SMALL_TRAIN_NEG, 1, tweets, labels)

tweet_emb = {}

tweet_emb['vector'] = tweets.apply(lambda x: tweet_to_glove_vector(x, embeddings_index))
tweet_emb['label'] = labels

# Prepare features and labels
X = np.array(tweet_emb['vector'].tolist())
y = tweet_emb['label']  # Assuming 'label' column contains 0 for negative and 1 for positive

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
