#!/usr/bin/env python3
import pickle
import numpy as np
from utils import *
#import torch
#import tensorflow as tf

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re

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
    #tweet = re.sub(r'@\w+', '', tweet)
    
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
def tweet_to_glove(tweet, embeddings_index, embedding_dim=50):
    words = preprocess_tweet(tweet)
    vectors = [embeddings_index.get(word, np.zeros(embedding_dim)) for word in words]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(embedding_dim)

def all_tweets_to_glove(tweets, path):
    embeddings_index = load_glove_embeddings(path)
    return [tweet_to_glove(tweet, embeddings_index, 50) for tweet in tweets]
    

    
# SHOULD WE DO LOGISTIC REGRESSION HERE TOO?

#glove_embeddings_path = 'glove_wiki/glove.6B.100d.txt'
#embeddings_index = load_glove_embeddings(glove_embeddings_path)

# Load your data
#tweets = []
#labels = []
#load_tweets(SMALL_TRAIN_POS, 0, tweets, labels)
#load_tweets(SMALL_TRAIN_NEG, 1, tweets, labels)

#print(tweet_to_glove_vector(tweets[0], embeddings_index))

#tweet_emb = {}

#contains all the embeddings

#tweet_emb['vector'] = [tweet_to_glove_vector(tweet, embeddings_index) for tweet in tweets]
# print(tweet_emb['vector'][:10])
# tweet_emb['label'] = labels

# # Prepare features and labels
# X = tweet_emb['vector']
# y = tweet_emb['label']  # Assuming 'label' column contains 0 for negative and 1 for positive

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a classifier
# clf = LogisticRegression()
# clf.fit(X_train, y_train)

# # Predict and evaluate
# y_pred = clf.predict(X_test)
# print('Accuracy:', accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
