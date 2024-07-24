#!/usr/bin/env python3
import sys
import os
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from utils import get_basic_metrics, load_tweets, split_train_test
from preprocessing import preprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# split into training and validation set
def split_train_test(tweets, seed):
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(tweets))
    split_idx = int(0.9 * len(tweets))
    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]
    return train_indices, val_indices


# create Bidirectional LSTM model
def create_bilstm_model(vocab_size, embedding_dim, input_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# train Bi-LSTM model
def train_bilstm(model, X_train, Y_train, epochs=2, batch_size=32):
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return model


# make predictions with the Bi-LSTM model
def predict_bilstm(model, X_val):
    y_pred = (model.predict(X_val) > 0.5).astype("int32")
    return y_pred


def main():
    tweets = []
    labels = []

    # load tweets
    load_tweets('../data/train_neg.txt', 0, tweets, labels)
    load_tweets('../data/train_pos.txt', 1, tweets, labels)
    # load_tweets('../data/train_neg_full.txt', 0, tweets, labels)
    # load_tweets('../data/train_pos_full.txt', 1, tweets, labels)

    # preprocess tweets and labels
    preprocess(tweets, labels)

    # convert to NumPy array to facilitate indexing
    tweets = np.array(tweets)
    labels = np.array(labels)

    train_indices, val_indices = split_train_test(tweets, seed=1, )

    # tokenize the tweets
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    vocab_size = len(tokenizer.word_index) + 1

    # convert tweets to sequences
    X = tokenizer.texts_to_sequences(tweets)
    X = pad_sequences(X, padding='post')

    input_length = X.shape[1]
    embedding_dim = 200

    bilstm_model = create_bilstm_model(vocab_size, embedding_dim, input_length)

    X_train = X[train_indices]
    X_val = X[val_indices]
    Y_train = labels[train_indices]
    Y_val = labels[val_indices]

    # train model
    model = train_bilstm(bilstm_model, X_train, Y_train)

    # predict on validation set
    y_pred = predict_bilstm(model, X_val)

    # print metrics
    get_basic_metrics(y_pred, Y_val)


if __name__ == '__main__':
    main()
