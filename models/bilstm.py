#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from utils import get_basic_metrics, get_top_pos_neg, load_tweets, split_train_test, build_vocab


def create_bilstm_model(vocab_size, embedding_dim, input_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_bilstm(model, X_train, Y_train, epochs=5, batch_size=32):
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return model


def predict_bilstm(model, X_val):
    y_pred = (model.predict(X_val) > 0.5).astype("int32")
    return y_pred


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

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    vocab_size = len(tokenizer.word_index) + 1

    X = tokenizer.texts_to_sequences(tweets)
    X = pad_sequences(X, padding='post')

    input_length = X.shape[1]

    embedding_dim = 100
    bilstm_model = create_bilstm_model(vocab_size, embedding_dim, input_length)

    X_train = X[train_indices]
    X_val = X[val_indices]

    Y_train = labels[train_indices]
    Y_val = labels[val_indices]

    # Train model
    model = train_bilstm(bilstm_model, X_train, Y_train)
    model.fit(X_train, Y_train)

    y_pred = model.predict(X_val)

    # Predict on validation set
    y_pred = predict_bilstm(bilstm_model, X_val)

    # Print metrics
    get_basic_metrics(y_pred, Y_val)


if __name__ == '__main__':
    main()
