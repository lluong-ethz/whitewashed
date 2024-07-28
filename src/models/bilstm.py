#!/usr/bin/env python3
import sys
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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


# tokenize and pad tweets & get vocabulary with tokenized tweets
def prepare_tweets(tweets):
    # tokenize the tweets
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    vocab_size = len(tokenizer.word_index) + 1

    # convert tweets to sequences
    X = tokenizer.texts_to_sequences(tweets)
    X = pad_sequences(X, padding='post')
    return X, vocab_size
