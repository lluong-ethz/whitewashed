#!/usr/bin/env python3
import pickle
import numpy as np
from utils import *
import torch
import tensorflow as tf

import numpy as np
import tf.keras.models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


def cbow(tweets):

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
