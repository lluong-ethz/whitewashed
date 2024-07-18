#!/usr/bin/env python3
from sklearn.feature_extraction.text import TfidfVectorizer


def transform_tweets_to_tfidf(tweets, vocabulary):
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    X = vectorizer.fit_transform(tweets)
    return X, vectorizer