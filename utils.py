import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SMALL_TRAIN_NEG = 'data/twitter-datasets/train_neg.txt'
SMALL_TRAIN_POS = 'data/twitter-datasets/train_pos.txt'
TRAIN_NEG = 'data/twitter-datasets/train_neg_full.txt'
TRAIN_POS = 'data/twitter-datasets/train_pos_full.txt'
GLOVE_WIKI_50D = 'data/glove_wiki/glove.6B.50d.txt'

def load_tweets(filename, label, tweets, labels):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                tweets.append(line.rstrip())
                labels.append(label)

def clean_text(text):
     return [w.lower().replace("\n", "").split() for w in text]

def split_train_test(tweets, labels, seed):
    np.random.seed(seed)

    shuffled_indices = np.random.permutation(len(tweets))
    split_idx = int(0.9 * len(tweets))
    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]

    return tweets[train_indices], tweets[val_indices], labels[train_indices], labels[val_indices]

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