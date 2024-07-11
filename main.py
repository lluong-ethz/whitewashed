from models.bow import *
from utils import *
from embeddings.glove import *
from models.nn import *
from torch.utils.data import DataLoader, TensorDataset
import torch
import pickle

def main():
    full = False

    tweets = []
    labels = []

    if(not full):
        load_tweets(SMALL_TRAIN_POS, 0, tweets, labels)
        load_tweets(SMALL_TRAIN_NEG, 1, tweets, labels)
    else:
        load_tweets(TRAIN_POS, 0, tweets, labels)
        load_tweets(TRAIN_NEG, 0, tweets, labels)

    # Convert to NumPy array to facilitate indexing
    tweets = np.array(tweets)
    labels = np.array(labels)

    X_train, X_val, Y_train, Y_val = split_train_test(tweets, labels, 1)

    X_train = all_tweets_to_glove(X_train, GLOVE_WIKI_50D)
    # Save the list to a file
    with open('glove_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    X_val = all_tweets_to_glove(X_val, GLOVE_WIKI_50D)
    with open('glove_val.pkl', 'wb') as f:
        pickle.dump(X_val, f)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = train_simple_nn(train_loader)
    test_simple_nn(test_loader, model)

if __name__ == '__main__':
    main()