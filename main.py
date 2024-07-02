from models.bow import *
from utils import *
from embeddings.glove import *

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

    # bow()
    #glove_embeddings_path = 'data/glove_wiki/glove.6B.50d.txt'
    #embeddings_index = load_glove_embeddings(glove_embeddings_path)

    # Load your data
    #tweets = []
    #labels = []
    #load_tweets(SMALL_TRAIN_POS, 0, tweets, labels)
    #load_tweets(SMALL_TRAIN_NEG, 1, tweets, labels)
    #print(tweet_to_glove_vector(tweets[0], embeddings_index))


if __name__ == '__main__':
    main()