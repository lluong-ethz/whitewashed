#from models.bow import *
from utils import *
from embeddings.glove import *
from models.nn import *
from torch.utils.data import DataLoader, TensorDataset
import torch
import pickle
from models.rnn import *
from models.bert import *
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from preprocessing import *
from torch.nn.utils.rnn import pad_sequence

def main():
    full = False

    tweets = []
    labels = []

    if(not full):
        load_tweets(SMALL_TRAIN_POS, 1, tweets, labels)
        load_tweets(SMALL_TRAIN_NEG, 0, tweets, labels)
    else:
        load_tweets(TRAIN_POS, 1, tweets, labels)
        load_tweets(TRAIN_NEG, 0, tweets, labels)
    
    #tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

    model = torch.load('bert.pt')
    submission(model)

    #max_length = 0
    #for tweet in tweets:
    #    tokenized_tweet = tokenizer(tweet, truncation=False)
    #    tweet_length = len(tokenized_tweet['input_ids'])
    #    if tweet_length > max_length:
    #        max_length = tweet_length

    # 103
    #print("Length of the largest tweet:", max_length)

    # tweets, labels = preprocess(tweets, labels)
    
    # Convert to NumPy array to facilitate indexing
    #tweets = np.array(tweets)
    #labels = np.array(labels)
    #X_train, X_val, Y_train, Y_val = split_train_test(tweets, labels, 1)

    # BERT
    #tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    #X_train = tokenize_tweets(X_train.tolist(), tokenizer)
    #X_val = tokenize_tweets(X_val.tolist(), tokenizer)

    #batch_size = 4

    #train_dataset = TensorDataset(torch.tensor(X_train['input_ids']), torch.tensor(X_train['attention_mask']), torch.tensor(Y_train))
    #test_dataset = TensorDataset(torch.tensor(X_val['input_ids']), torch.tensor(X_val['attention_mask']), torch.tensor(Y_val))

    #train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size)

    #model = train_bert(train_loader)
    #test_bert(test_loader, model)


    #print([tweet['input_ids'][0] for tweet in X_train])
    #print(torch.tensor([tweet['input_ids'][0] for tweet in X_train]))
    #print(torch.from_numpy(Y_train.astype(np.float32)).size())
    #train_dataset = TensorDataset(torch.tensor([tweet['input_ids'][0] for tweet in X_train]), torch.from_numpy(Y_train.astype(np.float32)))

    #input_ids_val = pad_sequence([tweet['input_ids'] for tweet in X_val], batch_first=True, padding_value=0)
    #attention_masks_val = pad_sequence([tweet['attention_mask'] for tweet in X_val], batch_first=True, padding_value=0)
    #test_dataset = TensorDataset(input_ids_val, torch.from_numpy(Y_val.astype(np.float32)))  
   
    #train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # GLOVE
    #X_train = all_tweets_to_glove(X_train,  GLOVE_TWEET_200D, 200)
    # Save the list to a file
    #with open('glove_train.pkl', 'wb') as f:
    #    pickle.dump(X_train, f)
    #X_val = all_tweets_to_glove(X_val, GLOVE_TWEET_200D, 200)
    #with open('glove_val.pkl', 'wb') as f:
    #    pickle.dump(X_val, f)

    # TF-IDF
    #vectorizer = TfidfVectorizer(max_features=5000)

    #X_train = vectorizer.fit_transform(X_train).toarray()
    #X_val = vectorizer.transform(X_val).toarray()

    #_, num_features = X_train.shape

    #train_dataset = TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(Y_train.astype(np.float32)))
    #test_dataset = TensorDataset(torch.from_numpy(X_val.astype(np.float32)), torch.from_numpy(Y_val.astype(np.float32)))

    #train_dataset = TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(Y_train))
    #test_dataset = TensorDataset(torch.from_numpy(X_val.astype(np.float32)), torch.from_numpy(Y_val))

    # NN WITH GLOVE
    #train_dataset = TensorDataset(torch.tensor(X_train).to(torch.float32), torch.tensor(Y_train).to(torch.float32))
    #test_dataset = TensorDataset(torch.tensor(X_val).to(torch.float32), torch.tensor(Y_val).to(torch.float32))

    # RNN
    #batch_size = 2
    #batch_size = 32

    #tokens_train, tokens_val = get_tokens_rnn(X_train, X_val)
    #train_dataset = TensorDataset(torch.tensor(tokens_train, dtype=torch.long), torch.from_numpy(Y_train.astype(np.float32)))
    #test_dataset = TensorDataset(torch.tensor(tokens_val, dtype=torch.long), torch.from_numpy(Y_val.astype(np.float32)))    

    #train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size)

    #model = train_rnn(train_loader)
    #test_rnn(test_loader, model)


    # NEURAL NETWORK
    #batch_size = 32
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size)

    #model = train_simple_nn(train_loader, 200)
    #test_simple_nn(test_loader, model)

if __name__ == '__main__':
    main()