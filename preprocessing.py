import re
from itertools import groupby
from utils import *
import string

def delete_duplicates(tweets, labels):
    seen = set()
    indices = []
    for i in range(len(tweets)):
        if tuple(tweets[i]) not in seen:
            indices.append(i)
            seen.add(tuple(tweets[i]))
    return [tweets[i] for i in indices], [labels[i] for i in indices]

def remove_repeated(tweets):
    return [[re.sub(r'(.)\1{2,}', '', word) for word in tweet] for tweet in tweets]

def remove_punctuation(tweets):
    return [[word.translate(str.maketrans('', '', string.punctuation)) for word in tweet] for tweet in tweets]

def lowercase(tweets):
     return [[w.lower().replace("\n", "") for w in tweet] for tweet in tweets]

USER_URL = ["<user>", "url"]
NEUTRAL_WORDS = [
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "between",
    "both", "but", "by", "can", "could", "did", "didn’t", "do", "doesn’t", "doing",
    "don’t", "down", "during", "each", "few", "for", "from", "further", "had", 
    "hadn’t", "has", "hasn’t", "have", "haven’t", "having", "he", "her", "here", 
    "him", "his", "how", "I", "if", "in", "into", "is", "isn’t", "it", "its", 
    "just", "like", "make", "me", "might", "more", "most", "mustn’t", "my", "no", 
    "of", "off", "on", "once", "only", "or", "other", "our", "out", "over", 
    "said", "same", "see", "should", "shouldn’t", "so", "some", "such", "than", 
    "that", "the", "their", "them", "then", "there", "these", "they", "this", 
    "those", "through", "to", "too", "under", "until", "up", "very", "was", 
    "wasn’t", "we", "were", "weren’t", "what", "when", "where", "which", "while", 
    "who", "whom", "why", "will", "with", "would", "wouldn’t", "you", "your", "yours"
]

def remove_words(tweets, words_to_remove):
    output = []

    for tweet in tweets:
    
        filtered_words = [word for word in tweet if word not in words_to_remove]
    
        output.append(filtered_words)
    
    return output

NEGATIONS = ["not", "no", "never", "without", "hardly", "rarely", "seldom"]
EXAMPLE_POSITIVE = ["happy", "good", "love", "great", "wonderful", "beautiful", "celebrate", "best", "terrific", "delighted", "joy"]
EXAMPLE_NEGATIVE = ["sad", "bad", "hate", "terrible", "awful", "ugly", "mourn", "worst", "awful", "disappointed", "sorrow"]

def handle_negation(tweets, negations, positives, negatives):
    output = []
    for tweet in tweets:
        copy_tweet = tweet.copy()

        for i in range(len(tweet)-1):
            if tweet[i] in negations and tweet[i+1] in positives:
                copy_tweet[i] = ""
                idx = positives.index(tweet[i+1])
                copy_tweet[i+1] = negatives[idx]
            if tweet[i] in negations and tweet[i+1] in negatives:
                copy_tweet[i] = ""
                idx = negatives.index(tweet[i+1])
                copy_tweet[i+1] = positives[idx]
        
        output.append(copy_tweet)
    return output

ABBREVIATIONS = {
    "omg": "oh my god",
    "lol": "laugh out loud",
    "idk": "i don't know",
    "tbh": "to be honest",
    "smh": "shaking my head",
    "rip": "rest in peace",
    "brb": "be right back",
    "btw": "by the way",
    "imho": "in my humble opinion",
    "imo": "in my opinion",
    "x": "kisses",
    "xo": "kisses and hugs",
    "xoxo": "kisses and hugs",
}

def expand_abbreviations(tweets, abbreviation_dict):
    return [[abbreviation_dict.get(word, word) for word in tweet] for tweet in tweets]

def remove_null_strings(tweets):
    outputs = []
    for tweet in tweets:
        cleaned = []
        for word in tweet:
            if(word):
                cleaned.append(word)
        outputs.append(cleaned)
    return outputs

def pad_tweets(tweets):
    max_len = max(len(tweet) for tweet in tweets)

    def pad_tweet(tweet, max_len, pad_token="<PAD>"):
        return tweet + [pad_token] * (max_len - len(tweet))

    return [pad_tweet(tweet, max_len) for tweet in tweets]

 
def preprocess(tweets, labels):
    tweets = [tweet.split() for tweet in tweets]

    tweets, labels = delete_duplicates(tweets, labels)
   
    tweets = remove_repeated(tweets)
    tweets = lowercase(tweets)
    tweets = remove_punctuation(tweets)

    tweets = handle_negation(tweets, NEGATIONS, EXAMPLE_POSITIVE, EXAMPLE_NEGATIVE)

    tweets = remove_words(tweets, USER_URL)

    # really bad for performance!
    #tweets = remove_words(tweets, NEUTRAL_WORDS)

    tweets = expand_abbreviations(tweets, ABBREVIATIONS)

    tweets = remove_null_strings(tweets)
    tweets = pad_tweets(tweets)

    return tweets, labels

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
    
    tweets, labels = preprocess(tweets, labels)

if __name__ == '__main__':
    main()



