import re
from itertools import groupby
from utils import *
import string
from wordsegment import load, segment
import numpy as np


# Delete tweets that appear more than once
def delete_duplicates(tweets, labels):
    seen = set()
    indices = []
    for i in range(len(tweets)):
        if tuple(tweets[i]) not in seen:
            indices.append(i)
            seen.add(tuple(tweets[i]))
    return [tweets[i] for i in indices], [labels[i] for i in indices]

# Remove user mentions "<user>"
def remove_user_mentions(tweets):
    return [word for word in tweets if word != "<user>"]

# Remove hashtags
def remove_hashtags(tweets):
    return [[word[1:] if word.startswith("#") else word for word in tweet] for tweet in tweets]

# If more than 2 consecutive characters are the same, keep to 2
def remove_repeated(tweets):
    return [[re.sub(r'(.)\1{2,}', r'\1\1', word) for word in tweet] for tweet in tweets]
    # return [[re.sub(r'(.)\1+', r'\1', word) for word in tweet] for tweet in tweets]
    # return [[re.sub(r'(.)\1{2,}', '', word) for word in tweet] for tweet in tweets]

# Remove punctuation 
def remove_punctuation(tweets):
    return [[word.translate(str.maketrans('', '', string.punctuation)) for word in tweet] for tweet in tweets]

# sets all words in lowercase
def lowercase(tweets):
     return [[w.lower().replace("\n", "") for w in tweet] for tweet in tweets]

USER_URL = ["<user>", "<url>"]
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
# Filter words in tweets given a set of words to remove
def remove_words(tweets, words_to_remove):
    output = []

    for tweet in tweets:
        filtered_words = [word for word in tweet if word not in words_to_remove]
        output.append(filtered_words)
    
    return output

NEGATIONS = ["not", "no", "never", "without", "hardly", "rarely", "seldom"]
EXAMPLE_POSITIVE = ["happy", "good", "love", "great", "wonderful", "beautiful", "celebrate", "best", "terrific", "delighted", "joy"]
EXAMPLE_NEGATIVE = ["sad", "bad", "hate", "terrible", "awful", "ugly", "mourn", "worst", "awful", "disappointed", "sorrow"]
# Replace negations with corresponding word ("not happy" -> "sad")
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

# replace emoticon with a word
def replace_emoticon(word):
    if "<3" in word:
        return "love"
    
    if ":)" in word or "(:" in word or ":-)" in word or ":-D" in word or ":D" in word:
        return "happy"
    
    if ":(" in word or "):" in word or ":/" in word or ":'(" in word:
        return "sad"
    
    if ":-P" in word:
        return "playful"
    
    if "XD" in word:
        return "laugh"
    
    return word

# replace in all tweets emoticons with corresponding word
def replace_emoticons(tweets):
    return [[replace_emoticon(word) for word in tweet] for tweet in tweets]

# Remove hashtags and then segment words ("#iloveyou" -> "i love you")
def process_hashtags(tweets):    
    processed_tweets = []
    for tweet in tweets:
        processed_tweet = []
        for word in tweet:
            if word.startswith('#'):
                word_without_hash = word[1:]
                if word_without_hash:  # Ensure the word is not empty
                    try:
                        segmented_words = " ".join(segment(word_without_hash))
                    except ValueError:  # Handle the case where segment returns an empty sequence
                        segmented_words = word_without_hash  # Fallback to the original word without hash
                    processed_tweet.extend(segmented_words.split())
                else:
                    processed_tweet.append(word)
            else:
                processed_tweet.append(word)
        processed_tweets.append(processed_tweet)
    return processed_tweets

ABBREVIATIONS = {
    "dunno": "do not know",
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
    "fml": "fuck my life",
    "x": "kisses",
    "xo": "kisses and hugs",
    "xoxo": "kisses and hugs",
    "thx": "thanks"
}
# expand abberviations in tweets
def expand_abbreviations(tweets, abbreviation_dict):
    return [[abbreviation_dict.get(word, word) for word in tweet] for tweet in tweets]

# when splitting a tweet, remove null strings
def remove_null_strings(tweets):
    outputs = []
    for tweet in tweets:
        cleaned = []
        for word in tweet:
            if(word):
                cleaned.append(word)
        outputs.append(cleaned)
    return outputs

# pad tweets to same length
def pad_tweets(tweets):
    max_len = max(len(tweet) for tweet in tweets)
    def pad_tweet(tweet, max_len, pad_token="<PAD>"):
        return tweet + [pad_token] * (max_len - len(tweet))

    return [pad_tweet(tweet, max_len) for tweet in tweets]

# pre-process all tweets with previous auxiliary methods
def preprocess(tweets, labels):
    load()

    tweets = [tweet.split() for tweet in tweets]

    tweets, labels = delete_duplicates(tweets, labels)
    tweets = lowercase(tweets)

    tweets = remove_user_mentions(tweets)
    tweets = process_hashtags(tweets)
    tweets = replace_emoticons(tweets)
    tweets = remove_repeated(tweets)

    # tweets = handle_negation(tweets, NEGATIONS, EXAMPLE_POSITIVE, EXAMPLE_NEGATIVE)
    #tweets = remove_punctuation(tweets)

    tweets = remove_words(tweets, USER_URL)

    # really bad for performance!
    #tweets = remove_words(tweets, NEUTRAL_WORDS)

    tweets = expand_abbreviations(tweets, ABBREVIATIONS)

    tweets = remove_null_strings(tweets)
    #tweets = pad_tweets(tweets)

    return tweets, labels

def main():
    sample_tweets = [
    ["worries", "fml", "tooo", "seeee", "youuuu"],
    ["thiiis", "is", "aaamazing", "and", "coooool"],
    ["whaaaaat", "a", "beauuutiful", "daaaay"],
    ["heyy", "theeerreeee", "what's", "uuup"]
    ]

    # Applying the function to the sample tweets
    processed_tweets = remove_repeated(sample_tweets)
    processed_tweets = expand_abbreviations(processed_tweets,ABBREVIATIONS)

    # Display the results
    for original, processed in zip(sample_tweets, processed_tweets):
        print("Original:", original)
        print("Processed:", processed)
        print()

if __name__ == '__main__':
    main()