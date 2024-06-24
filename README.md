# Whitewashed - ETHZ CIL Text Classification 2024 

The use of microblogging and text messaging as a medium of communication has greatly increased over the past 10 years. Such large volumes of data amplify the need for automatic methods to understand the opinion conveyed in a text.

## Training Data
For this problem, we have acquired 2.5M tweets classified as either positive or negative.

## Evaluation Metrics
Your approach is evaluated according to the following criteria: Classification Accuracy

## Build the Co-occurence Matrix
To build a co-occurence matrix, run the following commands.  (Remember to put the data files
in the correct locations)

Note that the cooc.py script takes a few minutes to run, and displays the number of tweets processed.

- build_vocab.sh
- cut_vocab.sh
- python3 pickle_vocab.py
- python3 cooc.py

##  Template for Glove Question
Your task is to fill in the SGD updates to the template
glove_template.py

Once you tested your system on the small set of 10% of all tweets, we suggest you run on the full datasets train_pos_full.txt, train_neg_full.txt

## Instructions on how to download the remaining code/data (if needed)
