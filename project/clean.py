""" this file is for data cleaning """

import pandas as pd
import re

from nltk import SnowballStemmer
from nltk.corpus import stopwords

LOAD_FILE_NAME = '../data/training.1600000.processed.noemoticon.csv'
STORE_FILE_NAME = '../data/tweets_processed_16000.csv'
DATASET_COLUMNS = ["target", "id", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TEXT_CLEANING_RE = r'@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+|[0-9]+'
TEST = True   # if TEST=True then fetch a smaller dataset


def read_tweets():
    tweets = pd.read_csv(LOAD_FILE_NAME, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
    if TEST:
        tweets = pd.concat([tweets.head(int(len(tweets) / 100)), tweets.tail(int(len(tweets) / 100))])
    print(tweets['text'])
    print("Read csv successfully, number of tweets: " + str(len(tweets)) + '\n')
    return tweets


def preprocess(tweet):
    stop_words = stopwords.words("english")
    stemmer = SnowballStemmer("english")
    tweet = re.sub(TEXT_CLEANING_RE, ' ', str(tweet).lower()).strip()
    tokens = []
    for token in tweet.split():
        if token not in stop_words:
            tokens.append(stemmer.stem(token))
    return " ".join(tokens)


if __name__ == '__main__':
    tweets = read_tweets()
    tweets['text'] = tweets['text'].apply(lambda x: preprocess(x))
    print(tweets.head())
    tweets.to_csv(STORE_FILE_NAME, encoding=DATASET_ENCODING)
