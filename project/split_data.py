# Fetch

import pandas as pd

LOAD_FILE_NAME = '../data/training.1600000.processed.noemoticon.csv'
STORE_FILE_NAME = '../data/training.csv'
DATASET_SIZE = 200000
DATASET_SIZE_DEFAULT = True
DATASET_COLUMNS = ["target", "id", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

if __name__ == '__main__':
    tweets = pd.read_csv(LOAD_FILE_NAME, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
    lines = int(len(tweets) / 100) if DATASET_SIZE_DEFAULT else DATASET_SIZE
    tweets = pd.concat([tweets.head(lines), tweets.tail(lines)])
    tweets.to_csv(STORE_FILE_NAME, encoding=DATASET_ENCODING)
