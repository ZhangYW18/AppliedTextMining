import pandas as pd
import numpy as np
import gensim

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

LOAD_FILE_NAME = '../data/tweets.csv'
DATASET_COLUMNS = ["target", "id", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
DATASET_SIZE = 32000

EPOCHS = 8
BATCH_SIZE = 1024
SEQUENCE_LENGTH = 300

W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

TEST_PERCENTAGE = 0.1

TEST = True  # if TEST=True then fetch a smaller dataset


def read_tweets():
    if TEST:
        tweets = pd.read_csv(LOAD_FILE_NAME, encoding=DATASET_ENCODING, names=DATASET_COLUMNS, nrows=DATASET_SIZE + 1)
    else:
        tweets = pd.read_csv(LOAD_FILE_NAME, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
    print(tweets['text'])
    print("Read csv successfully, number of tweets: " + str(len(tweets)) + '\n')
    return tweets


def train():
    data = read_tweets()
    documents = [str(_text).split() for _text in data.text]
    documents = documents[1:]
    l_index = int(len(documents) * (0.5 - TEST_PERCENTAGE / 2))
    r_index = int(len(documents) * (0.5 + TEST_PERCENTAGE / 2))
    data_train = documents[:l_index] + documents[r_index:]
    data_test = documents[l_index:r_index]
    print("TRAIN size:", len(data_train))
    print("TEST size:", len(data_test))
    targets = data.target
    targets = targets[0:]
    targets = targets.tolist()
    targets = targets[:l_index] + targets[r_index:]
    targets_test = targets[l_index:r_index]
    w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE,
                                                window=W2V_WINDOW,
                                                min_count=W2V_MIN_COUNT,
                                                workers=8)
    w2v_model.build_vocab(documents)
    w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
    print()
    print("model_test (most similar words of `sad`) :\n")
    print(w2v_model.most_similar("sad"))
    print()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data_train)

    x_train = pad_sequences(tokenizer.texts_to_sequences(data_train), maxlen=SEQUENCE_LENGTH)
    x_test = pad_sequences(tokenizer.texts_to_sequences(data_test), maxlen=SEQUENCE_LENGTH)
    #    labels = targets.unique().tolist()
    encoder = LabelEncoder()
    encoder.fit(targets)
    y_train = encoder.transform(targets)
    y_test = encoder.transform(targets_test)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    print("x_train", x_train.shape)
    print("y_train", y_train.shape)
    print("x_test", x_train.shape)
    print("y_test", y_train.shape)
    print()

    vocab_size = len(tokenizer.word_index) + 1
    print("Total words", vocab_size)
    print("Vocab size", vocab_size)
    print()
    embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
    cnt = int(0)
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    print("embedding_matrix", embedding_matrix.shape)
    print()
    print()

    model = Sequential()
    embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH,
                                trainable=False)
    model.add(embedding_layer)
    model.add(Dropout(0.5))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='relu'))
    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
                 EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]
    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_split=0.1,
                        verbose=1,
                        callbacks=callbacks)

    score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    print()
    print("ACCURACY:", score[1])
    print("LOSS:", score[0])


if __name__ == '__main__':
    np.random.seed(5261)
    train()
