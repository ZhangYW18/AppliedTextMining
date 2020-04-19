import os
import time
import pickle
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from clean import preprocess


POSITIVE = 0
NEGATIVE = 1
NEUTRAL = 2

SEQUENCE_LENGTH = 300


def decode_sentiment(score, include_neutral=True, threshold=0.6):
    if include_neutral:
        label = NEUTRAL
        if score <= threshold:
            label = NEGATIVE
        elif score >= 1-threshold:
            label = POSITIVE
        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE


def predict_and_train(input, include_neutral=True):
    # predict
    input_processed = preprocess(input)
    text = str(input_processed).split()
    tokenizer = pickle.load(open('./models/tokenizer.pkl', 'rb'))
    model = load_model('./models/model.h5')
    start_at = time.time()
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    score = model.predict([x_test])[0]
    print("score: ", score)
    label = decode_sentiment(score, include_neutral=include_neutral)
    # train
    return {"label": label, "score": float(score),
            "elapsed_time": time.time() - start_at}


if __name__ == '__main__':
    predict_and_train("I feel happy")
    predict_and_train("I feel sad")
    predict_and_train("i don't know what i'm doing")
