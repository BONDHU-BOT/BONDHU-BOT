import json
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
np.random.seed(0)

class NERModel:

    class SentenceGetter(object):
        def __init__(self, data):
            self.n_sent = 1
            self.data = data
            agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                            s["POS"].values.tolist(),
                                                            s["Tag"].values.tolist())]
            self.grouped = self.data.groupby("Sentence #").apply(agg_func)
            self.sentences = [s for s in self.grouped]

    global num_words
    global model_ner
    global words
    global word2idx

    def text_process(self, text):
        seq = []
        for w in text.split():
            try:
                seq.append(word2idx[w])
            except:
                seq.append(num_words-2)

        sz = len(seq)
        for _ in range(sz, 50):
            seq.append(num_words-2)
        seq = np.array(seq, dtype=int)
        return seq, sz

    def get_prediction(self, text):
        tags = list(
           ['O',
            'B-per',
            'I-org',
            'B-art',
            'B-nat',
            'I-nat',
            'B-eve',
            'B-geo',
            'I-tim',
            'B-tim',
            'I-gpe',
            'I-eve',
            'B-gpe',
            'I-art',
            'B-org',
            'I-geo',
            'I-per'])

        seq, sz = self.text_process(text)
        p = model_ner.predict(np.array([seq]))
        p = np.argmax(p, axis=-1)

        idx = 0
        ret_ner = []
        for w, pred in zip(seq, p[0]):
            if tags[pred] != 'O':
                print("{:15}\t{}".format(words[w-1], tags[pred]))
                ans = words[w-1] + " = " + tags[pred]
                ret_ner.append(ans)
            idx += 1
            if idx == sz:
                break
        return str(ret_ner)

    def main(self):
        global num_words
        global model_ner
        global words
        global word2idx

        # data = pd.read_csv('NamedEntityRecognition/data/ner_dataset.csv', encoding='latin1')
        # data = data.fillna(method='ffill')
        # words = list(set(data["Word"].values))
        
        with open('NamedEntityRecognition/word_file.txt', 'r') as f:
            words = json.loads(f.read())

        words = list(words)

        words.append("ENDPAD")
        num_words = len(words)

        word2idx = {w: i+1 for i, w in enumerate(words)}

        model_ner = load_model("NamedEntityRecognition/model_ner.h5")