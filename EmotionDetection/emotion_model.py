import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
import re
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model

class EmotionModel:

    max_len = None

    def load_dataset(self, filename):
        df = pd.read_csv(filename)
        label = df["label"]
        unique_label = list(set(label))
        sentences = list(df["text"])
        return (df, label, unique_label, sentences)

    def cleaning(self, sentences):
        words = []
        for s in sentences:
            clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
            w = word_tokenize(clean)
            words.append([i.lower() for i in w])
        return words  

    def create_tokenizer(self, words, filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'):
        token = Tokenizer(filters = filters)
        token.fit_on_texts(words)
        return token
    
    def max_length(self, words):
        return(len(max(words, key = len)))

    def encoding_doc(self, token, words):
        return(token.texts_to_sequences(words))

    def padding_doc(self, encoded_doc, max_length):
        return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))

    def main(self):
        _, _, _, sentences = self.load_dataset("EmotionDetection/data/iseardataset.csv")
        
        cleaned_words = self.cleaning(sentences)
        word_tokenizer = self.create_tokenizer(cleaned_words)
        # vocab_size = len(word_tokenizer.word_index) + 1

        global max_len
        max_len = self.max_length(cleaned_words)

        model = load_model("EmotionDetection/model.h5")

        return word_tokenizer, model

    
    ##### PREDICTIONS !!!!!!!!!!!!!!

    def predictions(self, text, word_tokenizer, model_emotion):

        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
        test_word = word_tokenize(clean)
        test_word = [w.lower() for w in test_word]
        test_ls = word_tokenizer.texts_to_sequences(test_word)
        #Check for unknown words
        if [] in test_ls:
            test_ls = list(filter(None, test_ls))
        test_ls = np.array(test_ls).reshape(1, len(test_ls))
        x = self.padding_doc(test_ls, max_len)
        pred = model_emotion.predict(x)
        return pred

    def get_final_output(self, pred, classes):
        predictions = pred[0]
        classes = np.array(classes)
        ids = np.argsort(-predictions)
        classes = classes[ids]
        predictions = -np.sort(-predictions)
        # for i in range(pred.shape[1]):
        #     print("%s has confidence = %s" % (classes[i], (predictions[i])))
        return classes[0]

    def get_final_prediction(self, text, word_tokenizer, model_emotion):
        unique_label = ['disgust',
                        'fear',
                        'anger',
                        'joy',
                        'guilt',
                        'shame',
                        'sadness']
        pred = self.predictions(text, word_tokenizer, model_emotion)
        result = self.get_final_output(pred, unique_label)
        return result