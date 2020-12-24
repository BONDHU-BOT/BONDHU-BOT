import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import nltk
import re
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model

class IntentModel:

    max_len = None

    def load_dataset(self, filename, Sentence, Intent):
        df = pd.read_csv(filename, names = [Sentence, Intent])
        intent = df[Intent]
        unique_intent = list(set(intent))
        sentences = list(df[Sentence])
        return (df, intent, unique_intent, sentences)

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
    
    def one_hot(self, encode):
        o = OneHotEncoder(sparse = False)
        return(o.fit_transform(encode))

    def main(self):
        global unique_intent
        df, intent, unique_intent, sentences = self.load_dataset("/home/shiningflash/Documents/BONDHU-BOT/IntentClassification/data/Dataset.csv", "text", "category")

        nltk.download("stopwords")
        nltk.download("punkt")
        
        cleaned_words = self.cleaning(sentences)
        word_tokenizer = self.create_tokenizer(cleaned_words)
        # vocab_size = len(word_tokenizer.word_index) + 1
        global max_len
        max_len = self.max_length(cleaned_words)

        # print("Vocab Size = %d and Maximum length = %d" % (vocab_size, max_len))

        encoded_doc = self.encoding_doc(word_tokenizer, cleaned_words)
        padded_doc = self.padding_doc(encoded_doc, max_len)
        output_tokenizer = self.create_tokenizer(unique_intent, filters = '!"#$%&()*+,-/:;<=>?@[\]^`{|}~')
        encoded_output = self.encoding_doc(output_tokenizer, intent)
        encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)
        output_one_hot = self.one_hot(encoded_output)

        model = load_model("/home/shiningflash/Documents/BONDHU-BOT/IntentClassification/data/model.h5")

        return word_tokenizer, model, unique_intent

    
    ##### PREDICTIONS !!!!!!!!!!!!!!

    def predictions(self, text, word_tokenizer, model_intent):

        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
        test_word = word_tokenize(clean)
        test_word = [w.lower() for w in test_word]
        test_ls = word_tokenizer.texts_to_sequences(test_word)
        #Check for unknown words
        if [] in test_ls:
            test_ls = list(filter(None, test_ls))
        test_ls = np.array(test_ls).reshape(1, len(test_ls))
        x = self.padding_doc(test_ls, max_len)
        pred = model_intent.predict(x)
        return pred

    def get_final_output(self, pred, classes):
        predictions = pred[0]
        # print(predictions)
        classes = np.array(classes)
        ids = np.argsort(-predictions)
        classes = classes[ids]
        predictions = -np.sort(-predictions)
        # for i in range(pred.shape[1]):
        #     print("%s has confidence = %s" % (classes[i], (predictions[i])))
        return classes[0]

    def get_final_prediction(self, text, word_tokenizer, model_intent, unique_intent):
        pred = self.predictions(text, word_tokenizer, model_intent)
        result = self.get_final_output(pred, unique_intent)
        return result

    

