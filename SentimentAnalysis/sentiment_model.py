import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from numpy import array, asarray, zeros
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences

class SentimentModel:

    def predictions(self, text, tokenizer):
        model_sentiment = load_model('/home/shiningflash/Documents/BONDHU-BOT/SentimentAnalysis/model_sentiment.m5')

        text1 = tokenizer.texts_to_sequences(text)
        text2 = pad_sequences(
                text1,
                padding = 'post',
                maxlen = 100
            )
        pred = model_sentiment.predict(text2)
        return 'positive' if pred[0] >= 0.5 else 'negative'

    def remove_html_tags(self, sen):
        TAG_RE = re.compile(r'<[^>]+>')
        return TAG_RE.sub('', sen)

    def data_processing_1(self, sen):
        sen = self.remove_html_tags(sen) # remove html tag
        sen = sen.replace('n\'t', ' not') # convert n't to not
        sen = re.sub(r"\s+[a-zA-Z]\s+", ' ', sen) # remove single letter
        sen = re.sub(r'\s+', ' ', sen) # remove multiple spaces
        sen = re.sub(r'[.]+', '.', sen) # remove multiple dots
        sen = sen.replace('\\\'', ' ') # remove \
        return sen

    def data_processing_2(self, sen):
        sen = sen.replace('&quot;3', '')
        sen = sen.replace('&quot;', '')
        sen = sen.replace('&lt;3', '')
        sen = sen.replace('&lt;', '')
        sen = sen.replace('&gt;', '')
        sen = re.sub('http[s]?://\S+', '', sen)
        sen = re.sub('[a-zA-Z0-9]*@[a-zA-Z0-9]*', '', sen)
        sen = sen.replace('an\'t', 'an not')
        sen = sen.replace('n\'t', ' not')
        sen = re.sub(r"\s+[A-Z]\s+", ' ', sen)
        sen = re.sub(r'[.]+', '.', sen)
        sen = re.sub(r'\s+', ' ', sen)
        sen = re.sub(r'[-]+', ' ', sen)
        return sen

    def main(self):
        df = []
        df.append(pd.read_csv('/home/shiningflash/Documents/BONDHU-BOT/SentimentAnalysis/data/IMDB Dataset.csv', nrows = 10000))
        df.append(pd.read_csv('/home/shiningflash/Documents/BONDHU-BOT/SentimentAnalysis/data/Sentiment Analysis Dataset 100000.csv', encoding = 'latin-1', nrows = 10000))
        df[1].drop(['ItemID'], axis = 'columns', inplace = True)

        X = [["0"]*10000]*3

        sentences = list(df[0]['text'])
        i = 0
        for sen in sentences:
            X[0][i] = self.data_processing_1(sen)
            i = i + 1
        X[0] = np.array(list(X[0]))
        y = [[0]*10000]*3
        y[0] = df[0]['sentiment']
        y[0] = np.array(list(map(lambda x: 1 if x == "positive" else 0, y[0])))

        sentences = list(df[1]['text'])
        i = 0
        for sen in sentences:
            X[1][i] = self.data_processing_2(sen)
            i = i + 1
        X[1] = np.array(list(X[1]))
        y[1] = np.array(list(df[1]['sentiment']))

        for i in range(0, 5000):
            X[2][i] = X[0][i]
            y[2][i] = y[0][i]
        for i in range(5000, 10000):
            X[2][i] = X[1][i]
            y[2][i] = y[1][i]
            
        X[2] = np.array(X[2])
        y[2] = np.array(y[2])

        X_train = [[]] * 3
        X_test = [[]] * 3
        y_train = [[]] * 3
        y_test = [[]] * 3

        for i in range(0, 3):
            X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(
                X[i], y[i], test_size = 0.25,
                random_state = 42
            )
        
        tokenizer = Tokenizer(num_words = 50000)

        for i in range(3):
            tokenizer.fit_on_texts(X_train[i])
        
        for i in range(3):
            X_train[i] = tokenizer.texts_to_sequences(X_train[i])
        
        for i in range(3):
            X_test[i] = tokenizer.texts_to_sequences(X_test[i])

        maxlen = 100
        vocab_size = len(tokenizer.word_index) + 1

        for i in range(3):
            X_train[i] = pad_sequences(
                X_train[i],
                padding = 'post',
                maxlen = maxlen
            )
        
        for i in range(3):
            X_test[i] = pad_sequences(
                X_test[i],
                padding = 'post',
                maxlen = maxlen
            )
        
        embed_dictionary = dict()
        glv_file = open('/home/shiningflash/Documents/BONDHU-BOT/SentimentAnalysis/data/glove.6B.100d.txt', encoding='utf8')

        for line in glv_file:
            records = line.split()
            word = records[0]
            vector_dim = asarray(records[1:], dtype='float32')
            embed_dictionary[word] = vector_dim
        glv_file.close()

        embed_matrix = zeros((vocab_size, 100))

        for word, index in tokenizer.word_index.items():
            embed_vector = embed_dictionary.get(word)
            if embed_vector is not None:
                embed_matrix[index] = embed_vector

        return tokenizer