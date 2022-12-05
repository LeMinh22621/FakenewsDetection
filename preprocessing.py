
import re
import gensim
import nltk
from nltk.corpus import stopwords


def toLowerCase(row):
    row = row.lower()
    return row


stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in stop_words:
            result.append(token)

    return result


def preprocessing(row):

    row = toLowerCase(row)

    row = preprocess(row)

    return ' '.join(row)
