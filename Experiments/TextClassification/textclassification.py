import numpy as np
import unicodedata
import html2text

import spacy
nlp = spacy.load('en')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#TODO
#support for phrase verbs or multi word entities
#option to keep :), ?, !, etc. see w.is_punct


text_pp_remove_symbols = False
text_pp_lemmatization = False
text_pp_remove_stop_words = False
bow_tfidf = True
bow_ngram_range = (1,2)
bow_min_df = 5

def ignore_word(word):
    return False

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def html2text_(html):
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_emphasis = True
    h.body_width = 0
    text = h.handle(html)
    text = text.strip('\r\n')
    return text
    
def text2words(text):
    #this works better for imdb. not sure why quotes are not handled inconsistently by spacy
    text = text.replace('"', ' ')
    
    doc = nlp(text)
    
    words = [w for w in doc if not w.is_space and not w.is_bracket and not w.is_punct]
    #words = [w for w in doc if not w.is_space]
    if text_pp_remove_stop_words:
        words = [w for w in words if not w.is_stop ]
        
    words = [w for w in words if not ignore_word(w)]
    
    if text_pp_lemmatization:
        words = [w.lemma_ if w.lemma_ != '-PRON-' else w.lower_ for w in words]
    else:
        words = [w.lower_ for w in words]
        
    words = [normalize_text(w) for w in words]
    return words

def build_vocabulary(rows):
    words = {}
    ids = []
    for row in rows:
        for word in row:
            if word not in words:
                words[word] = {'count': 0, 'id': len(words)}
                ids.append(word)
            words[word]['count'] = words[word]['count'] + 1

    return {'words': words, 'ids': ids}

def words2ids(words, vocabulary, mode='train'):
    if mode == 'train':
        return [vocabulary['words'][word]['id'] for word in words]
    else:
        return [vocabulary['words'][word]['id'] for word in words if word in vocabulary['words']]
        
def BOW(X_train, X_test, y_train, y_test):
    if bow_tfidf == False:
        cv = CountVectorizer(ngram_range=bow_ngram_range, min_df=bow_min_df)
        X_train = cv.fit_transform(X_train)
        X_test = cv.transform(X_test)
    else:
        tfidf = TfidfVectorizer(ngram_range=bow_ngram_range, min_df=bow_min_df)
        X_train = tfidf.fit_transform(X_train)
        X_test = tfidf.transform(X_test)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    y_predict = lr.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_predict)
    y_predict = lr.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_predict)
    return (accuracy_train, accuracy_test)
 