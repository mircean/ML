import numpy as np
import unicodedata
import html2text

import spacy
nlp = spacy.load('en')

#TODO
#support for phrase verbs or multi word entities
#option to keep :), ?, !, etc. see w.is_punct

opt = {'text_pp_remove_symbols': False,
       'text_pp_lemmatization': False,
       'text_pp_remove_stop_words': False
       }

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
    if opt['text_pp_remove_stop_words']:
        words = [w for w in words if not w.is_stop ]
        
    words = [w for w in words if not ignore_word(w)]
    
    if opt['text_pp_lemmatization']:
        words = [w.lemma_ if w.lemma_ != '-PRON-' else w.lower_ for w in words]
    else:
        words = [w.lower_ for w in words]
        
    words = [normalize_text(w) for w in words]
    return words

class Dataset:
    def __init__(self, options, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test
        
    def preprocess(self):
        self.df_train['text_parsed'] = self.df_train['text'].apply(lambda x: html2text_(x))
        self.df_test['text_parsed'] = self.df_test['text'].apply(lambda x: html2text_(x))
        self.df_train['words'] = self.df_train['text_parsed'].apply(lambda x: text2words(x))
        self.df_test['words'] = self.df_test['text_parsed'].apply(lambda x: text2words(x))
    
