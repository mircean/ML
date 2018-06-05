import numpy as np
import datetime
import util
import html2text

from sklearn import preprocessing

import spacy
nlp = spacy.load('en',  disable=['parser', 'ner'])
#TODO
#support for phrase verbs or multi word entities
#option to keep :), ?, !, etc. see w.is_punct

opt = {'text_pp_remove_symbols': False,
       'text_pp_lemmatization': False,
       'text_pp_remove_stop_words': False
       }

def cleanup_for_char_level(text):
    text = text.replace('\n', ' ')
    return text
    
def ignore_word(word):
    return False

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
        
    words = [util.normalize_text(w) for w in words]
    return words

#1014 characters always
#TODO: use UNK char? Or skip it? For now replace with ' ' (ID = 0)
#Remove very unfrequent chars?
def text2char(text, char2ids):
    text2 = np.zeros((len(char2ids), 1014))
    for i, c in enumerate(text[:1014]):
        if c in char2ids:
            text2[char2ids[c], i] = 1
    return text2

def documents2char(documents, char2ids):
    return [text2char(document, char2ids) for document in documents]

class Dataset:
    def __init__(self, options, df_train, df_test):
        self.df_train = df_train.copy()
        self.df_test = df_test.copy()
        self.features = [column for column in self.df_train.columns.values if column not in ['text', 'label', 'text_parsed', 'words']]

        self.le = None
        if 'numpy.int64' not in str(type(df_train.iloc[0].label)):
            self.le = preprocessing.LabelEncoder()
            self.le.fit(self.df_train.label)
            self.df_train['label'] = self.le.transform(self.df_train.label)
            self.df_test['label'] = self.le.transform(self.df_test.label)
            
        if 'text_parsed' not in df_train.columns:
            self.df_train = self.df_train.assign(text_parsed = self.df_train['text'].apply(lambda x: html2text_(x)))
            self.df_test = self.df_test.assign(text_parsed = self.df_test['text'].apply(lambda x: html2text_(x)))
            self.df_train = self.df_train.assign(words = self.df_train['text_parsed'].apply(lambda x: text2words(x)))
            self.df_test = self.df_test.assign(words = self.df_test['text_parsed'].apply(lambda x: text2words(x)))
            
        self.status = None

    def prepare_wordlevel(self, opt_trainer):
        print(datetime.datetime.now(), 'Loading Glove')
        glove_vocabulary = util.load_glove_vocabulary(opt_trainer)
        print(datetime.datetime.now(), 'Building vocabulary')
        self.vocabulary = util.build_vocabulary(self.df_train['words'].values, glove_vocabulary)
        print(datetime.datetime.now(), 'Preparing data')
        self.df_train['ids'] = self.df_train['words'].apply(lambda x: util.words2ids(x, self.vocabulary))
        self.df_test['ids'] = self.df_test['words'].apply(lambda x: util.words2ids(x, self.vocabulary))
        print(datetime.datetime.now(), 'Buidling embeddings')
        self.embeddings = util.build_embeddings(opt_trainer, self.vocabulary)
        
        self.status = 'WordLevel'
    
    def prepare_charlevel(self):
        print(datetime.datetime.now(), 'Prepare char level')
        #cleanup
        self.df_train['text_parsed'] = self.df_train['text_parsed'].apply(lambda x: x.lower())
        self.df_test['text_parsed'] = self.df_test['text_parsed'].apply(lambda x: x.lower())
        self.df_train['text_parsed'] = self.df_train['text_parsed'].apply(lambda x: cleanup_for_char_level(x))
        self.df_test['text_parsed'] = self.df_test['text_parsed'].apply(lambda x: cleanup_for_char_level(x))

        chars = {}
        for document in self.df_train['text_parsed'].values:
            for c in document:
                if c not in chars:
                    chars[c] = 0
                chars[c] += 1

        char_count = 0
        threshold = 0.0001
        for _, value in chars.items():
            char_count += value

        #build char set
        #space has code 0, used for padding
        self.char2ids = {' ': 0}
        for c, value in chars.items():
            if value/char_count > threshold:
                self.char2ids[c] = len(self.char2ids)

        self.status = 'CharLevel'
