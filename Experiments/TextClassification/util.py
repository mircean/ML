import numpy as np
import unicodedata

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def build_vocabulary(documents, glove_vocabulary=None):
    words = {'<PAD>': {'count': 0, 'id': 0}, '<UNK>': {'count': 0, 'id': 1}}
    ids = ['<PAD>', '<UNK>']
    words_ignored = {}

    for document in documents:
        for word in document:
            if word not in words:
                if glove_vocabulary == None:
                    words[word] = {'count': 0, 'id': len(words)}
                    ids.append(word)
                else:
                    if word in glove_vocabulary:
                        words[word] = {'count': 0, 'id': len(words)}
                        ids.append(word)
            if word in words:
                words[word]['count'] = words[word]['count'] + 1
            else:
                if word not in words_ignored:
                    words_ignored[word] = 0
                words_ignored[word] += 1

    return {'words': words, 'ids': ids, 'words_ignored': words_ignored}

def words2ids(words, vocabulary, unknown='ignore'):
    #word not in vocabulary if
    #a. test set
    #b. word not in glove
    #unknown can be
    #'ignore'
    #'unknown_id'
    #'fail'
    if unknown == 'fail':
        return [vocabulary['words'][word]['id'] for word in words]
    elif unknown == 'unkwown_id':
        return [vocabulary['words'][word]['id'] if word in vocabulary['words'] else vocabulary_unknown_id for word in words]
    else:
        return [vocabulary['words'][word]['id'] for word in words if word in vocabulary['words']]
        
def load_glove_vocabulary(opt):
    vocabulary = set()
    with open(opt['glove_file'], encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(''.join(elems[0:-opt['glove_dim']]))
            vocabulary.add(token)
    return vocabulary

def build_embeddings(opt, vocabulary):
    vocabulary_size = len(vocabulary['words'])
    embeddings = np.random.uniform(-1, 1, (vocabulary_size, opt['glove_dim']))
    embeddings[0] = 0 # <PAD> should be all 0 (using broadcast)

    with open(opt['glove_file'], encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(''.join(elems[0:-opt['glove_dim']]))
            if token in vocabulary['words']:
                embeddings[vocabulary['words'][token]['id']] = [float(v) for v in elems[-opt['glove_dim']:]]
    return embeddings


