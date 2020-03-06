"""
Auxiliary functions
"""
from gensim.models import KeyedVectors
import numpy as np

class PaddedW2V(object):
    def __init__(self, weights, index2word, word2index, vector_size):
        self.weights = weights
        self.index2word = index2word
        self.word2index = word2index
        self.vector_size = vector_size

    @classmethod
    def load(cls, path):
        def load_w2v_format(path):
            try:
                w2v = KeyedVectors.load_word2vec_format(path, binary=True)
                return w2v
            except Exception:
                return None

        def load_gensim(path):
            try:
                w2v = KeyedVectors.load(path)
                return w2v
            except Exception:
                return None

        loaders = [load_gensim, load_w2v_format]
        for loader in loaders:
            w2v = loader(path)
            if w2v != None:
                # adds UNK vector
                weights = np.vstack((np.zeros((1, w2v.vector_size)), w2v.vectors))

                index2word = {(i+1): word for (i, word) in enumerate(w2v.index2word)}
                index2word[0] = '\0'

                # word2index hashtable to speedup searches
                word2index = {word: (vocab_obj.index + 1) for (word, vocab_obj) in w2v.vocab.items()}
                word2index['\0'] = 0

                return cls(weights, index2word, word2index, w2v.vector_size)

        raise Exception("Invalid Word2Vec format!")
