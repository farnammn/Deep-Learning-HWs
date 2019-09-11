import numpy as np
import gensim


class Word2Vec_Loader:
    def __init__(self,base_directory,  word2vec_file='GoogleNews-vectors-negative300.bin'):
        self.word2vec_file = base_directory + word2vec_file

        self.load_word2vec()
        print("wor2vec successfuly loaded")

    def load_word2vec(self):
        '''
        loading the word2vec matrices
        '''
        self.init_embedding_w = gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_file, binary=True)
        self.embedding_size = len(self.init_embedding_w["Deep"])

    def return_word2vec(self, words_batch, mask_batch):
        '''
        :param words_batch: the batch of words
        :return vectors corresponding to them: with shape [batch_size, word2vec vector size]
        '''
        out = np.zeros((len(words_batch), self.embedding_size))
        for i, word in enumerate(words_batch):
            if( mask_batch[i] == 1 and word in self.init_embedding_w):
                out[i] = self.init_embedding_w[word]
            else:
                out[i] = np.zeros((self.embedding_size))
        return np.array(out)
