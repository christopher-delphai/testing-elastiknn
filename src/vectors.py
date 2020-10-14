""" 
just a wrapper to get some vectors from Word2Vec.
"""

from gensim.models import KeyedVectors
import numpy as np

model = KeyedVectors.load("src/keyvectors/concepts_descrs_wiki_v5.kv", mmap='r')


def get_vocab(model):
    """ returns all words from the model vocabulary."""
    words_keys = list(model.wv.vocab.keys())
    return words_keys


def get_vector(word):
    return model[word].tolist()


def generate_vectors():
    all_words = get_vocab(model)
    vecs_dict = {}
    for word in all_words:
        vecs_dict[word] = model[word].tolist()
        # if len(vecs_dict) == 5:
        #     break
    return vecs_dict


def get_similars(word):
    return model.most_similar(word, topn=10)


def get_random_vecs(count, dim=200):
    # returns random vectors
    vecs_dict = {}
    vectors = np.random.rand(count, 200).tolist()
    for i in range(count):
        name = f'random_vec-{i}'
        vecs_dict[name] = vectors[i]
    return vecs_dict

if __name__ == "__main__":
    get_vocab(model)