import torch

from tweet_disaster import X, y, X_test, test_df_id
import gensim.downloader as api
import numpy as np

word2vec_model = api.load("word2vec-google-news-300")

def get_word_vectors(text):
    words = text.split()
    word_vectors = []

    for word in words:
        if word in word2vec_model:
            word_vectors.append(torch.tensor(word2vec_model[word]))
        else:
            word_vectors.append(torch.zeros(word2vec_model.vector_size))  # OOV words as zero vectors

    return torch.stack(word_vectors) if word_vectors else torch.zeros(1, word2vec_model.vector_size)

X = X.apply(get_word_vectors)
X_test = X_test.apply(get_word_vectors)
