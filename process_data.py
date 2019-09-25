"""
Code has been adapted from https://github.com/yoonkim/CNN_sentence

An additional command line argument was added to point to the data folder.
"""

import pickle
from collections import defaultdict
import sys
import re

import gensim
import numpy as np


def build_data_cv(neg_file, pos_file, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    reviews = []
    vocab = defaultdict(float)
    for positivity, file in enumerate([neg_file, pos_file]):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if clean_string:
                    orig_rev = clean_str(line)
                else:
                    orig_rev = line.lower()
                words = orig_rev.split()
                unique_words = set(words)
                for word in unique_words:
                    vocab[word] += 1
                datum = {
                    "y": positivity,
                    "text": orig_rev,
                    "num_words": len(words),
                    "split": np.random.randint(0, cv),
                }
                reviews.append(datum)
    return reviews, vocab


def build_embedding_matrix(model_bin_path, vocab, min_threshold=1):
    """
    Builds the embedding matrix for this vocabulary.
    This uses the Gensim to load the Google News word2vec file which may
    take a while.

    For words that occur in at least min_df documents, create a separate
    word vector. 0.25 is chosen so the unknown vectors have (approximately)
    same variance as pre-trained ones.

    The w2v file is downloadable here:
    https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
    """
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
        model_bin_path, binary=True
    )
    embedding_dimension = w2v_model.vector_size
    word2id = {}

    # vocab size + 1 for padding index
    embedding_matrix = np.zeros(
        (len(vocab) + 1, embedding_dimension), dtype=np.float32
    )

    word2id[0] = "<pad>"
    for id, word in enumerate(vocab, start=1):
        word2id[word] = id
        if w2v_model.vocab.get(word, False):
            embedding_matrix[id] = w2v_model.get_vector(word)
        elif vocab[word] >= min_threshold:
            embedding_matrix[id] = np.random.uniform(
                -0.25, 0.25, embedding_dimension
            )

    return embedding_matrix, word2id


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


if __name__ == "__main__":
    w2v_file = sys.argv[1]
    data_folder = sys.argv[2]

    neg_file = f"{data_folder.rstrip('/')}/rt-polarity.neg"
    pos_file = f"{data_folder.rstrip('/')}/rt-polarity.pos"
    print("loading data...", end="")

    reviews, vocab = build_data_cv(
        neg_file, pos_file, cv=10, clean_string=True
    )
    max_length = max([r["num_words"] for r in reviews])

    print("data loaded!")
    print("number of sentences:", len(reviews))
    print("vocab size:", len(vocab))
    print("max sentence length:", max_length)

    print("Building embedding matrix...", end="", flush=True)
    embedding_matrix, word2id = build_embedding_matrix(w2v_file, vocab)
    print("Dictionary built!")

    print("num words already in word2vec:", len(word2id))
    random_matrix = np.random.random(embedding_matrix.shape)
    # Shift into range -0.25, 0.25
    random_matrix = random_matrix * 0.5 - 0.25
    # Set padding to zeros
    random_matrix[0] = np.zeros((embedding_matrix.shape[1]))

    with open("mr.p", "wb") as f:
        pickle.dump(
            [reviews, embedding_matrix, random_matrix, word2id, vocab], f
        )
    print("dataset created!")
