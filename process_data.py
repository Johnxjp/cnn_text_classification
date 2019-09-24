"""Adapted from yk github"""

import numpy as np
from copy import deepcopy
import pickle
from collections import defaultdict
import sys, re
import pandas as pd

def build_data_cv(neg_file, pos_file, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    vocab = defaultdict(float)
    for positivity, file in enumerate([neg_file, pos_file]):
        with open(file, 'r') as f:
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
                datum  = {
                    "y": positivity, 
                    "text": orig_rev,                       
                    "num_words": len(words),
                    "split": np.random.randint(0, cv)
                }
                revs.append(datum)
    return revs, vocab
    
def get_W(word_vecs, dimensions=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = {}
    W = np.zeros(shape=(vocab_size + 1, dimensions), dtype='float32')            
    W[0] = np.zeros(dimensions, dtype='float32')
    for i, word in enumerate(word_vecs, start=1):
        W[i] = word_vecs[word]
        word_idx_map[word] = i
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(
                   f.read(binary_len), dtype='float32'
                )  
            else:
                f.read(binary_len)
    return word_vecs

def get_unknown_words(word_vecs, vocab, min_df=1, dimensions=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    unknown_words = {}
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            unknown_words[word] = np.random.uniform(-0.25, 0.25, dimensions)
    return unknown_words

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    print(string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    print(string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

if __name__=="__main__":    
    w2v_file = sys.argv[1]
    data_folder = sys.argv[2]
    
    neg_file = f"{data_folder.rstrip('/')}/rt-polarity.neg"
    pos_file = f"{data_folder.rstrip('/')}/rt-polarity.pos"
    print("loading data...", end="")
    
    reviews, vocab = build_data_cv(neg_file, pos_file, cv=10, clean_string=True)
    max_length = max([r["num_words"] for r in reviews])
    
    print("data loaded!")
    print("number of sentences:", len(reviews))
    print("vocab size:", len(vocab))
    print("max sentence length:", max_length)
    print("loading word2vec vectors...", end="")
    w2v = load_bin_vec(w2v_file, vocab)
    print("word2vec loaded!")
    print("num words already in word2vec:", len(w2v))
    unknown_words = get_unknown_words(w2v, vocab)
    w2v.update(unknown_words)
    W, word_idx_map = get_W(w2v)
    rand_vecs = get_unknown_words({}, vocab)  # This is weird!!
    W2, _ = get_W(rand_vecs)
    pickle.dump([reviews, W, W2, word_idx_map, vocab], open("mr.p", "wb"))
    print("dataset created!")
    
