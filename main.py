"""Training and evaluation script"""

import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cnn import YKCNNClassifier
from utils import create_dataloader
from train import train_model
from evaluation import eval_model, accuracy


def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pickle_path", help="Path to pickle file produced by `process_data.py`"
    )
    parser.add_argument(
        "training_method",
        default="static",
        choices=["static", "non_static", "random"],
    )
    parser.add_argument("--cv_folds", type=int, default=10)

    return parser.parse_args()


def load_data(pickle_file):
    """
    For loading the pickle file created with process_data.py
    """
    with open(pickle_file, "rb") as f:
        contents = pickle.load(f)

    return contents


def train_eval_loop(
    train_x,
    train_y,
    test_x,
    test_y,
    embedding_matrix,
    freeze_embedding_layer=True,
    dropout=0.5,
    kernel_heights=[3, 4, 5],
    hidden_units=[100, 2],
    lr_decay=0.95,
    shuffle_batch=True,
    n_epochs=25,
    sqr_norm_lim=3,
    batch_size=50,
):
    model = YKCNNClassifier(
        vocab_size,
        max_sequence_length,
        output_dims=2,
        kernel_heights=kernel_heights,
        embed_dim=embedding_dims,
        fc_dropout=dropout,
        hidden_dims=hidden_units,
        embedding_matrix=embedding_matrix,
        freeze_embedding_layer=freeze_embedding_layer,
    )

    train_dataloader = create_dataloader(
        train_x, train_y, batch_size, shuffle_batch
    )

    use_gpu = False
    model = train_model(model, train_dataloader, n_epochs, lr_decay, use_gpu)
    class_predictions = eval_model(model, test_x, use_gpu)
    acc_score = accuracy(test_y, class_predictions.numpy())
    print("Accuracy", acc_score)

    return acc_score


def get_id_from_sequence(
    sequence, word2id, max_sequence_length=56, pad_index=0
):
    """Transforms sentence into a list of indices. Pad with zeroes."""
    x = np.zeros(max_sequence_length) + pad_index
    for index, word in enumerate(sequence.split()):
        x[index] = word2id[word]
    return x


def get_train_test_inds(cv, splits):
    """
    Returns training and test indices based on the split
    digit stored in the review object
    """
    id_split = np.array(splits, dtype=np.int)
    bool_mask = id_split == cv
    return np.where(~bool_mask), np.where(bool_mask)


def make_cv_data(reviews, word2id, cv, max_sequence_length=56):
    """Transforms sentences into a 2-d matrix of sequences"""
    sequence_ids = np.empty((len(reviews, max_sequence_length)), dtype=np.int)
    labels = np.empty((len(reviews)), dtype=np.int)

    for i, review in enumerate(reviews):
        sequence_ids[i] = get_id_from_sequence(
            review["text"], word2id, max_sequence_length
        )
        labels[i] = review["y"]

    train_inds, test_inds = get_train_test_inds(
        cv, [review["split"] for review in reviews]
    )
    train_x, train_y = sequence_ids[train_inds], labels[train_inds]
    test_x, test_y = sequence_ids[test_inds], labels[test_inds]

    return train_x, train_y, test_x, test_y


if __name__ == "__main__":

    reviews, embedding_matrix, random_matrix, word2id, vocab = load_data(
        pickle_file_path
    )

    # Parameters for model
    max_sequence_length = max([r["num_words"] for r in reviews])
    embedding_dims = embedding_matrix.shape[1]
    vocab_size = len(vocab)
    freeze_embedding_layer = True
    dropout = 0.5

    if train_method == "random":
        embedding_matrix = random_matrix

    if train_method == "non_static":
        freeze_embedding_layer = False

    performances = []
    for fold in range(cv_folds):
        train_x, train_y, test_x, test_y = make_cv_data(
            reviews,
            word2id,
            fold,
            max_l=max_sequence_length,
            dimension=embedding_dims,
            filter_h=5,
        )
        perf = train_eval_loop(
            train_x,
            train_y,
            embedding_matrix,
            lr_decay=0.95,
            filter_hs=[3, 4, 5],
            conv_non_linear="relu",
            hidden_units=[100, 2],
            shuffle_batch=True,
            n_epochs=25,
            sqr_norm_lim=3,
            batch_size=50,
            dropout_rate=[0.5],
            freeze_embedding_layer=freeze_embedding_layer,
        )

        performances.append(perf)

    print("Mean Accuracy", np.mean(performances))
    print("Std Perf", np.std(performances))
