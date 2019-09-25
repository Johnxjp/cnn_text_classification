"""Training and evaluation script"""

import pickle
import argparse

import numpy as np
import torch

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
        "train_method",
        default="static",
        choices=["static", "non_static", "random"],
    )
    parser.add_argument("--cv_folds", type=int, default=10)
    parser.add_argument("--use_gpu", type=bool, default=False)
    return parser.parse_args()


def load_data(pickle_file):
    """
    For loading the pickle file created with process_data.py
    """
    with open(pickle_file, "rb") as f:
        contents = pickle.load(f)

    return contents


def get_id_from_sequence(
    sequence, word2id, max_sequence_length=56, pad_index=0
):
    """
    Transforms sentence into a list of indices. Pad with zeroes.

    TODO: Correct with unknown words. Kim skips over them
    """
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
    sequence_ids = np.empty((len(reviews), max_sequence_length), dtype=np.int)
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


def train_eval_loop(
    train_x,
    train_y,
    test_x,
    test_y,
    vocab_size,
    embedding_matrix,
    max_sequence_length,
    kernel_heights=[3, 4, 5],
    hidden_units=[100],
    freeze_embedding_layer=True,
    dropout=0,
    lr_decay=None,
    shuffle_batch=True,
    n_epochs=25,
    l2=0,
    batch_size=50,
    use_gpu=False,
):
    """Training and evaluation loop for a single fold"""
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
    if use_gpu:
        model = model.cuda()

    train_dataloader = create_dataloader(
        train_x, train_y, batch_size, shuffle_batch
    )

    model = train_model(
        model, train_dataloader, n_epochs, l2, lr_decay, use_gpu
    )
    class_predictions = eval_model(model, torch.LongTensor(test_x), use_gpu)
    if use_gpu:
        class_predictions = class_predictions.cpu()

    return accuracy(test_y, class_predictions.numpy())


if __name__ == "__main__":
    args = cli_parser()
    data_file = args.pickle_path
    train_method = args.train_method
    cv_folds = args.cv_folds
    use_gpu = args.use_gpu

    reviews, embedding_matrix, random_matrix, word2id, vocab = load_data(
        data_file
    )
    print("Sample review", reviews[1:5])
    print("N Reviews", len(reviews))
    print("Embedding Matrix Size", embedding_matrix.shape)
    print("Vocab Size", len(vocab))
    print("Map Size", len(word2id))

    # Parameters for model
    max_sequence_length = max([r["num_words"] for r in reviews])
    embedding_dims = embedding_matrix.shape[1]
    freeze_embedding_layer = True
    # Use word2id because it contains special tokens
    all_vocab_size = len(word2id)

    if train_method == "random":
        embedding_matrix = random_matrix

    if train_method == "non_static":
        freeze_embedding_layer = False

    performances = []
    for fold in range(cv_folds):
        train_x, train_y, test_x, test_y = make_cv_data(
            reviews, word2id, fold, max_sequence_length=max_sequence_length
        )
        print("Data Shapes")
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        print(f"Training for fold {fold + 1}")
        acc = train_eval_loop(
            train_x,
            train_y,
            test_x,
            test_y,
            all_vocab_size,
            torch.FloatTensor(embedding_matrix),
            max_sequence_length,
            lr_decay=0.95,
            kernel_heights=[3, 4, 5],
            hidden_units=[100],
            shuffle_batch=True,
            n_epochs=25,
            l2=3,
            batch_size=50,
            dropout=0.5,
            freeze_embedding_layer=freeze_embedding_layer,
            use_gpu=use_gpu,
        )
        print("Accuracy", acc)
        performances.append(acc)
        break

    print("Mean Accuracy", np.mean(performances))
    print("Std Perf", np.std(performances))
