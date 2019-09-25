"""Training and evaluation script"""

import pickle
import sys

from cnn import YKCNNClassifier


def load_data(pickle_file):
    """
    For loading the pickle file created with process_data.py
    """
    with open(pickle_file, "rb") as f:
        contents = pickle.load(f)

    return contents


if __name__ == "__main__":
    pickle_file_path = sys.argv[1]
    train_method = sys.argv[2]

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

    model = YKCNNClassifier(
        vocab_size,
        max_sequence_length,
        output_dims=2,
        embed_dim=embedding_dims,
        dropout=dropout,
        embedding_matrix=embedding_matrix,
        freeze_embedding_layer=freeze_embedding_layer,
    )

    # TODO: Split data into CV
    # TODO: Load Data
    # TODO: Train
    # TODO: Evaluate
