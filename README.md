# CNN for Sentence Classification

PyTorch implementation of the CNN detailed in the paper [Convolutional Neural Netowrks for Sentence Classification](http://arxiv.org/abs/1408.5882).

## Process Data

The data is available in the `./data` folder. To process run:

```
python process_data.py <word2vec_bin_path> ./data
```

The pre-trained Google News word2vec binary can be downloaded [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit).

## Requirements

- python 3.7
- torch 1.2.0
- gensim 3.8.0

```
pip3 install -r requirements.txt
```
