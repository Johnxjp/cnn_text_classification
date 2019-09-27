# CNN for Text Classification

PyTorch implementation of the CNN detailed in the paper [Convolutional Neural Netowrks for Sentence Classification](http://arxiv.org/abs/1408.5882).

## Process Data

The data is available in the `./data` folder. To process run:

```
python3 process_data.py <word2vec_bin_path> ./data
```

The pre-trained Google News word2vec binary can be downloaded [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit). Alternatively you
can run `make google-news-word2vec` which will download the file to your directory. You will need `wget` installed.

## Training

```
python3 main.py <pickle_file> <mode>
```

The `<pickle_file>` is path the path to the output from the process step above. Training can be run in various
modes as specified in the paper. The three available modes are:
- `random`: Randomnly initialised embedding matrix which is updated during training
- `static`: Uses pre-trained embeddings which are fixed during training
- `non_static`: Starts with the pre-trained embeddings and fine tunes on the dataset.

Training follow the paper and uses 10-fold cross validation. The number of folds can be changed to with the 
`--cv_folds` argument. Note however, this should match the value used in the preprocessing step

Finally, if training on a gpu, set the `--use_gpu` to `true`.

```
python3 main.py <pickle_file> <mode>  --use_gpu true
```

### Other parameters
The CNN model parameters and other training parameters are specified in `main.py`.

## Other Comments
Experiments were run using the Adadelta optimiser however, unfortunately, we were unable to match the performance described in the
paper. The best results were approximately 70\%. Adam with a learning rate of 0.0001 performed marginally better.

## Requirements

- python 3.7

```
pip3 install -r requirements.txt
```

## References
- Original paper [Convolutional Neural Network for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf) and [code](https://github.com/yoonkim/CNN_sentence/blob/master/conv_net_sentence.py)
- Denny Britz [TensorFlow implementation](https://github.com/dennybritz/cnn-text-classification-tf) & [blog](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
