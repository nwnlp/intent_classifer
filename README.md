This is slightly simplified implementation of sentence intent classifer.
Mainly refer to [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper
## Requirements

- Python 3.6
- Tensorflow 1.4.0
- Numpy

## Training

Print parameters:

```bash
python3.6 cnn_cls.py --help
```

```
optional arguments:
  -h, --help           show this help message and exit
  --kernel KERNEL      stem_cnn:static-embedding cnn, nonstem_cnn:non static-
                       embedding cnn, multiem_cnn:static and non static
                       embedding cnn, svm:linear svm
  --train [TRAIN]      if True, begin to train
  --notrain
  --data_dir DATA_DIR  Directory containing corpus

```

Train:

```bash
python3.6 cnn_cls.py --Train True --kernel stem_cnn
using static wordembedding which learned by word2vec

```

## Evaluating

```bash
./eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/"
```

Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.


## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)
