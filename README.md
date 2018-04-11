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
  --data_dir DATA_DIR  Directory containing corpus

```

Train:

```bash
python3.6 cnn_cls.py --Train True --kernel stem_cnn
using static word embedding which learned by word2vec as the input of cnn

python3.6 cnn_cls.py --Train True --kernel nonstem_cnn
using non static word embedding which learned by model itself as the input of cnn

python3.6 cnn_cls.py --Train True --kernel multiem_cnn
using non static word embedding and static word embedding as the input of cnn

python3.6 cnn_cls.py --Train True --kernel svm
using linear svm

```

## Evaluating

```bash
python3.6 cnn_cls.py --Train False --kernel stem_cnn

python3.6 cnn_cls.py --Train False --kernel nonstem_cnn

python3.6 cnn_cls.py --Train False --kernel multiem_cnn

python3.6 cnn_cls.py --Train False --kernel svm

```

Defalut corpus dictory is atis, you can use --data_dir to use your own data


## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
