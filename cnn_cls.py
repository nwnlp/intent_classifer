import os
import json
import cnn_model as model
import tensorflow as tf
from sklearn.svm import SVC
import numpy as np
from sklearn.externals import joblib
tf.flags.DEFINE_string("kernel", "svm", "stem_cnn:static-embedding cnn,\
                                     nonstem_cnn:non static-embedding cnn,\
                                    multiem_cnn:static and non static embedding cnn,\
                                     svm:linear svm")
tf.flags.DEFINE_boolean('train', False, 'if True, begin to train')
tf.flags.DEFINE_string("data_dir", "atis", "Directory containing corpus")
FLAGS = tf.flags.FLAGS

def load_train_data(dir):
    files = os.listdir(dir)
    word2id = {}
    word2id['_PAD_'] = len(word2id)
    word2id['_UNKNOW_'] = len(word2id)
    intent2id = {}
    data = {}
    max_sent_size = 0
    sents_size = []
    for file in files:
        for line in open(os.path.join(dir, file),'r').readlines():
            sent = line.strip().split(' ') #for english
            #sent = line.strip() #for chinese
            for word in sent:
                if word not in word2id:
                    word2id[word] = len(word2id)
            max_sent_size = max(max_sent_size, len(sent))
            sents_size.append(len(sent))
            data.setdefault(file, [])
            data[file].append(sent)
        intent2id[file] = len(intent2id)
    id2intent = {}
    for intent, id in intent2id.items():
        id2intent[id] = intent
    return word2id, intent2id, id2intent,data,max_sent_size,sents_size

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    return np.eye(num_classes)[labels_dense]

def vectorize_data(word2id, intent2id, raw_data, vec_size, same_dim = True):
    y = []
    x = []
    text = []
    for intent, sent_list in raw_data.items():
        if intent not in intent2id:
            continue
        intent_id = intent2id[intent]
        for sent in sent_list:
            tmp = []
            for word in sent:
                if word in word2id:
                    tmp.append(word2id[word])
                else:
                    tmp.append(word2id['_UNKNOW_'])
            if same_dim:
                if len(tmp) > vec_size:
                    tmp = tmp[:vec_size]
                    x.append(tmp)
                elif len(tmp) <= vec_size:
                    tmp.extend([word2id['_PAD_']]*(vec_size-len(tmp)))
            x.append(np.array(tmp))
            y.append(dense_to_one_hot(intent_id, len(intent2id)))
            text.append(sent)
    return np.array(x), np.array(y), text



def load_dev_test_data(dir):
    data = {}
    files = os.listdir(dir)
    for file in files:
        for line in open(os.path.join(dir, file),'r').readlines():
            sent = line.strip().split(' ') #for english
            #sent = line.strip()  # for chinese
            data.setdefault(file, [])
            data[file].append(sent)
    return data

def mean_normalize(features):
    std_value = features.std()
    mean_value = features.mean()
    return (features - mean_value) / std_value

def dbug_data(X, Y, id2intent, file):
    x_str = []
    for i,x in enumerate(X):
        y = Y[i]
        intent = id2intent[y.argmax()]
        x = [str(ele) for ele in x]
        x_str.append(intent+':'+' '.join(x))
    result = '\n'.join(x_str)
    open(file, 'w').write(result)

def save_corpus_data(word2id, intent2id, id2intent, vec_size):
    dictObj = {}
    dictObj['word2id'] = word2id
    dictObj['intent2id'] = intent2id
    dictObj['id2intent'] = id2intent
    dictObj['vec_size'] = vec_size
    jsObj = json.dumps(dictObj)
    fileObject = open('corpus-data.json', 'w')
    fileObject.write(jsObj)
    fileObject.close()

def restore_corpus_data():
    fileObject = open('corpus-data.json', 'r')
    dictObj = json.load(fileObject)
    return dictObj['word2id'],dictObj['intent2id'],dictObj['id2intent'],dictObj['vec_size']


def non_static_train():
    word2id, intent2id, id2intent, raw_train_data, vec_size, sents_size = load_train_data(
        os.path.join(FLAGS.data_dir, 'train'))
    save_corpus_data(word2id, intent2id, id2intent, vec_size)
    # word2vec = load_word2vec(word2id)
    x_train, y_train, text_train = vectorize_data(word2id, intent2id, raw_train_data, vec_size)
    # x_train_st_chn = static_channel(word2vec, x_train)

    raw_dev_data = load_dev_test_data(os.path.join(FLAGS.data_dir, 'dev'))
    x_dev, y_dev, text_dev = vectorize_data(word2id, intent2id, raw_dev_data, vec_size)
    # x_dev_st_chn = static_channel(word2vec, x_dev)

    raw_test_data = load_dev_test_data(os.path.join(FLAGS.data_dir, 'test'))
    x_test, y_test, text_test = vectorize_data(word2id, intent2id, raw_test_data, vec_size)
    # x_test_st_chn = static_channel(word2vec, x_test)

    config = model.CNNConfig
    cnn = model.NonStaticChnIntentCNN(config, vec_size, len(intent2id), len(word2id))
    cnn.fit(x_train,y_train,x_dev,y_dev)

def non_static_test():
    word2id, intent2id, id2intent, vec_size = restore_corpus_data()
    raw_test_data = load_dev_test_data(os.path.join(FLAGS.data_dir, 'test'))
    x_test, y_test, text_test = vectorize_data(word2id, intent2id, raw_test_data, vec_size)
    config = model.CNNConfig
    cnn = model.NonStaticChnIntentCNN(config, vec_size, len(intent2id), len(word2id))
    cnn.test(x_test,y_test)
    # x_test_st_chn = static_channel(word2vec, x_test)

def static_train():
    word2id, intent2id, id2intent, raw_train_data, vec_size, sents_size = load_train_data(
        os.path.join(FLAGS.data_dir, 'train'))
    save_corpus_data(word2id, intent2id, id2intent, vec_size)
    x_train, y_train, text_train = vectorize_data(word2id, intent2id, raw_train_data, vec_size)

    raw_dev_data = load_dev_test_data(os.path.join(FLAGS.data_dir, 'dev'))
    x_dev, y_dev, text_dev = vectorize_data(word2id, intent2id, raw_dev_data, vec_size)

    config = model.CNNConfig
    cnn = model.StaticChnIntentCNN(config, vec_size, len(intent2id))
    cnn.fit(x_train, y_train, x_dev, y_dev, word2id)

def static_test():
    word2id, intent2id, id2intent, vec_size = restore_corpus_data()
    raw_test_data = load_dev_test_data(os.path.join(FLAGS.data_dir, 'test'))
    x_test, y_test, text_test = vectorize_data(word2id, intent2id, raw_test_data, vec_size)
    config = model.CNNConfig
    cnn = model.StaticChnIntentCNN(config, vec_size, len(intent2id))
    cnn.test(x_test,y_test, word2id)
    # x_test_st_chn = static_channel(word2vec, x_test)

def multi_train():
    word2id, intent2id, id2intent, raw_train_data, vec_size, sents_size = load_train_data(
        os.path.join(FLAGS.data_dir, 'train'))
    save_corpus_data(word2id, intent2id, id2intent, vec_size)
    x_train, y_train, text_train = vectorize_data(word2id, intent2id, raw_train_data, vec_size)

    raw_dev_data = load_dev_test_data(os.path.join(FLAGS.data_dir, 'dev'))
    x_dev, y_dev, text_dev = vectorize_data(word2id, intent2id, raw_dev_data, vec_size)

    config = model.CNNConfig
    cnn = model.MultiChnIntentCNN(config, vec_size, len(intent2id),len(word2id))
    cnn.fit(x_train, y_train, x_dev, y_dev, word2id)

def multi_test():
    word2id, intent2id, id2intent, vec_size = restore_corpus_data()
    raw_test_data = load_dev_test_data(os.path.join(FLAGS.data_dir, 'test'))
    x_test, y_test, text_test = vectorize_data(word2id, intent2id, raw_test_data, vec_size)
    config = model.CNNConfig
    cnn = model.MultiChnIntentCNN(config, vec_size, len(intent2id),len(word2id))
    cnn.test(x_test, y_test, word2id)

def svm_data_format_x(X):
    x_svm = []
    for x_sent in X:
        sum = [0.0]*x_sent.shape[1]
        for word in x_sent:
            sum += word
        #x_sent = np.dot(np.ones((1, x_sent.shape[0])), x_sent)
        x_svm.append(mean_normalize(sum/ x_sent.shape[1]))
    return np.array(x_svm)

def svm_data_format_y(Y):
    y_svm = []
    for y in Y:
        y_svm.append(y.argmax())
    return np.array(y_svm)

def svm_train():
    word2id, intent2id, id2intent, raw_train_data, vec_size, sents_size = load_train_data(
        os.path.join(FLAGS.data_dir, 'train'))
    save_corpus_data(word2id, intent2id, id2intent, vec_size)
    x_train, y_train, text_train = vectorize_data(word2id, intent2id, raw_train_data, vec_size, False)
    word_embedding = model.StaticWordEmbedding(word2id)
    x_train_embedding = word_embedding.embedding(x_train)
    clf = SVC(C=10, kernel='linear',decision_function_shape='ovo')
    x_train_svm = svm_data_format_x(x_train_embedding)
    y_train_svm = svm_data_format_y(y_train)
    clf.fit(x_train_svm,y_train_svm)
    joblib.dump(clf, 'svm.pkl')
    train_score = clf.score(x_train_svm,y_train_svm)
    print(train_score)

def svm_test():
    clf = joblib.load('svm.pkl')
    word2id, intent2id, id2intent, vec_size = restore_corpus_data()
    raw_test_data = load_dev_test_data(os.path.join(FLAGS.data_dir, 'test'))
    x_test, y_test, text_test = vectorize_data(word2id, intent2id, raw_test_data, vec_size,False)
    word_embedding = model.StaticWordEmbedding(word2id)
    x_test_embedding = word_embedding.embedding(x_test)
    x_test_svm = svm_data_format_x(x_test_embedding)
    y_test_svm = svm_data_format_y(y_test)
    test_score = clf.score(x_test_svm,y_test_svm)
    print(test_score)
if __name__ == '__main__':
    if FLAGS.train:
        print('start traning '+FLAGS.data_dir+' using kernel:'+FLAGS.kernel+'...')
        if FLAGS.kernel == "nonstem_cnn":
            non_static_train()
        elif FLAGS.kernel == "stem_cnn":
            static_train()
        elif FLAGS.kernel == "multiem_cnn":
            multi_train()
        elif FLAGS.kernel == "svm":
            svm_train()

    else:
        print('start testing '+FLAGS.data_dir+' using kernel:'+FLAGS.kernel+'...')
        if FLAGS.kernel == "nonstem_cnn":
            non_static_test()
        elif FLAGS.kernel == "stem_cnn":
            static_test()
        elif FLAGS.kernel == "multiem_cnn":
            multi_test()
        elif FLAGS.kernel == "svm":
            svm_test()