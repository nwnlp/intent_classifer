import os
import numpy as np
import json
import cnn_model as model
import tensorflow as tf
import time
from gensim.models import KeyedVectors
import numpy as np
import shutil
import gzip
import shutil
from sklearn.svm import SVC

tf.flags.DEFINE_integer("embedding_mode", 0, "0:non-static embedding,1:static embedding,2:two-embedding")
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

def vectorize_data(word2id, intent2id, raw_data, vec_size):
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
            if len(tmp) > vec_size:
                tmp = tmp[:vec_size]
                x.append(tmp)
            elif len(tmp) <= vec_size:
                tmp.extend([word2id['_PAD_']]*(vec_size-len(tmp)))
                x.append(np.array(tmp))
            y.append(dense_to_one_hot(intent_id, len(intent2id)))
            text.append(sent)
    return np.array(x), np.array(y), text

def static_channel(word2vec, x_data):
    result = []
    for x in x_data:
        vecs = []
        for word_id in x:
            vec = word2vec[word_id]
            vecs.append(vec)
        result.append(np.array(vecs))
    return np.array(result)

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

'''def batch_iter(X, Y, x_train_st_chn, batch_size=128):
    data_len = len(X)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = X[indices]
    y_shuffle = Y[indices]
    x_st_chn = x_train_st_chn[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id], x_st_chn[start_id:end_id]'''

def static_batch_iter(X, Y, batch_size=128):
    data_len = len(X)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = X[indices]
    y_shuffle = Y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

'''def feed_data(cnn, x_batch, y_batch, x_st_chn, keep_prob):
    feed_dict = {
        cnn.input_channel1_x: x_batch,
        cnn.input_y: y_batch,
        cnn.input_channel2_x: x_st_chn,
        cnn.keep_prob: keep_prob
    }
    return feed_dict'''

def feed_data_non_static(cnn, x_batch, y_batch):
    feed_dict = {
        cnn.input_channel1_x: x_batch,
        cnn.input_y: y_batch,
    }
    return feed_dict

def feed_data_static(cnn, x_static_batch, y_batch, keep_prob):
    feed_dict = {
        cnn.input_channel2_x: x_static_batch,
        cnn.input_y: y_batch,
        cnn.keep_prob: keep_prob
    }
    return feed_dict

def feed_data_two_chns(cnn, x_batch, x_static_batch, y_batch, keep_prob):
    feed_dict = {
        cnn.input_channel1_x: x_batch,
        cnn.input_channel2_x: x_static_batch,
        cnn.input_y: y_batch,
        cnn.keep_prob: keep_prob
    }
    return feed_dict

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

def load_word2vec(word2id):
    start = time.time()
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300-SLIM.bin.gz', binary=True)
    print('Finished loading slim model %.1f sec' % ((time.time() - start)))
    print('word2vec: %d' % len(model.vocab))
    word2vec = {}
    for word,id in word2id.items():
        if word in model.vocab:
            word2vec[id] = model.wv[word]
        else:
            word2vec[id] = np.random.normal(0,0.1,300)
    return word2vec
def svm_data_format_x(X):
    x_svm = []
    for x_sent in X:
        x_sent = np.dot(np.ones((1, x_sent.shape[0])), x_sent)
        x_svm.append(x_sent[0])
    return x_svm

def svm_data_format_y(Y):
    y_svm = []
    for y in Y:
        y_svm.append(y.argmax())
    return y_svm


def _train(argv=None):

    word2id, intent2id, id2intent,raw_train_data, vec_size, sents_size = load_train_data(os.path.join(FLAGS.data_dir, 'train'))
    #word2vec = load_word2vec(word2id)
    x_train, y_train, text_train = vectorize_data(word2id, intent2id, raw_train_data, vec_size)
    #x_train_st_chn = static_channel(word2vec, x_train)

    raw_dev_data = load_dev_test_data(os.path.join(FLAGS.data_dir, 'dev'))
    x_dev, y_dev, text_dev = vectorize_data(word2id, intent2id, raw_dev_data, vec_size)
    #x_dev_st_chn = static_channel(word2vec, x_dev)

    raw_test_data = load_dev_test_data(os.path.join(FLAGS.data_dir, 'test'))
    x_test, y_test, text_test = vectorize_data(word2id, intent2id, raw_test_data, vec_size)
    #x_test_st_chn = static_channel(word2vec, x_test)

    '''svm'''

    #clf = SVC(C = 0.1, kernel='linear')
    #clf.fit(svm_data_format_x(x_train_st_chn), svm_data_format_y(y_train))
    #svm_pred_test_y = clf.predict(svm_data_format_x(x_test_st_chn))

    #svm_acc = np.mean(np.equal(svm_pred_test_y, svm_data_format_y(y_test)).astype(float))
    #print(svm_acc)


    print('vocab size:'+str(len(word2id)))
    #dbug_data(x_train, y_train, id2intent, 'train.data')
    #dbug_data(x_dev, y_dev, id2intent, 'dev.data')
    #dbug_data(x_test, y_test, id2intent, 'test.data')
    config = model.CNNConfig
    if FLAGS.embedding_mode == 0:
        cnn = model.NonStaticIntentCNN(config, vec_size, len(intent2id), len(word2id))
    elif FLAGS.embedding_mode == 1:
        cnn = model.StaticIntentCNN(config, vec_size, len(intent2id))
        word2vec = load_word2vec(word2id)
        x_train_st_chn = static_channel(word2vec, x_train)
        x_dev_st_chn = static_channel(word2vec, x_dev)



    else:
        pass

    total_batch = 0
    if FLAGS.embedding_mode == 0:
        dev_feed_dict = feed_data_non_static(cnn, x_dev,y_dev)
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train)
        for x_batch, y_batch in batch_train:
            total_batch += 1
            if FLAGS.embedding_mode == 0:
                train_feed_dict = feed_data_non_static(cnn, x_batch, y_batch)
            elif FLAGS.embedding_mode == 1:
                train_feed_dict = feed_data_static(cnn)
            session.run(cnn.optim, feed_dict=train_feed_dict)
            if total_batch % config.print_per_batch == 0:
                train_loss, train_accuracy = session.run([cnn.loss, cnn.acc], feed_dict=train_feed_dict)

                valid_loss, valid_accuracy, valid_pred, pred_cls = session.run([cnn.loss, cnn.acc, cnn.correct_pred, cnn.y_pred_cls], feed_dict=dev_feed_dict)
                #print('Steps:' + str(total_batch))
                print(
                    'train_loss:' + str(train_loss) + ' train accuracy:' + str(train_accuracy) + '\tvalid_loss:' + str(
                        valid_loss) + ' valid accuracy:' + str(valid_accuracy))
                incorrect = {}
                all_incorrect_cnt = 0
                logtmp = []
                for i, pred in enumerate(valid_pred):
                    if pred == False:
                        index = y_dev[i].argmax()
                        intent = id2intent[index]
                        incorrect.setdefault(intent, 0)
                        incorrect[intent] += 1
                        all_incorrect_cnt += 1
                        text = text_dev[i]
                        pred_intent = id2intent[pred_cls[i]]
                        logtmp.append(' '.join(text) +' '+intent +'->'+pred_intent)

                for intent in incorrect:
                    incorrect[intent] = incorrect[intent]/all_incorrect_cnt
                logstr = 'valid incorrect percentage:'
                for intent, per in incorrect.items():
                    logstr += intent +' '+str(per)+'\t'
                #print(logstr)
                #print('\n'.join(logtmp))
            if total_batch % config.save_tb_per_batch == 0:
                train_s = session.run(merged_summary, feed_dict=train_feed_dict)
                train_writer.add_summary(train_s, total_batch)
                valid_s = session.run(merged_summary, feed_dict=dev_feed_dict)
                valid_writer.add_summary(valid_s, total_batch)

        saver.save(session, checkpoint_path, global_step=epoch)
    if FLAGS.embedding_mode == 0:
        test_feed_dict = feed_data_non_static(cnn, x_test,y_test,config.dropout_keep_prob)

    test_loss, test_accuracy = session.run([cnn.loss, cnn.acc],
                                           feed_dict=test_feed_dict)
    print('test_loss:' + str(test_loss) + ' test accuracy:' + str(test_accuracy))
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


def train():
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
    cnn = model.NonStaticIntentCNN(config, vec_size, len(intent2id), len(word2id))
    cnn.fit(x_train,y_train,x_dev,y_dev)

def test():
    word2id, intent2id, id2intent, vec_size = restore_corpus_data()
    raw_test_data = load_dev_test_data(os.path.join(FLAGS.data_dir, 'test'))
    x_test, y_test, text_test = vectorize_data(word2id, intent2id, raw_test_data, vec_size)
    config = model.CNNConfig
    cnn = model.NonStaticIntentCNN(config, vec_size, len(intent2id), len(word2id))
    cnn.test(x_test,y_test)
    # x_test_st_chn = static_channel(word2vec, x_test)

if __name__ == '__main__':
    if FLAGS.train:
        print('start traning '+FLAGS.data_dir+'...')
        train()
    else:
        print('start testing '+FLAGS.data_dir+'...')
        test()