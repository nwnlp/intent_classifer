import os
import numpy as np
import cnn_model as model
import tensorflow as tf
import time
from gensim.models import KeyedVectors
import numpy as np
import gzip
import shutil

def load_train_data(dir = 'atis/train'):
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
            sent = line.strip().split(' ')
            for word in sent:
                if word not in word2id:
                    word2id[word] = len(word2id)
            max_sent_size = max(max_sent_size, len(sent))
            sents_size.append(len(sent))
            data.setdefault(file, [])
            data[file].append(sent)
        intent2id[file] = len(intent2id)
    return word2id, intent2id, data,max_sent_size,sents_size

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
            sent = line.strip().split(' ')
            data.setdefault(file, [])
            data[file].append(sent)
    return data

def batch_iter(X, Y, x_train_st_chn, batch_size=128):
    data_len = len(X)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = X[indices]
    y_shuffle = Y[indices]
    x_st_chn = x_train_st_chn[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id], x_st_chn[start_id:end_id]


def feed_data(cnn, x_batch, y_batch, x_st_chn, keep_prob):
    feed_dict = {
        cnn.input_channel1_x: x_batch,
        cnn.input_y: y_batch,
        cnn.input_channel2_x: x_st_chn,
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

def train(argv=None):

    word2id, intent2id, raw_train_data, vec_size, sents_size = load_train_data()
    word2vec = load_word2vec(word2id)

    '''sent_size_cnt = [0]*(vec_size+1)
    for sent_size in sents_size:
        sent_size_cnt[sent_size] += 1
    print(sent_size_cnt)'''
    vec_size = 20
    id2intent = {}
    for intent, id in intent2id.items():
        id2intent[id] = intent
    x_train, y_train, text_train = vectorize_data(word2id, intent2id, raw_train_data, vec_size)
    x_train_st_chn = static_channel(word2vec, x_train)

    raw_dev_data = load_dev_test_data('atis/dev')
    x_dev, y_dev, text_dev = vectorize_data(word2id, intent2id, raw_dev_data, vec_size)
    x_dev_st_chn = static_channel(word2vec, x_dev)

    raw_test_data = load_dev_test_data('atis/test')
    x_test, y_test, text_test = vectorize_data(word2id, intent2id, raw_test_data, vec_size)
    x_test_st_chn = static_channel(word2vec, x_test)


    print('vocab size:'+str(len(word2id)))
    dbug_data(x_train, y_train, id2intent, 'train.data')
    dbug_data(x_dev, y_dev, id2intent, 'dev.data')
    dbug_data(x_test, y_test, id2intent, 'test.data')
    config = model.CNNConfig
    #cnn = model.IntentCNN(config, vec_size, len(intent2id), len(word2id))
    cnn = model.MultiChnIntentCNN(config, vec_size, len(intent2id), len(word2id))
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    checkpoint_path = os.path.join('cnn_model', 'model.ckpt')
    tensorboard_train_dir = 'tensorboard/train'
    tensorboard_valid_dir = 'tensorboard/valid'

    if not os.path.exists(tensorboard_train_dir):
        os.makedirs(tensorboard_train_dir)
    if not os.path.exists(tensorboard_valid_dir):
        os.makedirs(tensorboard_valid_dir)
    tf.summary.scalar("loss", cnn.loss)
    tf.summary.scalar("accuracy", cnn.acc)
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    valid_writer = tf.summary.FileWriter(tensorboard_valid_dir)

    total_batch = 0
    dev_feed_dict = feed_data(cnn, x_dev,y_dev,x_dev_st_chn,config.dropout_keep_prob)
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train,x_train_st_chn)
        for x_batch, y_batch, x_st_chn in batch_train:
            total_batch += 1
            train_feed_dict = feed_data(cnn, x_batch, y_batch, x_st_chn, config.dropout_keep_prob)
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
    test_feed_dict = feed_data(cnn, x_test,y_test,x_test_st_chn,config.dropout_keep_prob)

    test_loss, test_accuracy = session.run([cnn.loss, cnn.acc],
                                           feed_dict=test_feed_dict)
    print('test_loss:' + str(test_loss) + ' test accuracy:' + str(test_accuracy))

if __name__ == '__main__':
    train()