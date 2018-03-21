import os
import numpy as np
import cnn_model as model
import tensorflow as tf
def load_train_data(dir = 'TREC/train'):
    files = os.listdir(dir)
    word2id = {}
    word2id['_PAD_'] = len(word2id)
    word2id['_UNKNOW_'] = len(word2id)
    intent2id = {}
    data = {}
    max_sent_size = 0
    for file in files:
        for line in open(os.path.join(dir, file),'r').readlines():
            sent = line.strip().split(' ')
            for word in sent:
                if word not in word2id:
                    word2id[word] = len(word2id)
            max_sent_size = max(max_sent_size, len(sent))
            data.setdefault(file, [])
            data[file].append(sent)
        intent2id[file] = len(intent2id)
    return word2id, intent2id, data,max_sent_size

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    return np.eye(num_classes)[labels_dense]

def vectorize_data(word2id, intent2id, raw_data, vec_size):
    y = []
    x = []
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
                tmp = tmp[vec_size:]
                x.append(tmp)
            elif len(tmp) <= vec_size:
                tmp.extend([word2id['_PAD_']]*(vec_size-len(tmp)))
                x.append(np.array(tmp))
            y.append(dense_to_one_hot(intent_id, len(intent2id)))
    return np.array(x), np.array(y)

def load_dev_test_data(dir):
    data = {}
    files = os.listdir(dir)
    for file in files:
        for line in open(os.path.join(dir, file),'r').readlines():
            sent = line.strip().split(' ')
            data.setdefault(file, [])
            data[file].append(sent)
    return data

def batch_iter(X, Y, batch_size=128):
    data_len = len(X)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = X[indices]
    y_shuffle = Y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def feed_data(cnn, x_batch, y_batch, keep_prob):
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.keep_prob: keep_prob
    }
    return feed_dict


def mean_normalize(features):
    std_value = features.std()
    mean_value = features.mean()
    return (features - mean_value) / std_value

def train(argv=None):

    word2id, intent2id, raw_train_data, vec_size = load_train_data()
    x_train, y_train = vectorize_data(word2id, intent2id, raw_train_data, vec_size)
    raw_dev_data = load_dev_test_data('TREC/dev')
    x_dev, y_dev = vectorize_data(word2id, intent2id, raw_dev_data, vec_size)
    raw_test_data = load_dev_test_data('TREC/test')
    x_test, y_test = vectorize_data(word2id, intent2id, raw_test_data, vec_size)
    x_train = mean_normalize(x_train)
    x_dev = mean_normalize(x_dev)
    x_test = mean_normalize(x_test)
    print('vocab size:'+str(len(word2id)))
    config = model.CNNConfig
    cnn = model.IntentCNN(config, vec_size, len(intent2id), len(word2id))
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
    for epoch in range(config.num_epochs):
        # print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train)
        for x_batch, y_batch in batch_train:
            total_batch += 1
            feed_dict = feed_data(cnn, x_batch, y_batch, config.dropout_keep_prob)
            session.run(cnn.optim, feed_dict=feed_dict)
            if total_batch % config.print_per_batch == 0:
                train_loss, train_accuracy = session.run([cnn.loss, cnn.acc], feed_dict=feed_dict)
                valid_loss, valid_accuracy = session.run([cnn.loss, cnn.acc], feed_dict={cnn.input_x: x_dev,
                                                                                         cnn.input_y: y_dev,
                                                                                         cnn.keep_prob: config.dropout_keep_prob})
                print('Steps:' + str(total_batch))
                print(
                    'train_loss:' + str(train_loss) + ' train accuracy:' + str(train_accuracy) + '\tvalid_loss:' + str(
                        valid_loss) + ' valid accuracy:' + str(valid_accuracy))
            if total_batch % config.save_tb_per_batch == 0:
                train_s = session.run(merged_summary, feed_dict=feed_dict)
                train_writer.add_summary(train_s, total_batch)
                valid_s = session.run(merged_summary, feed_dict={cnn.input_x: x_dev, cnn.input_y: y_dev,
                                                                 cnn.keep_prob: config.dropout_keep_prob})
                valid_writer.add_summary(valid_s, total_batch)

        saver.save(session, checkpoint_path, global_step=epoch)
    test_loss, test_accuracy = session.run([cnn.loss, cnn.acc],
                                           feed_dict={cnn.input_x: x_test, cnn.input_y: y_test,
                                                      cnn.keep_prob: config.dropout_keep_prob})
    print('test_loss:' + str(test_loss) + ' test accuracy:' + str(test_accuracy))

if __name__ == '__main__':
    train()