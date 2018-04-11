# -*- coding: utf-8 -*-
import os
import shutil
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
import time
from gensim.models import KeyedVectors
class CNNConfig(object):
    """CNN配置参数"""
    embedding_dim = 300      # 词向量维度
    num_filters = 128        # 卷积核数目
    filter_sizes = [2,3,4]         # 卷积核尺寸
    hidden_dim = 32        # 全连接层神经元
    dropout_keep_prob = 0.5 # dropout保留比例
    REGULARIZATION_RATE = 1.0
    learning_rate = 1e-3    # 学习率
    num_epochs = 20         # 总迭代轮次
    print_per_batch = 10    # 每多少轮输出一次结果
    save_tb_per_batch = 10
    batch_size = 64
class BaseIntentCNN(object):
    __metaclass__ = ABCMeta

    def init(self, config, pooled_outputs, num_classes):

        self.config = config
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        num_filters_total = self.config.num_filters * len(self.config.filter_sizes)  # 500*4
        pooled_reshape = tf.reshape(tf.concat(pooled_outputs, 1), [-1, num_filters_total])
        pooled_flat = tf.nn.dropout(pooled_reshape, self.config.dropout_keep_prob)

        with tf.name_scope("score"):
            weight = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1))
            biases = tf.Variable(tf.constant(0.1, shape=[num_classes]))
            self.logits = tf.nn.relu(tf.matmul(pooled_flat, weight) + biases)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            regularizer = tf.contrib.layers.l2_regularizer(self.config.REGULARIZATION_RATE)
            regularization = regularizer(weight)
            self.loss = tf.reduce_mean(cross_entropy) + regularization

            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            self.correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())
        self.checkpoint_path = os.path.join('cnn_model', 'model.ckpt')
        tensorboard_train_dir = 'tensorboard/train'
        tensorboard_valid_dir = 'tensorboard/valid'

        if os.path.exists(tensorboard_train_dir):
            shutil.rmtree(tensorboard_train_dir)
        os.makedirs(tensorboard_train_dir)
        if os.path.exists(tensorboard_valid_dir):
            shutil.rmtree(tensorboard_valid_dir)
        os.makedirs(tensorboard_valid_dir)
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.acc)
        self.merged_summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(tensorboard_train_dir)
        self.valid_writer = tf.summary.FileWriter(tensorboard_valid_dir)



class NonStaticChnIntentCNN(BaseIntentCNN):

    def __init__(self,config,seq_length,num_classes,vocab_size):
        self.input_channel1_x = tf.placeholder(tf.int32, [None, seq_length], name='input_channel1_x')
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [vocab_size, config.embedding_dim])
            self.intput_x = tf.nn.embedding_lookup(embedding, self.input_channel1_x)
        pooled_outputs = []
        for i, filter_size in enumerate(config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                print("conv-maxpool-%s" % filter_size)
                conv = tf.layers.conv1d(self.intput_x, config.num_filters, filter_size)
                pooled = tf.reduce_max(conv, reduction_indices=[1])
                pooled_outputs.append(pooled)
        BaseIntentCNN.init(self,config, pooled_outputs, num_classes)

    def non_static_batch_iter(self,X, Y):
        data_len = len(X)
        num_batch = int((data_len - 1) / self.config.batch_size) + 1

        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = X[indices]
        y_shuffle = Y[indices]

        for i in range(num_batch):
            start_id = i * self.config.batch_size
            end_id = min((i + 1) * self.config.batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def fit(self,x_train,y_train,x_dev,y_dev):
        dev_feed_dict = {
            self.input_channel1_x: x_dev,
            self.input_y: y_dev,
        }
        for epoch in range(self.config.num_epochs):
            print('Epoch:', epoch + 1)
            batch_train = self.non_static_batch_iter(x_train,y_train)
            for x_batch, y_batch in batch_train:
                train_feed_dict = {
                    self.input_channel1_x: x_batch,
                    self.input_y: y_batch,
                }
                self.session.run(self.optim, feed_dict=train_feed_dict)
            train_loss, train_accuracy = self.session.run([self.loss, self.acc], feed_dict=train_feed_dict)
            valid_loss, valid_accuracy, valid_pred, pred_cls = self.session.run(
                    [self.loss, self.acc, self.correct_pred, self.y_pred_cls], feed_dict=dev_feed_dict)
            print(
                'train_loss:' + str(train_loss) + ' train accuracy:' + str(
                    train_accuracy) + '\tvalid_loss:' + str(
                    valid_loss) + ' valid accuracy:' + str(valid_accuracy))
            train_s = self.session.run(self.merged_summary, feed_dict=train_feed_dict)
            self.train_writer.add_summary(train_s, epoch)
            valid_s = self.session.run(self.merged_summary, feed_dict=dev_feed_dict)
            self.valid_writer.add_summary(valid_s, epoch)
            self.saver.save(self.session, self.checkpoint_path, global_step=epoch)
    def test(self, x_test, y_test):
        ckpt = tf.train.get_checkpoint_state('cnn_model')
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            print("...no checkpoint found...")
            return

        test_feed_dict = {
            self.input_channel1_x: x_test,
            self.input_y: y_test,
        }
        test_loss, test_accuracy = self.session.run([self.loss, self.acc],
                                               feed_dict=test_feed_dict)
        print('test_loss:' + str(test_loss) + ' test accuracy:' + str(test_accuracy))

class StaticWordEmbedding():
    def __init__(self, word2id):
        start = time.time()
        model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300-SLIM.bin.gz', binary=True)
        print('Finished loading slim model %.1f sec' % ((time.time() - start)))
        print('word2vec: %d' % len(model.vocab))
        self.word2vec = {}
        for word, id in word2id.items():
            if word in model.vocab:
                self.word2vec[id] = model.wv[word]
            else:
                self.word2vec[id] = np.random.normal(0, 0.1, 300)

    def embedding(self,x_data):
        result = []
        for x in x_data:
            vecs = []
            for word_id in x:
                vec = self.word2vec[word_id]
                vecs.append(vec)
            result.append(np.array(vecs))
        return np.array(result)


class StaticChnIntentCNN(BaseIntentCNN):
    def __init__(self,config,seq_length,num_classes):
        self.input_x = tf.placeholder(tf.float32, [None, seq_length, config.embedding_dim], name='input_x')
        pooled_outputs = []
        for i, filter_size in enumerate(config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                print("conv-maxpool-%s" % filter_size)
                conv = tf.layers.conv1d(self.input_x, config.num_filters, filter_size)
                pooled = tf.reduce_max(conv, reduction_indices=[1])
                pooled_outputs.append(pooled)
        BaseIntentCNN.init(self,config, pooled_outputs, num_classes)

    def static_batch_iter(self,X, Y):
        data_len = len(X)
        num_batch = int((data_len - 1) / self.config.batch_size) + 1

        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = X[indices]
        y_shuffle = Y[indices]

        for i in range(num_batch):
            start_id = i * self.config.batch_size
            end_id = min((i + 1) * self.config.batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def fit(self,x_train,y_train,x_dev,y_dev,word2id):
        word_embedding = StaticWordEmbedding(word2id)
        x_train = word_embedding.embedding(x_train)
        x_dev = word_embedding.embedding(x_dev)
        dev_feed_dict = {
            self.input_x: x_dev,
            self.input_y: y_dev,
        }
        for epoch in range(self.config.num_epochs):
            print('Epoch:', epoch + 1)
            batch_train = self.static_batch_iter(x_train,y_train)
            for x_batch, y_batch in batch_train:
                train_feed_dict = {
                    self.input_x: x_batch,
                    self.input_y: y_batch,
                }
                self.session.run(self.optim, feed_dict=train_feed_dict)
            train_loss, train_accuracy = self.session.run([self.loss, self.acc], feed_dict=train_feed_dict)
            valid_loss, valid_accuracy, valid_pred, pred_cls = self.session.run(
                    [self.loss, self.acc, self.correct_pred, self.y_pred_cls], feed_dict=dev_feed_dict)
            print(
                'train_loss:' + str(train_loss) + ' train accuracy:' + str(
                    train_accuracy) + '\tvalid_loss:' + str(
                    valid_loss) + ' valid accuracy:' + str(valid_accuracy))
            train_s = self.session.run(self.merged_summary, feed_dict=train_feed_dict)
            self.train_writer.add_summary(train_s, epoch)
            valid_s = self.session.run(self.merged_summary, feed_dict=dev_feed_dict)
            self.valid_writer.add_summary(valid_s, epoch)
            self.saver.save(self.session, self.checkpoint_path, global_step=epoch)

    def test(self, x_test, y_test, word2id):
        ckpt = tf.train.get_checkpoint_state('cnn_model')
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            print("...no checkpoint found...")
            return

        word_embedding = StaticWordEmbedding(word2id)
        x_test= word_embedding.embedding(x_test)
        test_feed_dict = {
            self.input_x: x_test,
            self.input_y: y_test,
        }
        test_loss, test_accuracy = self.session.run([self.loss, self.acc],
                                               feed_dict=test_feed_dict)
        print('test_loss:' + str(test_loss) + ' test accuracy:' + str(test_accuracy))


class MultiChnIntentCNN(object):
    def __init__(self, config, seq_length, num_classes, vocab_size):
        self.input_channel1_x = tf.placeholder(tf.int32, [None, seq_length], name='input_channel1_x')
        self.input_channel2_x = tf.placeholder(tf.float32, [None, seq_length, config.embedding_dim], name='input_channel2_x')
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [vocab_size, config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_channel1_x)

        intput_x = tf.stack([embedding_inputs, self.input_channel2_x], axis=3)

        pooled_outputs = []
        for i, filter_size in enumerate(config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                print("conv-maxpool-%s" % filter_size)
                conv = tf.layers.conv2d(intput_x, config.num_filters, [filter_size,config.embedding_dim])
                pooled = tf.reduce_max(conv, reduction_indices=[1])
                pooled_outputs.append(pooled)
        BaseIntentCNN.init(self, config, pooled_outputs, num_classes)

    def multi_batch_iter(self, X, Y, X_embedding):
        data_len = len(X)
        num_batch = int((data_len - 1) / self.config.batch_size) + 1

        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = X[indices]
        y_shuffle = Y[indices]
        x_embedding_chn = X_embedding[indices]

        for i in range(num_batch):
            start_id = i * self.config.batch_size
            end_id = min((i + 1) * self.config.batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id], x_embedding_chn[start_id:end_id]

    def fit(self,x_train,y_train,x_dev,y_dev,word2id):
        word_embedding = StaticWordEmbedding(word2id)
        x_train_embedding = word_embedding.embedding(x_train)
        x_dev_embedding = word_embedding.embedding(x_dev)
        dev_feed_dict = {
            self.input_channel1_x: x_dev,
            self.input_channel2_x:x_dev_embedding,
            self.input_y: y_dev,
        }
        for epoch in range(self.config.num_epochs):
            print('Epoch:', epoch + 1)
            batch_train = self.multi_batch_iter(x_train,y_train,x_train_embedding)
            for x_batch, y_batch, x_embedding_batch in batch_train:
                train_feed_dict = {
                    self.input_channel1_x: x_batch,
                    self.input_channel2_x:x_embedding_batch,
                    self.input_y: y_batch,
                }
                self.session.run(self.optim, feed_dict=train_feed_dict)
            train_loss, train_accuracy = self.session.run([self.loss, self.acc], feed_dict=train_feed_dict)
            valid_loss, valid_accuracy, valid_pred, pred_cls = self.session.run(
                    [self.loss, self.acc, self.correct_pred, self.y_pred_cls], feed_dict=dev_feed_dict)
            print(
                'train_loss:' + str(train_loss) + ' train accuracy:' + str(
                    train_accuracy) + '\tvalid_loss:' + str(
                    valid_loss) + ' valid accuracy:' + str(valid_accuracy))
            train_s = self.session.run(self.merged_summary, feed_dict=train_feed_dict)
            self.train_writer.add_summary(train_s, epoch)
            valid_s = self.session.run(self.merged_summary, feed_dict=dev_feed_dict)
            self.valid_writer.add_summary(valid_s, epoch)
            self.saver.save(self.session, self.checkpoint_path, global_step=epoch)
    def test(self, x_test, y_test, word2id):
        ckpt = tf.train.get_checkpoint_state('cnn_model')
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            print("...no checkpoint found...")
            return

        word_embedding = StaticWordEmbedding(word2id)
        x_test_embedding = word_embedding.embedding(x_test)
        test_feed_dict = {
            self.input_channel1_x: x_test,
            self.input_channel2_x: x_test_embedding,
            self.input_y: y_test,
        }

        test_loss, test_accuracy = self.session.run([self.loss, self.acc],
                                               feed_dict=test_feed_dict)
        print('test_loss:' + str(test_loss) + ' test accuracy:' + str(test_accuracy))

class IntentCNN(object):
    def __init__(self, config, seq_length, num_classes, vocab_size):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, [None, seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        pooled_outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                #filter_shape = [filter_size, self.config.embedding_dim,self.config.num_filters]
                '''W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    embedding_inputs,
                    W,
                    strides=[1, 1, 1,1],
                    padding='VALID',
                    name="conv-1"
                )'''
                print ("conv-maxpool-%s" % filter_size)
                conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, filter_size)
                pooled = tf.reduce_max(conv, reduction_indices=[1])
                pooled_outputs.append(pooled)
        num_filters_total = self.config.num_filters * len(self.config.filter_sizes)  # 500*4
        pooled_reshape = tf.reshape(tf.concat(pooled_outputs, 1), [-1, num_filters_total])
        pooled_flat = tf.nn.dropout(pooled_reshape, self.keep_prob)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            #fc = tf.layers.dense(pooled_flat, self.config.hidden_dim, name='fc1')
            '''weight1 = tf.Variable(tf.truncated_normal([num_filters_total, self.config.hidden_dim], stddev=0.1))
            biases1 = tf.Variable(tf.constant(0.1, shape=[self.config.hidden_dim]))
            fc = tf.nn.relu(tf.matmul(pooled_flat, weight1) + biases1)
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)'''
            # 分类器

            weight2 = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1))
            biases2 = tf.Variable(tf.constant(0.1, shape=[num_classes]))
            self.logits = tf.nn.relu(tf.matmul(pooled_flat, weight2) + biases2)
            #self.logits = tf.layers.dense(fc, num_classes,name='fc2')

            #self.logits = tf.contrib.layers.dropout(self.logits, self.keep_prob)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            regularizer = tf.contrib.layers.l2_regularizer(self.config.REGULARIZATION_RATE)
            regularization = regularizer(weight2)
            self.loss = tf.reduce_mean(cross_entropy)# + regularization

            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            self.correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


