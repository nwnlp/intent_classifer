# -*- coding: utf-8 -*-
import tensorflow as tf
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

class MultiChnIntentCNN(object):
    def __init__(self, config, seq_length, num_classes, vocab_size):
        self.config = config
        self.input_channel1_x = tf.placeholder(tf.int32, [None, seq_length], name='input_channel1_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.input_channel2_x = tf.placeholder(tf.float32, [None, seq_length, self.config.embedding_dim], name='input_channel2_x')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_channel1_x)
        #print(embedding_inputs.shape)
        #print(self.input_channel2_x)
        intput_x = tf.stack([embedding_inputs, self.input_channel2_x], axis = 3)
        print(intput_x.shape)

        pooled_outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # filter_shape = [filter_size, self.config.embedding_dim,self.config.num_filters]
                '''W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    embedding_inputs,
                    W,
                    strides=[1, 1, 1,1],
                    padding='VALID',
                    name="conv-1"
                )'''
                print("conv-maxpool-%s" % filter_size)
                conv = tf.layers.conv2d(intput_x, self.config.num_filters, [filter_size,self.config.embedding_dim])
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


