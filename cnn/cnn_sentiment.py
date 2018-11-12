import numpy as np
import tensorflow as tf
import yaml

with open('../config.yaml', 'rb') as f:
    param_all = yaml.load(f)
    params = param_all["CNN"]
    params_global = param_all["Global"]


class CNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    Original taken from https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py
    """

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes,
                 num_filters, l2_reg_lambda, embedding_matrix, static, embedding_init):
        '''
        Args:            
            embedding_init (boolean): True for initialize the embedding layer with glove false for not
            embedding_matrix (list of float): length vocabulary size * embedding_size
            static (boolean): False for embedding_layer trainable during training false True for not
        '''
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name="x")
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="y")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        if embedding_init:
            with tf.name_scope("embedding-layer-with-glove-initialized"):
                self.W = tf.get_variable(shape=[vocab_size, embedding_size], initializer=tf.constant_initializer(
                    embedding_matrix), name='W', trainable=not static)
                self.embedded_chars = tf.nn.embedding_lookup(
                    self.W, self.input_x)
        else:
            with tf.name_scope("embedding-layer"):
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.nn.embedding_lookup(
                    self.W, self.input_x)

        self.embedded_chars_expanded = tf.expand_dims(
            self.embedded_chars, -1)

        pooled_outputs = []
        self.convs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-{}".format(filter_size)):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(
                    0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # self.convs.append(conv)  # B, S - h + 1, 1, num_filters
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                self.convs.append(h) # B, S - h + 1, 1, num_filters 

                # pooled = tf.nn.max_pool(
                pooled = tf.nn.avg_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(
                self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            self.W_fc = tf.get_variable(
                "fc_w",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            self.b_fc = tf.Variable(tf.constant(
                0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(self.W_fc)
            l2_loss += tf.nn.l2_loss(self.b_fc)
            self.scores = tf.nn.xw_plus_b(
                self.h_drop, self.W_fc, self.b_fc, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.name_scope("get-word-sentiment"):
            '''learning word sentiment for each word in a sequence in each batch 
            '''
            mask = tf.sign(self.input_x)
            range_ = tf.range(
                start=1, limit=sequence_length + 1, dtype=tf.int32)
            mask = tf.multiply(mask, range_, name="mask")
            self.seq_len = tf.reduce_max(mask, axis=1)  # B

            temps = []
            for i in range(len(filter_sizes)):
                x = self.convs[i]
                batch_size = tf.shape(x)[0]
                temp = tf.squeeze(x, axis=2)  # B, S - h + 1, num_filters, 2
                temp_w = tf.slice(self.W_fc, begin=[
                                  num_filters * i, 0], size=[num_filters, num_classes])
                # B, S - h + 1, 2
                temp = tf.tensordot(temp, temp_w, [[2], [0]])
                temp = tf.expand_dims(temp, axis=2)
                pad = tf.zeros(
                    [batch_size, filter_sizes[i] - 1, 1, num_classes])
                temp = tf.concat([pad, temp], axis=1)
                temp = tf.concat([temp, pad], axis=1)
                temp = tf.nn.avg_pool(temp, ksize=[1, filter_sizes[i], 1, 1], strides=[
                    1, 1, 1, 1], padding='VALID')  # B, S, 1, 2
                temp = tf.squeeze(temp, axis=2)
                temps.append(temp)
            temp = tf.stack(temps, axis=1)
            temp = tf.reduce_mean(temp, axis=1)
            self.dist = tf.nn.softmax(temp, axis=2)  # B, S, 2


