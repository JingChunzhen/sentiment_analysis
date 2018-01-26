import tensorflow as tf


class NBOW_Ngram(object):
    '''
    Neural Bag of Words Using Softmax 
    future work:
        implement bi-gram or tri-gram to test feature combination
    # TODO: test tf.nn.avg_pool 
    '''

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, weighted,
                 l2_reg_lambda, embedding_init, embedding_matrix, static):
        '''        
        '''
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name="x")
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="y")

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

        with tf.name_scope("mask-layer"):
            mask = tf.sign(self.input_x)
            # shape: batch_size, sequence_length
            mask = tf.expand_dims(mask, -1)
            # shape: batch_size, sequence_length, 1
            mask = tf.tile(mask, multiples=[1, 1, embedding_size])
            # shape: batch_size, sequence_length, embedding_size
            mask = tf.cast(mask, dtype=tf.float32)
            self.embedded_chars = tf.multiply(self.embedded_chars, mask)
            # shape: batch_size, sequence_length, embedding_size

        with tf.name_scope("unigram-layer"):
            w = tf.Variable(tf.truncated_normal(
                [embedding_size, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.unigram = tf.tensordot(self.embedded_chars, w, [[2], [0]])
            # self.unigram = tf.nn.softmax(unigram, dim=-1)

        with tf.name_scope("bigram-layer"):
            w = tf.Variable(tf.truncated_normal(
                [embedding_size, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            inpt = tf.expand_dims(self.embedded_chars, -1)
            # shape: batch_size, sequence_length, embedding_size, 1
            bigram = tf.nn.avg_pool(
                inpt, ksize=[1, sequence_length - 1, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
            # shape: batch_size, 2, 100, 1
            print(bigram)
            bigram = tf.squeeze(bigram)
            print(bigram)
            self.bigram = tf.tensordot(bigram, w, [[2], [0]])

        with tf.name_scope("trigram-layer"):
            w = tf.Variable(tf.truncated_normal(
                [embedding_size, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            inpt = tf.expand_dims(self.embedded_chars, -1)
            trigram = tf.nn.avg_pool(
                inpt, ksize=[1, sequence_length - 2, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
            trigram = tf.squeeze(trigram)
            self.trigram = tf.tensordot(trigram, w, [[2], [0]])

        with tf.name_scope("softmax-layer"):
            self.unigram = tf.nn.softmax(self.unigram, dim=-1)   
            self.unigram = tf.reduce_sum(self.unigram, axis=1)         
            self.bigram = tf.nn.softmax(self.bigram, dim=-1)
            self.bigram = tf.reduce_sum(self.bigram, axis=1)
            self.trigram = tf.nn.softmax(self.trigram, dim=-1)
            self.trigram = tf.reduce_sum(self.trigram, axis=1)
            self.output = self.unigram + self.bigram + self.trigram

        with tf.name_scope("predictions"):
            self.predictions = tf.argmax(
                self.output, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.output, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)  # + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")

# test

import pandas as pd
import yaml

with open('../config.yaml', 'rb') as f:
    param_all = yaml.load(f)
    params = param_all["NBOW"]
    params_global = param_all["Global"]

file_name = '../data/Amazon_Unlocked_Mobile.csv'
df = pd.read_csv(file_name)[:10]

df = df[["Reviews", "Rating"]]
df = df.dropna(axis=0, how="any")

x = df["Reviews"].tolist()
labels = df["Rating"].tolist()

y = []
for label in labels:
    if label == 1 or label == 2:
        y.append([0, 0, 1])
    if label == 3:
        y.append([0, 1, 0])
    if label == 4 or label == 5:
        y.append([1, 0, 0])

import tensorflow as tf
from tensorflow.contrib import learn

max_document_length = 163
processor = learn.preprocessing.VocabularyProcessor(max_document_length)

x = processor.fit_transform(x)

nbow = NBOW_Ngram(
    sequence_length=max_document_length,
    num_classes=params_global["num_classes"],
    vocab_size=len(processor.vocabulary_),
    embedding_size=params_global["embedding_size"],
    weighted=params["weighted"],
    l2_reg_lambda=params["l2_reg_lamda"],
    embedding_init=params["embedding_init"],                
    embedding_matrix=None,
    static=params["static"]
)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run(
        [nbow.loss],
        feed_dict={nbow.input_x: x, nbow.input_y: y})








