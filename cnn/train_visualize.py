import os
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import pyecharts
import sklearn
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn

from cnn_sentiment import CNN
from utils.data_parser import batch_iter, load_data
from itertools import chain


with open('../config.yaml', 'rb') as f:
    params = yaml.load(f)



class EVAL(object):

    def __init__(self):
        '''
        tested 
        get the map of word to ids 
        get the split of train, dev and test data and labels               
        '''
        raw_x, y = load_data(params["Global"]["task"], params["Global"]["num_classes"])
        # self.max_document_length = max(
        #     [len(text.split(' ')) for text in raw_x])
        self.max_document_length = 163
        self.processor = learn.preprocessing.VocabularyProcessor(
            self.max_document_length)
        x = list(self.processor.fit_transform(raw_x))

        x_temp, self.x_test, y_temp, self.y_test = train_test_split(
            x, y, test_size=params["Global"]["test_size"])
        self.x_train, self.x_validate, self.y_train, self.y_validate = train_test_split(
            x_temp, y_temp, test_size=params["Global"]["validate_size"])

        if params["CNN"]["embedding_init"]:
            self.embedding_matrix = self._embedding_matrix_initializer()
        # free
        del x_temp, y_temp, raw_x, x, y

    def _embedding_matrix_initializer(self):
        '''
        initialize the embedding_layer using GloVe
        Return:
            embedding_matrix (matrix with float): shape (vocabulary_size, embedding_size)
        '''
        file_wv = "../data/glove.6B/glove.6B.{}d.txt".format(
            params["Global"]["embedding_size"])
        wv = {}
        embedding_matrix = []

        with open(file_wv, 'r') as f:
            for line in f:
                line = line.split(' ')
                word = line[0]
                wv[word] = list(map(float, line[1:]))

        for idx in range(len(self.processor.vocabulary_)):
            word = self.processor.vocabulary_.reverse(idx)
            embedding_matrix.append(
                wv[word] if word in wv else np.random.normal(size=params["Global"]["embedding_size"]))        
        return embedding_matrix

    def process(self, learning_rate, batch_size, epochs, evaluate_every):

        with tf.Graph().as_default():

            self.instance = CNN(
                num_classes=params["Global"]["num_classes"],
                num_filters=params["CNN"]["num_filters"],
                filter_sizes=list(map(int, params["CNN"]["filter_sizes"].split(','))),
                embedding_size=params["Global"]["embedding_size"],
                vocab_size=len(self.processor.vocabulary_),
                sequence_length=self.max_document_length,
                l2_reg_lambda=params["CNN"]["l2_reg_lambda"],
                embedding_init= params["CNN"]["embedding_init"],              
                #embedding_matrix=list(chain.from_iterable(self.embedding_matrix)),
                embedding_matrix=None,
                static=params["CNN"]["static"]
            )            

            global_step = tf.Variable(0, trainable=False)
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(
                self.instance.loss, global_step=global_step)

            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            tf.summary.scalar("loss", self.instance.loss)
            tf.summary.scalar("accuracy", self.instance.accuracy)
            merged_summary_op = tf.summary.merge_all()
            with tf.Session() as sess:

                def train_step(x_batch, y_batch):
                    feed_dict = {
                        self.instance.input_x: x_batch,
                        self.instance.input_y: y_batch,
                        self.instance.dropout_keep_prob: params["CNN"]["dropout_keep_prob"]
                    }
                    _, step, accuracy_, loss_ = sess.run(
                        [train_op, global_step, self.instance.accuracy, self.instance.loss], feed_dict=feed_dict)
                    return step, accuracy_, loss_

                def dev_step(x_batch, y_batch):
                    feed_dict = {
                        self.instance.input_x: x_batch,
                        self.instance.input_y: y_batch,
                        self.instance.dropout_keep_prob: 1.0
                    }
                    pred_, accuracy_, loss_ = sess.run(
                        [self.instance.predictions, self.instance.accuracy, self.instance.loss], feed_dict=feed_dict)
                    return pred_, accuracy_, loss_

                sess.run(init)

                train_summary_writer = tf.summary.FileWriter(
                    logdir='../temp/summary/CNN_sentiment/train', graph=sess.graph)
                dev_summary_writer = tf.summary.FileWriter(
                    logdir='../temp/summary/CNN_sentiment/dev', graph=sess.graph)

                for batch in batch_iter(list(zip(self.x_train, self.y_train)), batch_size, epochs):
                    x_batch, y_batch = zip(*batch)
                    current_step, accuracy_, loss_ = train_step(
                        x_batch, y_batch)
                    print("Training, step: {}, accuracy: {:.2f}, loss: {:.5f}".format(
                        current_step, accuracy_, loss_))
                    current_step = tf.train.global_step(sess, global_step)
                    
                    if current_step % evaluate_every == 0:
                        print("\nEvaluation:")

                        losses = []
                        accuracies = []

                        y_true = []
                        y_pred = []

                        for batch in batch_iter(list(zip(self.x_validate, self.y_validate)), 50, 1):
                            x_dev, y_dev = zip(*batch)
                            pred_, accuracy_, loss_ = dev_step(x_dev, y_dev) # not enough value to unpack 
                            accuracies.append(accuracy_)
                            losses.append(loss_)

                            y_pred.extend(pred_.tolist())
                            y_true.extend(np.argmax(y_dev, axis=1).tolist())

                        print("Evaluation Accuracy: {}, Loss: {}".format(
                            np.mean(accuracies), np.mean(losses)))
                        print(classification_report(
                            y_true=y_true, y_pred=y_pred))

                    if current_step % 400 == 0:
                        saver.save(sess, "../temp/model/CNN_sentiment/CNN_sentiment",
                                   global_step=current_step)
                    if current_step % 500 == 0:
                        data = list(self._generate(sess))
                        if params["Global"]["num_classes"] == 3:
                            df = pd.DataFrame(
                                data, columns=["word", "positive", "neutral", "negative"])
                        else:
                            df = pd.DataFrame(
                                data, columns=["word", "positive", "negative"])
                        df.to_csv(
                            '../temp/visualization/CNN_sentiment_{}.csv'.format(current_step))
                    if current_step % evaluate_every == 0:
                        print("\nEvaluation:")
                        losses = []
                        accuracies = []
                        for batch in batch_iter(list(zip(self.x_validate, self.y_validate)), 50, 1):
                            x_dev, y_dev = zip(*batch)
                            pred_, accuracy_, loss_ = dev_step(x_dev, y_dev)
                            accuracies.append(accuracy_)
                            losses.append(loss_)
                        print("Evaluation Accuracy: {}, Loss: {}".format(
                            np.mean(accuracies), np.mean(losses)))

    def _generate(self, sess):
        '''
        visualize the result every 2000 steps
        '''
        for batch in batch_iter(list(zip(self.x_validate, self.y_validate)), 50, 1):
            x_batch, y_batch = zip(*batch)
            feed_dict = {
                self.instance.input_x: x_batch,
                self.instance.input_y: y_batch
            }
            dist, seq_len = sess.run(
                [self.instance.dist, self.instance.seq_len], feed_dict=feed_dict)
            for i in range(np.shape(x_batch)[0]):
                temp = []
                text = x_batch[i]
                for j in range(seq_len[i]):
                    word = self.processor.vocabulary_.reverse(text[j])
                    temp = [word]
                    temp.extend(dist[i][j])
                    yield temp


def visualization(file_in):
    from pyecharts import Bar
    df = pd.read_csv(file_in)
    words = df["word"].tolist()
    positive = df["positive"].tolist()
    if params["Global"]["num_classes"] == 3:
        neutral = df["neutral"].tolist()
    negative = df["negative"].tolist()
    bar = Bar("")
    bar.add("positive", words, positive, is_stack=True,
            xaxis_interval=0, label_color=["#4682B4"])
    if params["Global"]["num_classes"] == 3:
        bar.add("neutral", words, neutral, is_stack=True,
                xaxis_interval=0, label_color=["#E3E3E3"])
    bar.add("negative", words, negative, is_stack=True,
            xaxis_interval=0, label_color=["#CD3333"])
    bar.render("../temp/visualization/bar_negative.html")


if __name__ == "__main__":
    eval = EVAL()
    eval.process(
        learning_rate=1e-3,
        batch_size=128,
        epochs=100,
        evaluate_every=50
    )
