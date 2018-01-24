import os
import sys
sys.path.append('..')
from itertools import chain

import numpy as np
import pyecharts
import sklearn
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn

from cnn.cnn_model import CNN
from nbow.nbow_model import NBOW
from rnn.attention import attention
from rnn.rnn_model import RNN
from utils.data_parser import batch_iter, load_data

with open("../config.yaml", 'rb') as f:
    params = yaml.load(f)


class EVAL(object):

    def __init__(self, method):
        self.method = method
        raw_x, raw_y = load_data(
            params["Global"]["task"], params["Global"]["num_classes"])
        self.max_document_length = 163

        self.processor = learn.preprocessing.VocabularyProcessor(
            self.max_document_length)
        self.processor.fit(raw_x)

        raw_x = list(self.processor.transform(raw_x))

        x, y = [], []
        for tmp_x, tmp_y in zip(raw_x, raw_y):
            tmp_x = tmp_x.tolist()
            if np.sum(tmp_x) != 0:
                x.append(tmp_x)
                y.append(tmp_y)

        x_temp, self.x_test, y_temp, self.y_test = train_test_split(
            x, y, test_size=params["Global"]["test_size"])
        self.x_train, self.x_validate, self.y_train, self.y_validate = train_test_split(
            x_temp, y_temp, test_size=params["Global"]["validate_size"])

        self.embedding_matrix = self._embedding_matrix_initializer(
        ) if params[method]["embedding_init"] else None
        self.model = self._model_config_initializer()
        del x_temp, y_temp, raw_x, x, y

    def _embedding_matrix_initializer(self):
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

    def _model_config_initializer(self):
        vocab_size = len(self.processor.vocabulary_)
        embedding_matrix = list(chain.from_iterable(self.embedding_matrix))
        filter_sizes = list(map(int, params["CNN"]["filter_sizes"].split(',')))
        model = {
            "NBOW": {
                "sequence_length": self.max_document_length,
                "num_classes": params["Global"]["num_classes"],
                "vocab_size": vocab_size,
                "embedding_size": params["Global"]["embedding_size"],
                "weighted": params["NBOW"]["weighted"],
                "l2_reg_lambda": params["NBOW"]["l2_reg_lamda"],
                "embedding_init": params["NBOW"]["embedding_init"],
                "embedding_matrix": embedding_matrix,
                "static": params[method]["static"]
            },
            "CNN": {
                "num_classes": params["Global"]["num_classes"],
                "num_filters": params["CNN"]["num_filters"],
                "filter_sizes": filter_sizes,
                "embedding_size": params["Global"]["embedding_size"],
                "vocab_size": vocab_size,
                "sequence_length": self.max_document_length,
                "l2_reg_lambda": params["CNN"]["l2_reg_lambda"],
                "embedding_init": params["CNN"]["embedding_init"],
                "embedding_matrix": embedding_matrix,
                "static": params["CNN"]["static"]
            },
            "RNN": {
                "sequence_length": self.max_document_length,
                "num_classes": params["Global"]["num_classes"],
                "embedding_size": params["Global"]["embedding_size"],
                "vocab_size": vocab_size,
                "hidden_size": params["RNN"]["hidden_size"],
                "num_layers": params["RNN"]["num_layers"],
                "l2_reg_lambda": params["RNN"]["l2_reg_lambda"],
                "dynamic": params["RNN"]["dynamic"],
                "use_attention": params["RNN"]["use_attention"],
                "attention_size": params["RNN"]["attention_size"],
                "embedding_init": params["RNN"]["embedding_init"],
                "embedding_matrix": embedding_matrix,
                "static": params["RNN"]["static"]
            }
        }
        return model

    def process(self, learning_rate, batch_size, epochs, evaluate_every):

        with tf.Graph().as_default():

            instance = globals()[self.method](**self.model[self.method])

            global_step = tf.Variable(0, trainable=False)
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(
                instance.loss, global_step=global_step)

            init = tf.global_variables_initializer()
            with tf.Session() as sess:

                def train_step(x_batch, y_batch):
                    feed_dict = {
                        instance.input_x: x_batch,
                        instance.input_y: y_batch
                    }
                    if self.method == "NBOW":
                        pass
                    if self.method == "CNN":
                        feed_dict[instance.dropout_keep_prob] = params[self.method]["dropout_keep_prob"]
                    if self.method == "RNN":
                        feed_dict[instance.input_keep_prob] = params[self.method]["input_keep_prob"]
                        feed_dict[instance.output_keep_prob] = params[self.method]["output_keep_prob"]

                    _, step, accuracy_, loss_ = sess.run(
                        [train_op, global_step, instance.accuracy, instance.loss], feed_dict=feed_dict)
                    return step, accuracy_, loss_

                def dev_step(x_batch, y_batch):
                    feed_dict = {
                        instance.input_x: x_batch,
                        instance.input_y: y_batch
                    }
                    if self.method == "NBOW":
                        pass
                    if self.method == "CNN":
                        feed_dict[instance.dropout_keep_prob] = 1.0
                    if self.method == "RNN":
                        feed_dict[instance.input_keep_prob] = 1.0
                        feed_dict[instance.output_keep_prob] = 1.0

                    pred_, accuracy_, loss_ = sess.run(
                        [instance.predictions, instance.accuracy, instance.loss], feed_dict=feed_dict)
                    return pred_, accuracy_, loss_

                sess.run(init)

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
                            pred_, accuracy_, loss_ = dev_step(x_dev, y_dev)
                            accuracies.append(accuracy_)
                            losses.append(loss_)

                            y_pred.extend(pred_.tolist())
                            y_true.extend(np.argmax(y_dev, axis=1).tolist())

                        print("Evaluation Accuracy: {}, Loss: {}".format(
                            np.mean(accuracies), np.mean(losses)))
                        print(classification_report(
                            y_true=y_true, y_pred=y_pred))


if __name__ == "__main__":
    eval = EVAL("CNN")
    eval.process(
        learning_rate=1e-3,
        batch_size=128,
        epochs=100,
        evaluate_every=1000
    )
