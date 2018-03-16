import os
import re
import sys
sys.path.append('..')
from itertools import chain

import numpy as np
import pandas as pd
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


with open("../config.yaml", "r") as f:
    params = yaml.load(f)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    this is very crude
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data(task, train_or_test):
    file_in = "../data/mtl-dataset/{}.task.{}".format(task, train_or_test)
    reviews = []
    polarities = []
    with open(file_in, "r", encoding='ISO-8859-1') as f:
        for line in f.readlines():
            line = clean_str(line)
            polarities.append([1, 0] if int(line[0]) == 0 else [0, 1])
            review = line[1:].strip()
            reviews.append(review)
    return reviews, polarities


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    need to be tested 
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)  
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


class EVAL(object):

    def __init__(self, method):
        self.method = method
        # task = params["Global"]["task"]
        raw_x, raw_y = load_data("apparel", "train")

        self.max_document_length = 256

        self.processor = learn.preprocessing.VocabularyProcessor(
            self.max_document_length)
        self.processor.fit(raw_x)

        raw_x = list(self.processor.transform(raw_x))

        x, y = [], []
        # following code can be optimized
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

    def test_proessor(self):
        batch_size = 10        
        for batch in batch_iter(list(zip(self.x_test, self.y_test)), batch_size, num_epochs=1):
            x_batch, y_batch = zip(*batch)
            texts = list(self.processor.reverse(x_batch))
            print(np.shape(texts))
            # print(texts)
            break        
        for text in texts:
            words = text.split(" ")
            print(words)
            print(len(words))
            break

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
        embedding_matrix = list(chain.from_iterable(
            self.embedding_matrix)) if self.embedding_matrix else None
        filter_sizes = list(map(int, params["CNN"]["filter_sizes"].split(',')))
        model = {
            "NBOW": {
                "sequence_length": self.max_document_length,
                "num_classes": 2,
                "vocab_size": vocab_size,
                "embedding_size": params["Global"]["embedding_size"],
                "weighted": params["NBOW"]["weighted"],
                "l2_reg_lambda": params["NBOW"]["l2_reg_lamda"],
                "embedding_init": params["NBOW"]["embedding_init"],
                "embedding_matrix": embedding_matrix,
                "static": params["NBOW"]["static"]
            },
            "CNN": {
                "num_classes": 2,
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
                "num_classes": 2,
                "embedding_size": params["Global"]["embedding_size"],
                "vocab_size": vocab_size,
                "hidden_size": params["RNN"]["hidden_size"],
                "num_layers": params["RNN"]["num_layers"],
                "l2_reg_lambda": params["RNN"]["l2_reg_lambda"],
                "dynamic": False,
                "use_attention": True,
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

            # tf.summary.scalar("loss", instance.loss)
            # tf.summary.scalar("accuracy", instance.accuracy)
            # merged_summary_op = tf.summary.merge_all()

            with tf.Session() as sess:

                sess.run(init)
                # train_summary_writer = tf.summary.FileWriter(
                #     logdir='../temp/summary/{}/train'.format(self.method), graph=sess.graph)
                # dev_summary_writer = tf.summary.FileWriter(
                #     logdir='../temp/summary/{}/dev'.format(self.method), graph=sess.graph)

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
                        feed_dict[instance.input_keep_prob] = 0.5
                        feed_dict[instance.output_keep_prob] = 0.5

                    _, step, accuracy_, loss_ = sess.run(
                        [train_op, global_step,
                            instance.accuracy, instance.loss],
                        feed_dict=feed_dict)
                    # train_summary_writer.add_summary(summary, step)

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

                    pred_, step, accuracy_, loss_, alpha_ = sess.run(
                        [instance.predictions,
                            global_step, instance.accuracy, instance.loss, instance.alpha],
                        feed_dict=feed_dict)
                    # dev_summary_writer.add_summary(summary, step)
                    return pred_, accuracy_, loss_, alpha_

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

                        vis_data = []
                        for batch in batch_iter(list(zip(self.x_validate, self.y_validate)), 50, 1):
                            x_dev, y_dev = zip(*batch)
                            pred_, accuracy_, loss_, alpha_ = dev_step(x_dev, y_dev)

                            texts = list(self.processor.reverse(x_dev))
                            # batch_size, 1
                            for text, weights in zip(texts, alpha_):  
                                # sequence_length, 
                                text_list = text.split(" ")                                                              
                                for word, weight in zip(text_list, weights):
                                    vis_data.append([word, weight])
                                vis_data.append([" ", " "])
                            accuracies.append(accuracy_)
                            losses.append(loss_)

                            y_pred.extend(pred_.tolist())
                            y_true.extend(np.argmax(y_dev, axis=1).tolist())
                            
                        print("Evaluation Accuracy: {}, Loss: {}".format(
                            np.mean(accuracies), np.mean(losses)))
                        print(classification_report(
                            y_true=y_true, y_pred=y_pred))
                        df = pd.DataFrame(data=vis_data, columns=["text", "weight"])
                        df.to_csv("{}.csv".format(current_step))


if __name__ == "__main__":
    eval = EVAL("RNN")
    # eval.test_proessor()
    
    eval.process(
        learning_rate=1e-3,
        batch_size=100,
        epochs=100,
        evaluate_every=100
    )
