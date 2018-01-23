import sys
sys.path.append('..')

import numpy as np
import pyecharts
import sklearn
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn

from cnn_model import CNN
from itertools import chain
from utils.data_parser import batch_iter, load_data


with open('../config.yaml', 'rb') as f:
    param_all = yaml.load(f)
    params = param_all["CNN"]
    params_global = param_all["Global"]


class EVAL(object):

    def __init__(self):
        '''
        tested 
        get the map of word to ids 
        get the split of train, dev and test data and labels        
        # TODO: 效果可视化 tensorboard
        # TODO: 模型保存和加载 
        '''
        raw_x, y = load_data(params_global["task"], params_global["num_classes"])
        # self.max_document_length = max(
        #     [len(text.split(' ')) for text in raw_x])
        self.max_document_length = 163
        self.processor = learn.preprocessing.VocabularyProcessor(
            self.max_document_length)
        x = list(self.processor.fit_transform(raw_x))

        x_temp, self.x_test, y_temp, self.y_test = train_test_split(
            x, y, test_size=params_global["test_size"])
        self.x_train, self.x_validate, self.y_train, self.y_validate = train_test_split(
            x_temp, y_temp, test_size=params_global["validate_size"])

        if params["embedding_init"]:
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
            params_global["embedding_size"])
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
                wv[word] if word in wv else np.random.normal(size=params_global["embedding_size"]))        
        return embedding_matrix

    def process(self, learning_rate, batch_size, epochs, evaluate_every):

        with tf.Graph().as_default():

            cnn = CNN(
                num_classes=params_global["num_classes"],
                num_filters=params["num_filters"],
                filter_sizes=list(map(int, params["filter_sizes"].split(','))),
                embedding_size=params_global["embedding_size"],
                vocab_size=len(self.processor.vocabulary_),
                sequence_length=self.max_document_length,
                l2_reg_lambda=params["l2_reg_lambda"],
                embedding_init=params["embedding_init"],                
                embedding_matrix=list(chain.from_iterable(self.embedding_matrix)),
                static=params["static"]
            )

            global_step = tf.Variable(0, trainable=False)
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(
                cnn.loss, global_step=global_step)

            init = tf.global_variables_initializer()
            with tf.Session() as sess:

                def train_step(x_batch, y_batch):
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: params["dropout_keep_prob"]
                    }
                    _, step, accuracy_, loss_ = sess.run(
                        [train_op, global_step, cnn.accuracy, cnn.loss], feed_dict=feed_dict)
                    return step, accuracy_, loss_

                def dev_step(x_batch, y_batch):
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: 1.0
                    }
                    accuracy_, loss_ = sess.run(
                        [cnn.accuracy, cnn.loss], feed_dict=feed_dict)
                    return accuracy_, loss_

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
                        for batch in batch_iter(list(zip(self.x_validate, self.y_validate)), 50, 1):
                            x_dev, y_dev = zip(*batch)
                            accuracy_, loss_ = dev_step(x_dev, y_dev)
                            accuracies.append(accuracy_)
                            losses.append(loss_)
                        print("Evaluation Accuracy: {}, Loss: {}".format(
                            np.mean(accuracies), np.mean(losses)))


if __name__ == "__main__":
    eval = EVAL()
    eval.process(
        learning_rate=1e-3,
        batch_size=128,
        epochs=100,
        evaluate_every=1000
    )
