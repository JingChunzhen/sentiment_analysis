import sys
sys.path.append('..')

import numpy as np
import pyecharts
import sklearn
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.contrib import learn

from rnn_model import RNN
from itertools import chain
from utils.data_parser import batch_iter, load_data


with open('../config.yaml', 'rb') as f:
    param_all = yaml.load(f)
    params = param_all["RNN"]
    params_global = param_all["Global"]


class EVAL(object):

    def __init__(self):
        '''
        tested 
        get the map of word to ids 
        get the split of train, dev and test data and labels                
        '''
        raw_x, y = load_data(params_global["task"], params_global["num_classes"])
        self.max_document_length = 163
        # optimum max_document_length coming from ../test/statistics
        # processor.restore(file_path)
        self.processor = learn.preprocessing.VocabularyProcessor(
            self.max_document_length)
        x = list(self.processor.fit_transform(raw_x))
        x = [l.tolist() for l in x]
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

            rnn = RNN(
                sequence_length=self.max_document_length,
                num_classes=params_global["num_classes"],
                embedding_size=params_global["embedding_size"],
                vocab_size=len(self.processor.vocabulary_),
                hidden_size=params["hidden_size"],
                num_layers=params["num_layers"],
                l2_reg_lambda=params["l2_reg_lambda"],
                dynamic=params["dynamic"],
                use_attention=params["use_attention"],
                attention_size=params["attention_size"],
                embedding_init=params["embedding_init"],                
                embedding_matrix=list(chain.from_iterable(self.embedding_matrix)),
                static=params["static"]
            )

            global_step = tf.Variable(0, trainable=False)
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(
                rnn.loss, global_step=global_step)

            init = tf.global_variables_initializer()
            with tf.Session() as sess:

                def train_step(x_batch, y_batch):
                    feed_dict = {
                        rnn.input_x: x_batch,
                        rnn.input_y: y_batch,
                        rnn.input_keep_prob: params["input_keep_prob"],
                        rnn.output_keep_prob: params["output_keep_prob"]
                    }
                    _, step, accuracy_, loss_ = sess.run(
                        [train_op, global_step, rnn.accuracy, rnn.loss], feed_dict=feed_dict)
                    return step, accuracy_, loss_

                def dev_step(x_batch, y_batch):
                    feed_dict = {
                        rnn.input_x: x_batch,
                        rnn.input_y: y_batch,
                        rnn.input_keep_prob: 1.0,
                        rnn.output_keep_prob: 1.0
                    }
                    accuracy_, loss_ = sess.run(
                        [rnn.accuracy, rnn.loss], feed_dict=feed_dict)
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
