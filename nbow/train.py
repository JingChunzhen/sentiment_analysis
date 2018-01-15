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

from nbow_model import NBOW
from utils.data_parser import batch_iter, load_data


with open('../config.yaml', 'rb') as f:
    param_all = yaml.load(f)
    params = param_all["NBOW"]
    params_global = param_all["Global"]


class EVAL(object):

    def __init__(self):
        '''
        tested 
        get the map of word to ids 
        get the split of train, dev and test data and labels                
        '''
        raw_x, raw_y = load_data(params_global["task"], params_global["num_classes"])
        print(len(raw_x))
        self.max_document_length = 163
        # optimum max_document_length coming from ../test/statistics
        # processor.restore(file_path)
        self.processor = learn.preprocessing.VocabularyProcessor(
            self.max_document_length)
        raw_x = list(self.processor.fit_transform(raw_x))

        # rid x with all zeros 
        x, y = [], []
        for tmp_x, tmp_y in zip(raw_x, raw_y):
            tmp_x = tmp_x.tolist()
            if np.sum(tmp_x) != 0:
                x.append(tmp_x)
                y.append(tmp_y)                

        x_temp, self.x_test, y_temp, self.y_test = train_test_split(
            x, y, test_size=params_global["test_size"])
        self.x_train, self.x_validate, self.y_train, self.y_validate = train_test_split(
            x_temp, y_temp, test_size=params_global["validate_size"])

        # free
        del x_temp, y_temp, raw_x, x, y

    def process(self, learning_rate, batch_size, epochs, evaluate_every):

        with tf.Graph().as_default():

            nbow = NBOW(
                sequence_length=self.max_document_length,
                num_classes=params_global["num_classes"],
                vocab_size=len(self.processor.vocabulary_),
                embedding_size=params_global["embedding_size"],
                l2_reg_lambda=params["l2_reg_lamda"]
            )            

            global_step = tf.Variable(0, trainable=False)
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(
                nbow.loss, global_step=global_step)

            init = tf.global_variables_initializer()
            with tf.Session() as sess:

                def train_step(x_batch, y_batch):
                    feed_dict = {
                        nbow.input_x: x_batch,
                        nbow.input_y: y_batch                        
                    }
                    _, step, accuracy_, loss_ = sess.run(
                        [train_op, global_step, nbow.accuracy, nbow.loss], feed_dict=feed_dict)
                    return step, accuracy_, loss_

                def dev_step(x_batch, y_batch):
                    feed_dict = {
                        nbow.input_x: x_batch,
                        nbow.input_y: y_batch                        
                    }
                    accuracy_, loss_ = sess.run(
                        [nbow.accuracy, nbow.loss], feed_dict=feed_dict)
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
