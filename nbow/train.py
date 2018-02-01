import os
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

from nbow_model import NBOW
from nbow_unformed import NBOW_unformed
from nbow_ngram import NBOW_Ngram
from nbow_fc import NBOW_fc
from utils.data_parser import batch_iter, load_data
from itertools import chain

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
        raw_x, raw_y = load_data(
            params_global["task"], params_global["num_classes"])
        self.max_document_length = 163

        # if os.path.exists(params_global["vocabulray"]):
        #     self.processor = learn.preprocessing.VocabularyProcessor.restore(
        #         "../temp/data/test_vocabulary_processor")
        # else:
        #     self.processor = learn.preprocessing.VocabularyProcessor(
        #         self.max_document_length)
        #     self.processor.fit(raw_x)

        self.processor = learn.preprocessing.VocabularyProcessor(
            self.max_document_length)
        self.processor.fit(raw_x)

        raw_x = list(self.processor.transform(raw_x))

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
        
        self.embedding_matrix = self._embedding_matrix_initializer() if params["embedding_init"] else None
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
            embedding_matrix = list(chain.from_iterable(self.embedding_matrix)) if self.embedding_matrix else None
            nbow = NBOW_Ngram(
                sequence_length=self.max_document_length,
                num_classes=params_global["num_classes"],
                vocab_size=len(self.processor.vocabulary_),
                embedding_size=params_global["embedding_size"],
                weighted=params["weighted"],
                l2_reg_lambda=params["l2_reg_lamda"],
                embedding_init=params["embedding_init"],                
                embedding_matrix=embedding_matrix,
                static=params["static"]
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
                    pred_, accuracy_, loss_ = sess.run(
                        [nbow.predictions, nbow.accuracy, nbow.loss], feed_dict=feed_dict)
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
                        print(classification_report(y_true=y_true, y_pred=y_pred))
    
    
if __name__ == "__main__":
    eval = EVAL()
    eval.process(
        learning_rate=1e-3,
        batch_size=128,
        epochs=100,
        evaluate_every=1000
    )
