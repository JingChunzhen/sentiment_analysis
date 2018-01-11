import tensorflow as tf
import numpy as np


class NBOW(object):

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, l2_reg_lambda):
        '''
        '''
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name="x")
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="y")

        l2_loss = tf.constant(0.0)

        with tf.name_scope("embedding-laye"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)            

        with tf.name_scope("averaging-layer"):
            mask = tf.sign(self.input_x)
            range_ = tf.range(
                start=1, limit=sequence_length + 1, dtype=tf.int32)
            mask = tf.multiply(mask, range_, name="mask")  # element wise # shape: batch_size, sequence_length
            mask = tf.cast(mask, dtype=tf.float32)
            # mask = tf.unstack(mask,  num=tf.shape(self.input_x)[0], axis=0)
            # print(mask)
            # print(mask[0])
            mask = tf.expand_dims(mask, -1) # shape: batch_size, sequence_length, 1            
            self.seq_len = tf.reduce_max(mask, axis=1) # shape: batch_size      
            #self.embedded_chars = tf.transpose(self.embedded_chars, [0, 2, 1])  # get shape batch_size, embedding_size, sequence_length
            print(self.embedded_chars)    

            batch_size = tf.shape(self.input_x)[0] # this num param should be a num not a tensor 
            # self.embedded_chars = tf.unstack(self.embedded_chars, num=batch_size, axis=0)
            # cannot infer num from shape (?)
            self.embedded_chars = tf.unstack(self.embedded_chars, axis=0)
            print(self.embedded_chars)
            nbow_output = tf.dot(self.embedded_chars, mask) 

            nbow = tf.tensordot()
            print(nbow_output)            
            self.nbow_output = tf.div(nbow_output, self.seq_len)                    
            print(nbow_output)
        
        '''
        with tf.name_scope("fully-connected-layer"):
            W = tf.Variable(tf.truncated_normal(
                [embedding_size, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(
                self.nbow_output, W, b, name="scores")   
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
        '''
tf.nn.softmax_cross_entropy_with_logits
tf.tensordot
import pandas as pd
from tensorflow.contrib import learn
import sys
sys.path.append("..")
from rnn.rnn_model import RNN
df = pd.read_csv('../data/Amazon_Unlocked_Mobile.csv')
df = df[:10]
df = df[df['Rating'] != 3][['Reviews', 'Rating']]
x = df['Reviews']
Ratings = df['Rating'].tolist()
y = []
[y.append([0, 1] if rating == 1 or rating == 2 else [1, 0])
    for rating in Ratings] 
max_document_length = 163 
processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = list(processor.fit_transform(x))
x = [l.tolist() for l in x]

nbow = NBOW(
    sequence_length=163,
    num_classes=2,
    vocab_size=len(processor.vocabulary_),
    embedding_size=100,
    l2_reg_lambda=0.0
)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    ecs, opt = sess.run(
        [nbow.embedded_chars, nbow.nbow_output],
        feed_dict={
            nbow.input_x:x,
            nbow.input_y:y,           
        }
    )

    print(np.shape(ecs))
    print(np.shape(opt))