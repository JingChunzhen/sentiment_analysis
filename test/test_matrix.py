import numpy as np
import scipy
import tensorflow as tf


'''
with tf.name_scope("averaging-layer"):
    mask = tf.sign(self.input_x)
    # shape: batch_size, sequence_length            

    range_ = tf.range(
        start=1, limit=sequence_length + 1, dtype=tf.int32)
    seq_len = tf.reduce_max(tf.multiply(
        mask, range_), axis=1)            
    seq_len = tf.cast(seq_len, dtype=tf.float32)
    # shape: batch_size

    divisor = tf.eye(tf.shape(self.input_x)[0])
    divisor = tf.multiply(divisor, seq_len)            
    divisor = tf.matrix_inverse(divisor)            
    # element in seq_len must be all greater than 0 in case the error: input is not invertible            
    # shape: batch_size, batch_size

    mask = tf.expand_dims(mask, -1)
    # shape: batch_size, sequence_length, 1
    mask = tf.tile(mask, multiples=[1, 1, embedding_size])
    # shape: batch_size, sequence_length, embedding_size
    mask = tf.cast(mask, dtype=tf.float32)

    output = tf.multiply(self.embedded_chars, mask)
    # shape: batch_size, sequence_length, embedding_size
    output = tf.reduce_sum(output, axis=1)
    # shape: batch_size, embedding_size

    output = tf.matmul(tf.transpose(output), divisor)
    self.nbow_output = tf.transpose(output)
'''


sequence_length = 3
embedding_size = 5
input_x = tf.constant(
    [
        [
            [1, 2, -3, 4, -5],
            [1, -5, 7, -8, 8],
            [4, -5, 8, 2, 3]
        ],
        [
            [5, -8, -6, 5, 5],
            [2, 3, 4, -5, 2],
            [5, -4, -9, 2, -4]
        ]
    ]
)

weight = tf.constant([3, 7, 0, -5, 2])
weight = tf.expand_dims(weight, -1) # embedding_size, 1
weight = tf.expand_dims(weight, 0) # 1, embedding_size, 1
weight = tf.tile(weight, multiples=[tf.shape(input_x)[0], 1, 1]) # batch_size, embedding_size, 1
weight = tf.matmul(input_x, weight) # batch_size, sequence_length, 1 
# weight = tf.squeeze(weight, axis=2) # batch_size, sequence_length
temp = tf.cast(weight, tf.float16)
weight = tf.nn.sigmoid(temp) # batch_size, sequence_length, 1

weight = tf.tile(weight, multiples=[1, 1, embedding_size])
res = tf.multiply(input_x, weight)









with tf.Session() as sess:
    temp_ = sess.run(temp)
    print(temp_)
    w = sess.run(weight)
    print(w)
   