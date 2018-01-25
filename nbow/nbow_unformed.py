import tensorflow as tf


class NBOW_unformed(object):
    '''
    Neural Bag of Words Using Softmax 
    future work:
        implement bi-gram or tri-gram to test feature combination
        use fully-connected layer in the final to test the result 
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

        with tf.name_scope("weighted-layer"):
            w = tf.Variable(tf.truncated_normal(
                [embedding_size, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            output = tf.tensordot(self.embedded_chars, w, [[2], [0]])
            output = tf.nn.softmax(output, dim=-1)

        with tf.name_scope("mask-layer"):
            mask = tf.sign(self.input_x)
            # shape: batch_size, sequence_length
            mask = tf.expand_dims(mask, -1)
            # shape: batch_size, sequence_length, 1
            mask = tf.tile(mask, multiples=[1, 1, num_classes])
            # shape: batch_size, sequence_length, num_classes
            mask = tf.cast(mask, dtype=tf.float32)
            output = tf.multiply(output, mask)
            # shape: batch_size, sequence_length, num_classes
            self.nbow_output = tf.reduce_sum(output, axis=1)

        with tf.name_scope("predictions"):
            self.predictions = tf.argmax(
                self.nbow_output, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.nbow_output, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)  # + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")
