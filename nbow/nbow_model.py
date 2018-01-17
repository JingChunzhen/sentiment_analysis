import tensorflow as tf


class NBOW(object):
    '''
    Neural Bag of Words
    '''

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, weighted, l2_reg_lambda):
        '''        
        Args:
            sequence_length (int):
            num_classes (int):
            vocab_size (int):
            embedding_size (int):
            weighted (boolean): Learning Word Importance with the Neural Bag-of-Words Model
                cf. http://www.aclweb.org/anthology/W/W16/W16-1626.pdf            
            l2_reg_lamda (float)：
        '''
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name="x")
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="y")

        l2_loss = tf.constant(0.0)

        with tf.name_scope("embedding-layer"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

        if weighted:            
            with tf.name_scope("weighted"):
                w = tf.Variable(tf.random_uniform([1, embedding_size, 1]), name="weight")
                w = tf.tile(w, multiples=[tf.shape(self.input_x)[0], 1, 1])
                # batch_size, embedding_size, 1
                w = tf.matmul(self.embedded_chars, w)
                # batch_size, sequence_length, 1
                w = tf.nn.sigmoid(w)
                # batch_size, sequence_length, 1
                w = tf.tile(w, multiples=[1, 1, embedding_size])
                # batch_size, sequence_length, embedding_size
                self.embedded_chars = tf.multiply(self.embedded_chars, w)
                # batch_size, sequence_length, embedding_size
        
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
