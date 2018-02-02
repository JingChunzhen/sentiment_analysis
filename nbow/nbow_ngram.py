import tensorflow as tf


class NBOW_Ngram(object):
    '''
    Sentiment Analysis and Lexicon Polarity Visualization Using Neural Bag of Words with N-Gram (NBOW-NG)    
    Sentiment Analysis and Lexicon Polarity Visualization based on Neural Bag of Words and N-Gram (NBOW-NG)
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
                    embedding_matrix), name='embedding_W', trainable=not static)
                self.embedded_chars = tf.nn.embedding_lookup(
                    self.W, self.input_x)
        else:
            with tf.name_scope("embedding-layer"):
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="embedding_W")
                self.embedded_chars = tf.nn.embedding_lookup(
                    self.W, self.input_x)

        with tf.name_scope("mask-layer"):
            mask = tf.sign(self.input_x)
            # shape: batch_size, sequence_length
            mask = tf.expand_dims(mask, -1)
            # shape: batch_size, sequence_length, 1
            mask = tf.tile(mask, multiples=[1, 1, embedding_size])
            # shape: batch_size, sequence_length, embedding_size
            mask = tf.cast(mask, dtype=tf.float32)
            self.embedded_chars = tf.multiply(self.embedded_chars, mask)
            # shape: batch_size, sequence_length, embedding_size

        with tf.name_scope("unigram-layer"):
            self.unigram_w = tf.Variable(tf.truncated_normal(
                [embedding_size, num_classes], stddev=0.1), name="unigram_W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.unigram = tf.tensordot(
                self.embedded_chars, self.unigram_w, [[2], [0]])
            # self.unigram = tf.nn.softmax(unigram, dim=-1)

        with tf.name_scope("bigram-layer"):
            w = tf.Variable(tf.truncated_normal(
                [embedding_size, num_classes], stddev=0.1), name="W")
            # b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            inpt = tf.expand_dims(self.embedded_chars, -1)
            # shape: batch_size, sequence_length, embedding_size, 1
            bigram = tf.nn.avg_pool(
                inpt, ksize=[1, 2, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
            # shape: batch_size, 2, 100, 1
            bigram = tf.squeeze(bigram, axis=3)
            self.bigram = tf.tensordot(bigram, w, [[2], [0]])

        with tf.name_scope("trigram-layer"):
            w = tf.Variable(tf.truncated_normal(
                [embedding_size, num_classes], stddev=0.1), name="W")
            # b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            inpt = tf.expand_dims(self.embedded_chars, -1)
            trigram = tf.nn.avg_pool(
                inpt, ksize=[1, 3, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
            trigram = tf.squeeze(trigram, axis=3)
            self.trigram = tf.tensordot(trigram, w, [[2], [0]])

        with tf.name_scope("softmax-layer"):
            self.unigram = tf.nn.softmax(self.unigram, axis=-1)
            self.unigram = tf.reduce_sum(self.unigram, axis=1)
            self.bigram = tf.nn.softmax(self.bigram, axis=-1)
            self.bigram = tf.reduce_sum(self.bigram, axis=1)
            self.trigram = tf.nn.softmax(self.trigram, axis=-1)
            self.trigram = tf.reduce_sum(self.trigram, axis=1)
            self.output = self.unigram + self.bigram + self.trigram

        with tf.name_scope("predictions"):
            self.predictions = tf.argmax(
                self.output, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.output, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)  # + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.name_scope("visualization"):
            mask = tf.sign(self.input_x)
            range_ = tf.range(
                start=1, limit=sequence_length + 1, dtype=tf.int32)
            mask = tf.multiply(mask, range_, name="mask")  # element wise
            self.seq_len = tf.reduce_max(mask, axis=1)

            dist = tf.matmul(self.W, self.unigram_w)
            dist = tf.nn.softmax(dist, axis=-1)
            # shape: vocab_size, num_classes
            self.dist = tf.nn.embedding_lookup(dist, self.input_x)
