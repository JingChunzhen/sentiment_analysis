import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import DropoutWrapper


class RNN(object):

    def __init__(self, num_classes, vocab_size, embedding_size, hidden_size, num_layers, l2_reg_lambda):
        '''
        Args:
            num_classes (int):
            vocab_size (int):
            embedding_size (int):            
            hidden_size (int):
            num_layer (int):
            l2_reg_lambda (float):
        '''
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name="x")
        # convert to dtype: list(list) in case the error in tf.sign
        # original dtype list(np.ndarray)
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="y")  # dtype: list(list)

        self.input_keep_prob = tf.placeholder(
            tf.float32, name="keep_prob_inp")
        self.output_keep_prob = tf.placeholder(
            tf.float32, name="keep_prob_out")

        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

        with tf.name_scope("sequence length"):
            mask = tf.sign(self.input_x)
            range_ = tf.range(
                start=1, limit=self.sentence_len + 1, dtype=tf.int32)
            mask = tf.multiply(mask, range_, name="mask")  # element wise
            self.seq_len = tf.reduce_max(mask, axis=1)

        with tf.name_scope("rnn process"):
            '''
            sequence_length
            '''
            self.rnn_cell = DropoutWrapper(
                GRUCell(hidden_size),
                input_keep_prob=self.input_keep_prob,
                output_keep_prob=self.output_keep_prob
            )
            rnn_cell_seq = tf.contrib.rnn.MultiRNNCell(
                [self.rnn_cell] * num_layers, state_is_tuple=True)

            outputs, _, _ = tf.nn.static_bidirectional_rnn(
                inputs=self.embedded_chars,
                cell_fw=rnn_cell_seq,
                cell_bw=rnn_cell_seq,
                sequence_length=self.seq_lengths
            )
            self.output = tf.concat(values=outputs, axis=1) # TODO need to be tested 
            
        # Final (unnormalized) scores and predictions        
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[2 * hidden_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.output, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")        


if __name__ == '__main__':
    # def __init__(self, num_classes, vocab_size, embedding_size, hidden_size, num_layers, )
    RNN(2, 100, 100, 100, 4)
    pass