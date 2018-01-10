import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import DropoutWrapper


class RNN(object):

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, hidden_size, num_layers, l2_reg_lambda):
        '''
        Args:
            sequence_length (int)
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
            tf.float32, name="keep_prob_in")
        self.output_keep_prob = tf.placeholder(
            tf.float32, name="keep_prob_out")

        l2_loss = tf.constant(0.0)

        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars = tf.unstack(
                embedded_chars, sequence_length, axis=1)

        with tf.name_scope("sequence-length"):
            mask = tf.sign(self.input_x)
            range_ = tf.range(
                start=1, limit=sequence_length + 1, dtype=tf.int32)
            mask = tf.multiply(mask, range_, name="mask")  # element wise
            self.seq_len = tf.reduce_max(mask, axis=1)

        with tf.name_scope("forward_cell"):
            if num_layers != 1:
                cells = []
                for i in range(num_layers):
                    rnn_cell = DropoutWrapper(
                        GRUCell(hidden_size),
                        input_keep_prob=self.input_keep_prob,
                        output_keep_prob=self.output_keep_prob
                    )
                    cells.append(rnn_cell)
                self.cell_fw = MultiRNNCell(cells)
            else:
                self.cell_fw = DropoutWrapper(
                    GRUCell(hidden_size),
                    input_keep_prob=self.input_keep_prob,
                    output_keep_prob=self.output_keep_prob
                )

        with tf.name_scope("backward_cell"):
            if num_layers != 1:
                cells = []
                for i in range(num_layers):
                    rnn_cell = DropoutWrapper(
                        GRUCell(hidden_size),
                        input_keep_prob=self.input_keep_prob,
                        output_keep_prob=self.output_keep_prob
                    )
                    cells.append(rnn_cell)
                self.cell_bw = MultiRNNCell(cells)
            else:
                self.cell_bw = DropoutWrapper(
                    GRUCell(hidden_size),
                    input_keep_prob=self.input_keep_prob,
                    output_keep_prob=self.output_keep_prob
                )

        with tf.name_scope("rnn_with_{}_layers".format(num_layers)):
            outputs, _, _ = tf.nn.static_bidirectional_rnn(
                inputs=self.embedded_chars,
                cell_fw=self.cell_fw,
                cell_bw=self.cell_bw,
                sequence_length=self.seq_len,
                dtype=tf.float32
            )
            # If no initial_state is provided, dtype must be specified
            self.rnn_output = tf.concat(values=outputs[-1], axis=1)

        with tf.name_scope("fully_connected_layer"):
            W = tf.Variable(tf.truncated_normal(
                [hidden_size * 2, 2], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(
                self.rnn_output, W, b, name="scores")  # TODO error
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
