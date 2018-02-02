import numpy as np
import scipy
import tensorflow as tf


def test_nbow_weight():
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
        ], dtype=tf.float16
    )  # 2, 3 ,5

    weight = tf.constant([3, 7, 0, -5, 2], dtype=tf.float16)
    a = tf.constant([3, 7, 0, -5, 2], dtype=tf.float16)
    a = tf.expand_dims(a, -1)
    weight = tf.expand_dims(weight, -1)  # embedding_size, 1
    weight = tf.expand_dims(weight, 0)  # 1, embedding_size, 1
    # batch_size, embedding_size, 1
    weight = tf.tile(weight, multiples=[tf.shape(input_x)[0], 1, 1])
    weight = tf.matmul(input_x, weight)  # batch_size, sequence_length, 1
    # weight = tf.squeeze(weight, axis=2) # batch_size, sequence_length
    temp = tf.cast(weight, tf.float16)
    weight = tf.nn.sigmoid(temp)  # batch_size, sequence_length, 1

    weight = tf.tile(weight, multiples=[1, 1, embedding_size])

    res = tf.multiply(input_x, weight)

    td_res = tf.tensordot(input_x, a, axes=[[2], [0]])
    # batch_size, sequence_length, 1
    print(td_res)
    td_res = tf.nn.sigmoid(td_res)
    print(td_res)

    with tf.Session() as sess:
        # temp_ = sess.run(temp)
        # print(temp_)
        # w = sess.run(weight)
        # print(w)
        td = sess.run(td_res)
        print(np.shape(td_res))


def test_rnn_attention():
    # np.random.randn(4, 5, 6) -> get matrix with shape (4, 5, 6)
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
        ], dtype=tf.float16
    )  # 2, 3 ,5

    # represent batch_size, sequence_length, hidden_size
    # or sequence_length, batch_size, hidden_size
    # or sequence_length, batch_size, 2 * hidden_size

    a = tf.constant(
        [
            [2, 1, 1, 5],
            [2, 3, 4, 5],
            [2, 3, 4, 3],
            [5, 2, 6, 3],
            [6, 3, 2, 6]
        ], dtype=tf.float16
    )

    res = tf.tensordot(input_x, a, axes=[[2], [0]])
    # batch_size, sequence_length, 4 (2, 3, 4)

    with tf.Session() as sess:
        res_ = sess.run(res)
        print(np.shape(res_))


def test_nbow_with_softmax():
    '''
    test_new_method
    '''
    sequence_length = 3
    embedding_size = 5
    batch_size = 2
    num_classes = 7

    x = tf.constant(
        [
            [
                [4, 8, 3, 4, 1],
                [3, 4, 9, 8, 5],
                [5, 7, 3, 9, 1]
            ],
            [
                [2, 3, 5, 6, 1],
                [6, 7, 9, 8, 2],
                [3, 4, 6, 7, 9]
            ],
        ], dtype=tf.float32
    )

    w = tf.Variable(
        tf.random_uniform([embedding_size, num_classes], -1.0, 1.0)
    )

    res = tf.tensordot(x, w, [[2], [0]])  # (2, 3, 7)
    res = tf.nn.softmax(res, dim=2)  # (2, 3, 7)
    # (2, 3, 7) set 0 for length greater than seqlen
    # multiply
    res = tf.reduce_sum(res, axis=1)  # argmax
    predict = tf.argmax(res, 1)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        x_ = sess.run(predict)
        print(np.shape(x_))
        print(x_)


def test_avg_pool():
    # batch_size 2
    # sequence_length 3
    # embedding_size 5
    # num_classes 3
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
        ], dtype=tf.float32
    )  # 2, 3 ,5

    # tf.nn.avg_pool()
    inpt = tf.expand_dims(input_x, -1)
    print(inpt)
    # shape: batch_size, sequence_length, embedding_size, 1

    with tf.name_scope("bigram"):
        w = tf.Variable(tf.truncated_normal([5, 3], stddev=0.1))
        bigram = tf.nn.avg_pool(
            inpt, ksize=[1, 2, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
        # bigram -> ksize = [1, 2, 1, 1]
        # trigram -> ksize = [1, 3, 1, 1]
        bigram = tf.squeeze(bigram, 3)
        bigram = tf.tensordot(bigram, w, [[2], [0]])
    print(bigram)

    with tf.name_scope("trigram"):
        trigram = tf.nn.avg_pool(
            inpt, ksize=[1, 3, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
        # ksize
        trigram = tf.squeeze(trigram, 3)
    print(trigram)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        bg = sess.run(bigram)
        print(bg)
        print(np.shape(bg))
        tg = sess.run(trigram)
        print(tg)
        print(np.shape(tg))


def test_embedding_lookup():
    '''
    '''
    a = tf.constant(
        [
            [2, 1, 1, 5],
            [2, 3, 4, 5],
            [2, 3, 4, 3],
            [5, 2, 6, 3],
            [6, 3, 2, 6]
        ], dtype=tf.float16
    )

    b = tf.constant(
        [
            [2, 3, 4],
            [2, 3, 6],
            [5, 6, 7],
            [5, 6, 8]
        ], dtype=tf.float16
    )

    c = tf.tensordot(a, b, [[1], [0]])
    res = tf.nn.embedding_lookup(c, [1, 4])

    with tf.Session() as sess:
        res_ = sess.run(res)
        print(res_)

def test_get_variable():
    #with tf.name_scope("")
    a = tf.Variable(
        tf.random_normal([2, 3], name="test_w")
    )
    b = tf.get_variable(name="test_w", shape=[2, 3], trainable=False)
    print(a)
    print(b)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        a_, b_ = sess.run([a, b])
        print(a_)
        print(b_)

    pass
if __name__ == '__main__':
    # test_nbow_with_softmax()
    # test_avg_pool()
    #test_embedding_lookup()
    #test_get_variable()
    a = 100
    b = 200
    c = a % b
    print(c)
    pass
