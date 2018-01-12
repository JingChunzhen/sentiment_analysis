import tensorflow as tf
from tensorflow.contrib import learn

import numpy as np
import pandas as pd


def test_vocab_processor():
    file_in = "../data/Amazon_Unlocked_Mobile.csv"
    df = pd.read_csv(file_in)[:10]
    texts = df['Reviews'].tolist()

    print(type(texts))
    print(len(texts))
    # print(texts)

    max_document_length = max([len(text.split(' ')) for text in texts])
    processor = learn.preprocessing.VocabularyProcessor(max_document_length)

    res = processor.fit_transform(texts)
    res = list(res)  # generator -> list

    print(np.shape(res))

    for text in res:
        print(text)  # get list of word ids

    print("processor")
    docs = processor.reverse(res)
    # reverse word ids to word
    docs = list(docs)  # generator -> list


def test_calc_seq_len():
    file_in = "../data/Amazon_Unlocked_Mobile.csv"
    df = pd.read_csv(file_in)[:5]
    texts = df['Reviews'].tolist()

    max_document_length = max([len(text.split(' ')) for text in texts])
    processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    res = list(processor.fit_transform(texts))
    print(np.shape(res))  # shape (5, 72)
    print(type(res), type(res[0]))  # list np.ndarray
    res = [a.tolist() for a in res]  # 转换之后可以成功运行
    print(type(res), type(res[0]))  # list np.ndarray
    mask = tf.sign(res)
    # tf.sign的参数只可以使用list(list)的参数 如果参数为list(array)则会报错
    print(mask)

    range_ = tf.range(start=1, limit=max_document_length + 1, dtype=tf.int32)
    print(range_)

    mask1 = tf.multiply(mask, range_)
    seq_len = tf.reduce_max(mask1, axis=1)
    print(seq_len)

    with tf.Session() as sess:
        m, r, m1, sl = sess.run([mask, range_, mask1, seq_len])
        print(m)
        print(r)
        print(m1)
        print(sl)


def test_model():
    '''
    test model  
    '''
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

    rnn = RNN(
        sequence_length=max_document_length,
        num_classes=2,
        embedding_size=100,
        vocab_size=len(processor.vocabulary_),
        hidden_size=128,
        num_layers=4,
        l2_reg_lambda=0
    )

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        ecs, ls = sess.run(
            [rnn.embedded_chars, rnn.loss],
            feed_dict={
                rnn.input_x: x,
                rnn.input_y: y,
                rnn.input_keep_prob: 1.0,
                rnn.output_keep_prob: 1.0
            }
        )


def test_tensordot_toy():
    a = np.arange(60).reshape(3, 4, 5)
    b = np.arange(24).reshape(4, 3, 2)

    print(a)
    print(b)
    c = np.tensordot(a, b, axes=([1, 0], [0, 1]))
    print(c)
    pass


def test_tensordot(batch_size, sequence_length, embedding_size):
    data = np.arange(10000).reshape(
        batch_size, sequence_length, embedding_size)
    mask = np.arange(200).reshape(batch_size, sequence_length, 1)
    res = np.tensordot(data, mask, axes=([0, 1], [0, 1]))
    print(np.shape(res))


def test_tf_multiply(batch_size, sequence_length, embedding_size):
    data = np.arange(60).reshape(
        batch_size, sequence_length, embedding_size)  # 4 5 3
    mask = np.arange(20).reshape(batch_size, 1, sequence_length)  # 4 5 1

    res = np.dot(mask, data)
    print(np.shape(res))
    '''
    seq_len = np.arange(4).reshape(batch_size)

    print(mask)
    mask = np.tile(mask, reps=embedding_size)
    print(np.shape(mask))
    print(mask)
    res = np.multiply(data, mask)
    res = np.sum(res, axis=1)
    print(res)
    print(np.shape(res))    
    print(np.shape(res))
    '''


def test_tile():
    a = tf.constant(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0]
        ]
    )
    a = tf.expand_dims(a, -1)
    a = tf.tile(a, multiples=[1, 1, 5])
    shape = tf.shape(a)

    with tf.Session() as sess:
        shape_ = sess.run(a)
        print(shape_)

def test_div():
    a = [
        [10,2,6,2,4],
        [3,9,6,9,27],
        [4,16,8,4,4],
    ]

    b = [
        2, 3, 4
    ]
    
    print(b)
    c = np.eye(3)
    print(c)
    c = np.multiply(c, b)
    print(c)
    c = np.linalg.inv(c)
    print(c)

if __name__ == "__main__":
    #test_tf_multiply(4, 5, 3)
    #test_tile()
    test_div()
