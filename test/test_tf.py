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
    print(np.shape(res)) # shape (5, 72)
    print(type(res), type(res[0])) # list np.ndarray
    res = [a.tolist() for a in res] # 转换之后可以成功运行
    print(type(res), type(res[0])) # list np.ndarray
    mask = tf.sign(res)
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
    


def test_tf_sign():
    x = [1, 2, 4, 8, 1, 2, 5, 0, 0, 0, 0, 0, 2, 5, 4, 6]
    res = tf.sign(x)
    print(res) # shape (16, ) dtype = int32 
    print(type(res))

    x = 0
    res = tf.sign(x)
    print(res)
    print(type(res))

    x = 1
    res = tf.sign(x)
    print(res)
    print(type(res))

    x = [
        [1,2,3,4,0,6,7,8,9,0],
        [0,0,0,0,0,2,3,1,5,1]
    ]    
    res = tf.sign(x) # 使用sign转换只能使用list list 不能使用array
    print(res) # Tensor shape (2, 10) dtype = int32
    print(type(res))

    x = [
        np.array([1,2,3,4,0,6,7,8,9,0]),
        np.array([0,0,0,0,0,2,3,1,5,1])
    ]
    res = tf.sign(x) # error occurred 
    print(x)
    print(res) 
    print(type(res))

def test():
    x = [
        np.array([1,2,3,4,0,6,7,8,9,0]),
        np.array([0,0,0,0,0,2,3,1,5,1])
    ]   
    # x x[0] 均可以迭代
    return hasattr(x[0], '__iter__') 


if __name__ == "__main__":
    # test_calc_seq_len()
    from tensorflow.nn import GRU
    # test_tf_sign()
    # print(test())
