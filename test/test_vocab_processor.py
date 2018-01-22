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
    for doc in docs:
        print(doc)
    # docs = list(docs)  # generator -> list

    for idx in range(len(processor.vocabulary_)):
        word = processor.vocabulary_.reverse(idx)
        print(idx, word)


def test_embedding_proessor(embedding_size, max_document_length):
    '''
    initialize the embedding_layer using pre-trained word2vec
    # TODO test save and restore in VocabularyProcessor 
    # TODO test save specified weights in tensorflow model to get word embedding after supervised learning 
    initialize the embedding_layer using tf.assign(ref, value)
    '''
    file_in = "../data/Amazon_Unlocked_Mobile.csv"
    df = pd.read_csv(file_in)  # [:10]
    df = df.dropna(axis=0, how="any")
    texts = df['Reviews'].tolist()

    file_wv = "../data/glove.6B/glove.6B.{}d.txt".format(embedding_size)
    wv = {}
    with open(file_wv, 'r') as f:
        for line in f:
            line = line.split(' ')
            word = line[0]
            wv[word] = list(map(float, line[1:]))            

    processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    processor.fit_transform(texts)

    embedding_matrix = []
    for idx in range(len(processor.vocabulary_)):
        word = processor.vocabulary_.reverse(idx)
        embedding_matrix.append(
            wv[word] if word in wv else np.random.normal(size=embedding_size))

    # del wv
    return processor, embedding_matrix


if __name__ == "__main__":
    p, e = test_embedding_proessor(100, 163)
    print(np.shape(e))
    pass
