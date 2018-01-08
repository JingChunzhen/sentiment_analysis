import os
import re
import sqlite3

import numpy as np
import pandas as pd
from tensorflow.contrib import learn


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    this is very crude
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data():
    '''
    tested 
    load data from csv file of Amazon Unlocked Mobile 
    rid nans 
    Returns:
        x (list) shape (nums of data, str length)
        y (list) shape (nums of data, 2)
    '''
    # load
    df = pd.read_csv('../data/Amazon_Unlocked_Mobile.csv')
    df = df[df['Rating'] != 3][['Reviews', 'Rating']]
    df = df.dropna(axis=0, how="any")

    # data
    Reviews = df['Reviews'].tolist()
    x = [clean_str(review) for review in Reviews]

    # label
    Ratings = df['Rating'].tolist()
    y = []
    [y.append([0, 1] if rating == 1 or rating == 2 else [1, 0])
     for rating in Ratings]

    return x, y


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    need to be tested 
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def convert_to_sqlite(file_in, sql_path):
    '''    
    depreated
    convert the csv to db file (sqlite)
    Args:
        file_in (string)
        sql_path (string)
    '''
    conn = sqlites.connect(sql_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE amazon_mobiles
        (REVIEWS TEXT NOT NULL,
        POLARITY INT NULL);''')

    df = pd.read_csv(file_in)
    data = []
    for index, row in df.iterrows():
        if row['Rating'] == 3:
            continue
        else:
            polarity = 1 if row['Raing'] == 4 or row['Rating'] == 5 else 0
            data.append((row['Review'], polarity))

    c.executemany('INSERT INTO amazon_mobiles VALUES (?,?)', data)
    conn.commit()
    conn.close()


def statistics_amazon():
    '''
    deprecated 
    statistics for amazon unlocked phones 
    '''
    df = pd.read_csv('../data/Amazon_Unlocked_Mobile.csv')

    print(df['Rating'].count())  # 413840
    print(df[(df['Rating'] == 1)]['Rating'].count())  # 72350
    print(df[(df['Rating'] == 2)]['Rating'].count())  # 24728
    print(df[(df['Rating'] == 3)]['Rating'].count())  # 31765
    print(df[(df['Rating'] == 4)]['Rating'].count())  # 61392
    print(df[(df['Rating'] == 5)]['Rating'].count())  # 223605

    print(df['Reviews'].count())  # 413778
    print(df[(df['Rating'] == 1)]['Reviews'].count())  # 72337
    print(df[(df['Rating'] == 2)]['Reviews'].count())  # 24728
    print(df[(df['Rating'] == 3)]['Reviews'].count())  # 31765
    print(df[(df['Rating'] == 4)]['Reviews'].count())  # 61392
    print(df[(df['Rating'] == 5)]['Reviews'].count())  # 223605

    # row['Reviews'] -> str


if __name__ == '__main__':
    x, y = load_data()

    max_document_length = max([len(text.split(' ')) for text in x])
    processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    res = processor.fit_transform(x)
    res_list = list(res)

    print(len(res_list))
