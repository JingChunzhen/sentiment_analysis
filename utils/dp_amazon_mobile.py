import os
import re
import sqlite3
import time
import pickle

import numpy as np
import pandas as pd
from pyecharts import Line
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
    
    x = [] # data 
    y = [] # label

    for _, row in df.iterrows():
        review = row['Reviews']
        rating = row['Rating']
        review = clean_str(review)
        tmp = review.strip().replace(" ", '')
        if len(tmp) != 0:
            x.append(review)
            y.append([0, 1] if rating == 1 or rating == 2 else [1, 0])
    
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
            polarity = 1 if row['Rating'] == 4 or row['Rating'] == 5 else 0
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


def statistics_processor():            
    x, y = load_data()
    l = [len(text.split(' ')) for text in x]
    max_document_length = max(l)

    print("max document length: {}".format(max_document_length)) # 5655
    print("mean document length: {}".format(np.mean(l))) # 42.41
    print("median document length: {}".format(np.median(l))) # 18.0

    processor = learn.preprocessing.VocabularyProcessor(max_document_length)    
    document_list = list(processor.fit_transform(x))

    print("data size: {}".format(len(document_list))) 
    # 382015 trim the empty review get data size 381643
    print("vocab size： {}".format(len(processor.vocabulary_))) # 65434

    # reviews_num = {}

    # for length in l:        
    #     if length not in reviews_num:
    #         reviews_num[length] = 1
    #     else:
    #         reviews_num[length] += 1
         
    # line = Line("")    
    # axis0 = []
    # axis1 = []
    # sorted(reviews_num.items(), key=lambda d: d[0])
    # for k, v in reviews_num.items():
    #     axis0.append(k)
    #     axis1.append(v)
    
    # with open("../temp/reviews_num.pkl", 'wb') as f:
    #     pickle.dump(reviews_num, f)

    # line.add("评论长度人数", axis0, axis1)        
    # time_stamp = time.strftime("_%Y-%m-%d_%H:%M:%S", time.localtime())
    # file_name = "statistics_for_doc_length" + time_stamp    
    # line.render('../doc/{}'.format(file_name) + '.html')        


def optimize_max_document_length():
    '''
    To find the optimum max_document_length
    get the graph in ../temp/statistics_for_num_percentage.html
    '''
    with open('../temp/reviews_num.pkl', 'rb') as f:
        reviews_num = pickle.load(f)
    
    data_size = 382015
    reviews_percentage = {}
    s = 0
    for k, v in reviews_num.items():
        s += v
        percentage = s * 1.0 / data_size
        reviews_percentage[k] = percentage
        
    line = Line("")    
    axis0 = []
    axis1 = []

    for k, v in reviews_percentage.items():
        axis0.append(k)
        axis1.append(v)
    
    line.add("评论长度百分比", axis0, axis1)        
    time_stamp = time.strftime("_%Y-%m-%d_%H:%M:%S", time.localtime())
    file_name = "statistics_for_num_percentage" + time_stamp    
    line.render('../doc/{}'.format(file_name) + '.html')  

if __name__ == "__main__":
    statistics_processor()