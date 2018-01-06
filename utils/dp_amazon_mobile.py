import os
import sqlite3

import numpy as np
import pandas as pd
from tensorflow.contrib import learn


def convert_to_sqlite(file_in, sql_path):
    '''
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

    for _, row in df.iterrows():
        print(type(row['Reviews']))  # str
        break

# 针对数据预处理，进行learn.preprocessing.VocabularyProcessor
# 将词语转换成一个词的id
# 生成训练数据
# 使用Embedding_layer


def statistics_max_length():
    '''
    查到最大的文档长度    
    '''
    processor = learn.preprocessing.VocabularyProcessor
    # tf.keras.preprocessing.text.text_to_word_sequence()

    pass


def batch_iter(batch_size):
    '''
    Args:
        batch_size (int) stored in yaml
    Returns:
        X ():
        Y ():
    '''
    x = []
    y = []
    df = pd.read_csv()
    for _, row in df.iterrows():
        '''
        '''
        yield x, y


#from tensorflow.contrib import learn
learn.preprocessing.VocabularyProcessor
learn.preprocessing.VocabularyProcessor
# 在loss中增加对于类别不平衡的处理（同时）详看之前看故的论文
if __name__ == '__main__':
    statistics_amazon()
    pass
