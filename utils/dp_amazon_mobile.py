import pandas as pd
import numpy as np
import sqlites
import os


def convert_to_sqlite(file_in, sql_path):
    '''
    将处理好的原始数据以数据库文件形式存放
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




    