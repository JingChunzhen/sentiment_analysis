import tensorflow as tf
from tensorflow.contrib import learn

import numpy as np
import pandas as pd

file_in = "../data/Amazon_Unlocked_Mobile.csv"
df = pd.read_csv(file_in)[:10]
texts = df['Reviews'].tolist()

print(type(texts))
print(len(texts))
# print(texts)

max_document_length = max([len(text.split(' ')) for text in texts])

processor = learn.preprocessing.VocabularyProcessor(max_document_length)

res = processor.fit_transform(texts)

print(np.shape(res))
for text in res:
    print(text) # get list of word ids

print("processor")
docs = processor.reverse(res)
# reverse
print(type(docs))
for doc in docs:
    # processor restore first 
    print('doc')
    print(doc) 










