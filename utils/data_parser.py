import re

import numpy as np
import pandas as pd


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


def load_data(task, num_classes):
    '''    
    Args:
        task (string): load data from specific task 
        num_classes (int): categories of labels, can either be 2 or 3                
    Returns:
        x (list of string): data
        y (list of integer): labels
    '''
    x, y = [], []

    if task == "Twitter_Airlines":
        file_name = "../data/Tweets.csv"
        df = pd.read_csv(file_name)

        if num_classes == 2:
            df = df[df["airline_sentiment"] !=
                    "neutral"][["airline_sentiment", "text"]]
            df = df.dropna(axis=0, how="any")

            reviews = df['text'].tolist()
            labels = df['airline_sentiment'].tolist()
            x = [clean_str(review) for review in reviews]
            for label in labels:
                y.append([0, 1] if label == "negative" else [1, 0])

        elif num_classes == 3:
            df = df[["airline_sentiment", "text"]]
            df = df.dropna(axis=0, how="any")
            reviews = df['text'].tolist()
            labels = df['airline_sentiment'].tolist()
            x = [clean_str(review) for review in reviews]
            for label in labels:
                if label == "neutral":
                    y.append([0, 1, 0])
                elif label == "positive":
                    y.append([1, 0, 0])
                else:
                    y.append([0, 0, 1])

    elif task == "Amazon_Unlocked_Mobile":
        file_name = '../data/Amazon_Unlocked_Mobile.csv'
        df = pd.read_csv(file_name)

        if num_classes == 2:
            df = df[df['Rating'] != 3][['Reviews', 'Rating']]
            df = df.dropna(axis=0, how="any")

            reviews = df["Reviews"].tolist()
            labels = df["Rating"].tolist()
            x = [clean_str(review) for review in reviews]
            for label in labels:
                y.append([0, 1] if label == 1 or label == 2 else [1, 0])

        elif num_classes == 3:
            df = df[["Reviews", "Rating"]]
            df = df.dropna(axis=0, how="any")

            reviews = df["Reviews"].tolist()
            labels = df["Rating"].tolist()

            x = [clean_str(review) for review in reviews]
            for label in labels:
                if label == 1 or label == 2:
                    y.append([0, 0, 1])
                if label == 3:
                    y.append([0, 1, 0])
                if label == 4 or label == 5:
                    y.append([1, 0, 0])

    else:
        raise ValueError("task must be 'Twitter_Airlines' or "
                         "'Amazon_Unlocked_Mobile'. Got {} instead".format(task))

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
