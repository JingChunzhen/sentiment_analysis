import re
import pandas as pd
import pyecharts


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
    """
    deprecated
    """
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


def find_optimum_length(task):
    """
    draw a review length and percentage line diagram
    get the optimum max_document_length for specific task   
    """
    if task is "Twitter_Airlines":
        df = pd.read_csv("../data/Tweets.csv")
        total_num = df["text"].count()    
        reviews = df["text"].tolist()
    elif task is "Amazon_Unlocked_Mobile":
        df = pd.read_csv("../data/Amazon_Unlocked_Mobile.csv")
        df = df.dropna(axis=0, how="any")
        total_num = df["Reviews"].count()
        reviews = df["Reviews"].tolist()
        
    reviews = [clean_str(review).split(' ') for review in reviews]
    lengths = [len(review) for review in reviews]
    review_nums = {}

    for l in lengths:
        if l in review_nums:
            review_nums[l] += 1
        else:
            review_nums[l] = 1

    sorted(review_nums.items(), key=lambda d: d[0])

    c, x, y = 0, [], []
    for k, v in review_nums.items():
        c += v
        percentage = c * 1.0 / total_num
        # total num indicate how many users
        x.append(k)
        y.append(percentage)

    from pyecharts import Line
    line = Line("")

    line.add("评论长度百分比", x, y) #, xaxis_interval=0, yaxis_interval=0)
    line.render("../doc/{}.html".format(task))


if __name__ == "__main__":
    find_optimum_length("Amazon_Unlocked_Mobile")
    pass
