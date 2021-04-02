import pandas as pd
import os
import json
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import matplotlib.pyplot as plt
import seaborn as sns
import content_based as content_based
sns.set()


def load_data(path):
    """
    Load events from files and convert to dataframe.
    """
    print("Loading dataset...")
    map_lst = []
    for f in os.listdir(path):
        file_name=os.path.join(path,f)
        if os.path.isfile(file_name):
            with open(file_name, encoding = 'utf-8') as _f:
                for line in _f:
                    obj = json.loads(line.strip())
                    if not obj is None:
                        map_lst.append(obj)
    print("Done loading dataset")
    return pd.DataFrame(map_lst) 


def statistics(df):
    """
    Basic statistics based on loaded dataframe
    """
    total_num = df.shape[0]
    
    print("Total number of events(front page incl.): {}".format(total_num))
    df.sort_values(by=['userId', 'time'], ascending=True, inplace=True)
    df_ref = df[df['documentId'].notnull()]
    num_act = df_ref.shape[0]
    
    print("Total number of events(without front page): {}".format(num_act))
    num_docs = df_ref['documentId'].nunique()
    
    print("Total number of documents: {}".format(num_docs))
    print('Sparsity: {:4.3f}%'.format(float(num_act) / float(1000*num_docs) * 100))
    df_ref.drop_duplicates(subset=['userId', 'documentId'], inplace=True)
    print("Total number of events(drop duplicates): {}".format(df_ref.shape[0]))
    print('Sparsity (drop duplicates): {:4.3f}%'.format(float(df_ref.shape[0]) / float(1000*num_docs) * 100))
    
    user_df = df_ref.groupby(['userId']).size().reset_index(name='counts')
    print("Describe by user:")
    print(user_df.describe())


if __name__ == '__main__':
    df=load_data("dataset/active1000")
    
    ###### Get Statistics from dataset ############
    print("Basic statistics of the dataset...")
    statistics(df)
    
    # ###### Recommendations based on Collaborative Filtering (Matrix Factorization) #######
    # print("Recommendation based on MF...")
    # collaborative_filtering(df)
    
    # ###### Recommendations based on Content-based Method (Cosine Similarity) ############
    print("Recommendation based on content-based method 0")
    content_based.recommendation_method_0(df, k=20)
    print("Recommendation based on content-based method 1")
    content_based.recommendation_method_1(df, k=20)
    print("Recommendation based on content-based method 2")
    content_based.recommendation_method_2(df, k=20)
    
    
    
