import pandas as pd
import os
import json
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import matplotlib.pyplot as plt
import seaborn as sns
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


def evaluate(pred, actual, k):
    """
    Evaluate recommendations according to recall@k and ARHR@k
    """
    total_num = len(actual)
    tp = 0.
    arhr = 0.
    for p, t in zip(pred, actual):
        if t in p:
            tp += 1.
            arhr += 1./float(p.index(t) + 1.)
    recall = tp / float(total_num)
    arhr = arhr / len(actual)
    print("Recall@{} is {:.4f}".format(k, recall))
    print("ARHR@{} is {:.4f}".format(k, arhr))


def content_processing(df):
    """
        Remove events which are front page events, and calculate cosine similarities between
        items. Here cosine similarity are only based on item category information, others such
        as title and text can also be used.
        Feature selection part is based on TF-IDF process.
    """
    # make copy and remove invalid entries
    df = df[df['documentId'].notnull()]
    df.drop_duplicates(subset=['userId', 'documentId'], inplace=True)

    # convert all null category to str
    # df['category'] = df['category'].fillna("").astype('str')
    # df['category'] = df['category'].str.split('|') + df["title"].str.split(' ')
    # df['category'] = df['category'].fillna("").astype('str')
    df['category'] = df['category'].str.split('|')
    df['category'] = df['category'].fillna("").astype('str')
    
    item_ids = df['documentId'].unique().tolist()
    new_df = pd.DataFrame({'documentId':item_ids, 'tid':range(1,len(item_ids)+1)})
    df = pd.merge(df, new_df, on='documentId', how='outer')
    df_item = df[['tid', 'category']].drop_duplicates(inplace=False)
    df_item.sort_values(by=['tid', 'category'], ascending=True, inplace=True)
    
    # select features/words using TF-IDF 
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0)
    tfidf_matrix = tf.fit_transform(df_item['category'])
    print('Dimension of feature vector: {}'.format(tfidf_matrix.shape))
    # measure similarity of two articles with cosine similarity
    
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    #np.savetxt("tfidf-sim.csv", tfidf_matrix.toarray(), delimiter=",")
    print("Similarity Matrix:")
    print(cosine_sim[:4, :4])
    return cosine_sim, df, tfidf_matrix

def content_recommendation(df, k=20):
    """
        Generate top-k list according to cosine similarity
    """
    cosine_sim, df, tfidf_matrix = content_processing(df)
    print("size of cosine-sim: ", cosine_sim.shape)
    #np.savetxt("cosine-sim.csv", cosine_sim, delimiter=",")

    #df[:500].to_csv('df.csv', sep=',')
    df = df[['userId','time', 'tid', 'title', 'category']]
    df.sort_values(by=['userId', 'time'], ascending=True, inplace=True)
    #df.to_csv('output2.csv', sep=',')
    #df[:1000].to_csv('df2.csv', sep=',')
    print(df[:20]) # see how the dataset looks like
    pred, actual = [], []
    puid, ptid1, ptid2 = None, None, None
    # for row in df.itertuples():
    #     uid, tid = row[1], row[3]
    #     if uid != puid and puid != None:
    #         idx = ptid1
    #         sim_scores = list(enumerate(cosine_sim[idx]))
    #         sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    #         sim_scores = sim_scores[1:k+1]
    #         sim_scores = [i for i,j in sim_scores]
    #         pred.append(sim_scores)
    #         actual.append(ptid2)
    #         puid, ptid1, ptid2 = uid, tid, tid
    #     else:
    #         ptid1 = ptid2
    #         ptid2 = tid
    #         puid = uid

    # find the last event of each user
    last_events, prev_user = [], df['userId'].iloc[0]
    for i, row in df.iterrows():
        user = row['userId']
        if user != prev_user:
            last_events.append(i - 1)
            prev_user = user
    last_events.append(df.shape[0] - 1)
    print(len(last_events))
    # loop the last events and check if recommendations matches the events
    user_preferred_docs = []
    count = 0
    for idx in last_events:
        count += 1
        recommends = []
        for prev_event in range(2):
            doc_id = df['tid'].iloc[idx - 1 - prev_event]
            sim_scores = list(enumerate(cosine_sim[doc_id]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:k+1]  # skip the first one, as we don't want to recommend the same doc again
            recommends.extend(sim_scores)
        # sort and remove duplicate recommendations, and take k items
        recommends = sorted(recommends, key=lambda x: x[1], reverse=True)
        recommends = [i for i,j in recommends]
        recommends = list(set(recommends))[:k]
        pred.append(recommends)
        actual.append(df['tid'].iloc[idx])
        if count % 100 == 0:
            print(count)
    # print("PRED ----------- ")
    # print(pred)
    # print("^^^^^^^^^^")
    # print("ACTUAL ----------- ")
    # print(actual)
    # print("^^^^^^^^^^")
    evaluate(pred, actual, k)

def content_recommendation_method_1(df, k=20):
    """
    Loop through all events in dataset chronologically for each user
    for the last event of the user, we try to evaluate our recommendation method
    against the clicked news article.

    The method: Recommend the k articles most similar to the most recent viewed news. 
    """
    # NOTE: preexisting code does not evaluate the last user. Essentially a bug fixed in this method
    cosine_sim, df, tfidf_matrix = content_processing(df)
    print("size of cosine-sim: ", cosine_sim.shape)

    df = df[['userId','time', 'tid', 'title', 'category']]
    df.sort_values(by=['userId', 'time'], ascending=True, inplace=True)
    print(df[:20]) # see how the dataset looks like
    pred, actual = [], []


    # find the last event of each user
    last_events, prev_user = [], df['userId'].iloc[0]
    for i, row in df.iterrows():
        user = row['userId']
        if user != prev_user:
            last_events.append(i - 1)
            prev_user = user
    last_events.append(df.shape[0] - 1)

    # loop the last events and check if recommendations matches the events
    for idx in last_events:
        doc = df['tid'].iloc[idx]
        prev_doc = df['tid'].iloc[idx - 1]
        sim_scores = list(enumerate(cosine_sim[prev_doc]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # skip the first one, as we don't want to recommend the same doc again
        sim_scores = sim_scores[1:k+1]
        recommends = [i for i,j in sim_scores]
        pred.append(recommends)
        actual.append(doc)

    evaluate(pred, actual, k)



if __name__ == '__main__':
    df=load_data("dataset/active1000")
    
    ###### Get Statistics from dataset ############
    print("Basic statistics of the dataset...")
    statistics(df)
    
    # ###### Recommendations based on Collaborative Filtering (Matrix Factorization) #######
    # print("Recommendation based on MF...")
    # collaborative_filtering(df)
    
    # ###### Recommendations based on Content-based Method (Cosine Similarity) ############
    print("Recommendation based on content-based method...")
    #content_recommendation(df, k=50)
    print("Recommendation based on content-based method 1")
    content_recommendation_method_1(df, k=20)
    
    
    
