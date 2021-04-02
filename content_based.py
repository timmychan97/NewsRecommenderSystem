import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


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
    

def content_processing_category(df):
    """
    Remove events which are front page events, and calculate cosine similarities between
    items. Here cosine similarity are only based on item category information, others such
    as title and text can also be used.
    Feature selection part is based on TF-IDF process.
    """
    # NOTE: This is the preexisting method in the project example. 
    # We kept the code to show the difference in evaluation results

    # Problems with the code:
    #  - df contain multiple rows with the same documentId 
    #    BUT some with more category info than some other.
    #    randomly dropping duplicates is not a dependable solution
    #  - total number of documents is 20344 but number of rows in
    #    df_items is 20393. It does not match, as there are still
    #    duplicated documentIds in df

    df = df[df['documentId'].notnull()].copy()
    df.drop_duplicates(subset=['userId', 'documentId'], inplace=True)
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
    
    print("Similarity Matrix:")
    print(cosine_sim[:4, :4])
    return cosine_sim, df


def recommendation_method_0(df, k=20):
    """
    Generate top-k list according to cosine similarity
    
    Result:
        Recall@20 is 0.0070
        ARHR@20 is 0.0006
    """
    # NOTE: This is the preexisting method in the project example. 
    # We kept the code to show the difference in evaluation results

    # Problems with the code:
    #  - Loops only 999 times. The last user in the df does not get tested against

    cosine_sim, df = content_processing_category(df)
    df = df[['userId','time', 'tid', 'title', 'category']]
    df.sort_values(by=['userId', 'time'], ascending=True, inplace=True)
    pred, actual = [], []

    puid, ptid1, ptid2 = None, None, None
    for row in df.itertuples():
        uid, tid = row[1], row[3]
        if uid != puid and puid != None:
            idx = ptid1
            recommends = get_similar_docs(cosine_sim[idx], k)
            pred.append(recommends)
            actual.append(ptid2)
            puid, ptid1, ptid2 = uid, tid, tid
        else:
            ptid1 = ptid2
            ptid2 = tid
            puid = uid
    evaluate(pred, actual, k)


#-----------------
# Our methods
#-----------------

def get_similar_docs(list_similar, n=20):
    """
    Return the tid of n similar docs by a list of similar docs
    """
    sim_scores = list(enumerate(list_similar))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Skip the first. We don't want to recommend the same doc again
    sim_scores = sim_scores[1:n+1]
    return [i for i,j in sim_scores]


def content_processing_category_and_title(df):
    """
        Remove events which are front page events, and calculate cosine similarities between
        items. Here cosine similarity are only based on item category information, others such
        as title and text can also be used.
        Feature selection part is based on TF-IDF process.
    """
    # make copy and remove invalid entries
    df = df[df['documentId'].notnull()].copy()

    # Remove additional events a user acted upon the same document
    df.sort_values(by='category', key=lambda x: x.str.len())
    df.drop_duplicates(subset=['userId', 'documentId'], inplace=True, keep='last')
    print("shape of df here:", df.shape)

    # convert all null category to str
    # df['category'] = df['category'].fillna("").astype('str')
    # df['category'] = df['category'].str.split('|') + df["title"].str.split(' ')
    # df['category'] = df['category'].fillna("").astype('str')
    df['category'] = df['category'].str.split('|')
    df['category'] = df['category'].fillna("").astype('str')
    
    item_ids = df['documentId'].unique().tolist()
    print("item_ids length:", len(item_ids))
    new_df = pd.DataFrame({'documentId':item_ids, 'tid':range(1,len(item_ids)+1)})
    df = pd.merge(df, new_df, on='documentId', how='outer')

    df_item = df[['tid', 'category']].drop_duplicates(inplace=False)
    #df_item.to_csv('df_item.csv', sep=',')
    print("df item shape:", df_item.shape)
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


def recommendation(df, k=20):
    """
        Generate top-k list according to cosine similarity
    """
    cosine_sim, df, tfidf_matrix = content_processing_method_0(df)
    print("size of cosine-sim: ", cosine_sim.shape)
    #np.savetxt("cosine-sim.csv", cosine_sim, delimiter=",")

    df = df[['userId','time', 'tid', 'title', 'category']]
    df.sort_values(by=['userId', 'time'], ascending=True, inplace=True)
    print(df[:20]) # see how the dataset looks like
    pred, actual = [], []
    puid, ptid1, ptid2 = None, None, None

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
    evaluate(pred, actual, k)


def recommendation_method_1(df, k=20):
    """
    Loop through all events in dataset chronologically for each user
    for the last event of the user, we try to evaluate our recommendation method
    against the clicked news article.

    The method: Recommend the k articles most similar to the most recent viewed news.
    
    Result:
        Recall@20 is 0.0070
        ARHR@20 is 0.0006
    """
    cosine_sim, df = content_processing_category(df)
    df = df[['userId','time', 'tid', 'title', 'category']]
    df.sort_values(by=['userId', 'time'], ascending=True, inplace=True, ignore_index=True)
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
        recommends = get_similar_docs(cosine_sim[prev_doc], k)
        pred.append(recommends)
        actual.append(doc)
    evaluate(pred, actual, k)


def recommendation_method_2(df, k=20):
    """
    Loop through all events in dataset chronologically for each user
    for the last event of the user, we try to evaluate our recommendation method
    against the clicked news article.

    The method: 
     - Recommend the k articles most similar to the 4 most recent viewed news
    
    Result:
        Recall@20 is 0.0020
        ARHR@20 is 0.0001
    """
    cosine_sim, df = content_processing_category(df)
    df = df[['userId','time', 'tid', 'title', 'category']]
    df.sort_values(by=['userId', 'time'], ascending=True, inplace=True, ignore_index=True)
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
        recommends = []
        for prev_event in range(4):
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
    evaluate(pred, actual, k)