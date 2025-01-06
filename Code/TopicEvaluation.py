import pandas as pd
import numpy as np
import os
from glob import glob
import ast
from itertools import combinations
import gensim.downloader as api
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, load_npz

DATE = '1121'
MODELS_DIR = '/Users/natal/Desktop/Disaster Analysis/Codes/Saved_models/'
DATA_DIR = '/Users/natal/Desktop/Disaster Analysis/Codes/Output/{}/Kws/'.format(DATE)
OUT =  '/Users/natal/Desktop/Disaster Analysis/Codes/Output/{}/'.format(DATE)
if not os.path.exists(OUT):  os.makedirs(OUT)

OUT_IMG_DIR = OUT + 'Imgs/'
if not os.path.exists(OUT_IMG_DIR):  os.makedirs(OUT_IMG_DIR)

OUT_SCORE = OUT + 'Score/'
if not os.path.exists(OUT_SCORE):  os.makedirs(OUT_SCORE)
print(os.getcwd())

def diversity_score(topic_kws):
    """
    Parameters
    ----------
    model_output : dictionary, output of the model key 'topics' required.
    Returns score
    """
    if topic_kws is None:
        return 0
    # if self.topk > len(topic_kws[0]):
    #     raise Exception('Words in topics are less than ' + str(self.topk))
    else:
        unique_words = set()
        for topic in topic_kws:
            unique_words = unique_words.union(set(topic))
        td = len(unique_words) / (len(topic_kws[0]) * len(topic_kws)) # score - number of unique words/(number of kws in each topic * number of topics)
        return td

        """
        Initialize metric WE pairwise similarity

        Parameters
        ----------
        :param topk: top k words on which the topic diversity will be computed
        :param word2vec_path: word embedding space in gensim word2vec format
        :param binary: If True, indicates whether the data is in binary word2vec format.
        """
        # super().__init__()
        # if word2vec_path is None:
        #     self.wv = api.load('word2vec-google-news-300')
        # else:
        #     self.wv = KeyedVectors.load_word2vec_format( word2vec_path, binary=binary)
def similarity_score(topic_kws, wv): # WordEmbeddingsPairwiseSimilarity
    #topics = model_output['topics']
    count = 0
    sum_sim = 0
    for list1, list2 in combinations(topic_kws, 2):
        word_counts = 0
        sim = 0
        for word1 in list1:
            for word2 in list2:
                if word1 in wv.key_to_index.keys() and word2 in wv.key_to_index.keys():
                    sim = sim + wv.similarity(word1, word2)
                    word_counts = word_counts + 1
        sim = sim / word_counts
        sum_sim = sum_sim + sim
        count = count + 1
    return sum_sim / count


def embedding_similarity_score(topic_embeddings): # WordEmbeddingsPairwiseSimilarity
    count = 0
    sim = 0
    for list1, list2 in combinations(topic_embeddings, 2):
        #list1 = list1.reshape(1, -1)
        #list2 = list2.reshape(1, -1)
        #sim = sim + np.dot(list1, list2)/(np.linalg.norm(list1)*np.linalg.norm(list2))
        sim = sim + cosine_similarity(list1, list2)
        count = count + 1
    print('{} combinations'.format(count))
    return sim / count

def convert_str_to_list(df, col):
    for index in df[col].index:
        df.at[index, col] = ast.literal_eval(df.loc[index, col]) # ast literal_eval method traverse thru str and parse a str as it value. ex: " "['a', 'b', 'c']" " will be intepreted as a list of str
    return df

def _replace_zeros_lines(arr):
    print('Arr shape', arr.shape, 'Len of arr: ', arr.size,'Arr is empty? ', arr.any() is False)
    zero_lines = np.where(arr.any(axis=1) is False)[0] # get zero lines
    val = 1.0 / len(arr[0])
    vett = np.full(len(arr[0]), val)
    for zero_line in zero_lines:
        arr[zero_line] = vett.copy()
    return arr

def _KL(P, Q):
    """ Perform Kullback-Leibler divergence
    Parameters
    ----------
    P : distribution P,    Q : distribution Q
    Returns
    -------
    divergence : divergence from Q to P"""
    # add epsilon to grant absolute continuity
    epsilon = 0.00001
    P = P+epsilon
    Q = Q+epsilon
    divergence = np.sum(P*np.log(P/Q))
    return divergence
""" Calculate the significance score for a list of topic-word matrices and the average significance score"""
def significance_score_for_multi_topics(topic_word_matrix_list):
    significance_scores = []
    count = 0
    for matrix in topic_word_matrix_list:
        avg_significance, significance_scores_per_topic = calculate_significance_score(matrix)
        significance_scores.append(significance_scores_per_topic)
        print('Topic {} - Significance score: {}'.format(count, significance_scores_per_topic))
        print('Average significance score: {}'.format(avg_significance))
        count += 1
    avg = np.mean(sum(significance_scores, []))
    return significance_scores, avg

""" Calculate the significance score for a single topic-word matrix"""
def calculate_significance_score(topic_word_matrix):
    """Retrieves the score of the metric
    Parameters
    ----------
    model_output : dictionary, output of the model 'topic-word-matrix' required
    per_topic: if True, it returns the score for each topic
    Returns
    -------
    result : score"""
    phi = _replace_zeros_lines(topic_word_matrix) # .astype(float)
    # make uniform distribution
    val = 1.0 / len(phi[0])
    unif_distr = np.full(len(phi[0]), val)
    divergences = []
    for topic in range(len(phi)):
        # normalize phi, sum up to 1
        P = phi[topic] / phi[topic].sum()
        divergence = _KL(P, unif_distr)
        divergences.append(divergence)
    # KL-uniform = mean of the divergences between topic-word distributions and uniform distribution

    result = np.array(divergences).mean()
    return result, divergences


def get_topic_embeddings(modelname):
    """Parameters
    ----------
    modelname : model name
    Returns a df with Topic-Subtopic numbers and their embeddings (Each topic-subtopic has 1 vector of embedding) and a list of topic-term matrices"""

    indir = MODELS_DIR + modelname + '/'
    topic_term_matrices = []

    if 'Topic' in ' '.join(os.listdir(indir)): #Look for substring 'Topic' in the list of files in the directory
        indir = indir + 'Topic-*/'
        df = pd.DataFrame(columns=['Topic', 'Subtopic', 'Name', 'No. Docs', 'Topic_Embeddings'])
        # iterate over all files with the format Topic-* in the directory
        topic_list = []
        # get a list of topics from the directory and sort them
        for filedir in glob(indir):
            topic = int(re.sub('[^a-zA-Z0-9 \n\.]', "/" , filedir).split('/')[-2])
            topic_list.append(topic)
            print('Topic: ', topic, ' Type: ', type(topic))
        topic_list.sort()
        print('Sorted topic list: ', topic_list)
        # Iterate through each topic in order and get its embeddings, subtopics and topic-term matrix
        for topic in topic_list:
            filedir = indir.replace('*', str(topic))
            file = filedir + 'topic_embeddings.npy'
            print('File: ', file)
            if '~$' in file: continue  # Ignore temp files
            one_topic_embedding = np.load(file)
            # topic_embeddings.append(one_topic_embedding)
            print('Len of one topic embedding: ', one_topic_embedding.shape)

            # Get subtopics
            subtopics_file = open(filedir +'topics.json', 'r')
            subtopics = json.load(subtopics_file)
            subtopics_file.close()
            print('Topic {} has subtopics: {} '.format(topic, subtopics['topic_labels'].keys()))

            # Get topic-term matrix
            topic_term_matrix = load_npz(filedir + 'topic_term_matrix.npz')
            topic_term_matrices.append(topic_term_matrix.toarray()) # a list of topic-term matrices. Each topic has a topic-term matrix generated from the subtopics. The number of topic-term matrices should be the same with the number of topics
            print('Topic {} has topic term matrix: {} '.format(topic, topic_term_matrix.shape))
            print('Len subtopics: {}, len topic_embeddings: {}'.format(len(subtopics['topic_labels'].keys()), one_topic_embedding.shape))
            topic_sizes = {int(k):int(v) for k,v in subtopics['topic_sizes'].items()}  # convert keys and values to int
            topic_sizes =  dict(sorted(topic_sizes.items())) # sort by keys
            print('Topic sizes: ', topic_sizes)
            df = pd.concat([df, pd.DataFrame({'Topic': topic, 'Subtopic': subtopics['topic_labels'].keys(), 'Name': subtopics['topic_labels'].values(), 'No. Docs': topic_sizes.values(),
                                              'Topic_Embeddings': np.split(one_topic_embedding, one_topic_embedding.shape[0], axis =0)})]) # split topic embedding by rows
    else:
        print('No Topic dir found')
        file = indir + 'topic_embeddings.npy'
        print('File: ', file)
        one_topic_embedding = np.load(file)
        print('Len of one topic embedding: ', len(one_topic_embedding))

        # Get topics
        topics_file = open(indir + 'topics.json', 'r')
        topics = json.load(topics_file)
        topics_file.close()
        print('Topics: {} '.format(topics['topic_labels'].keys()))
        topic_sizes = {int(k): int(v) for k, v in topics['topic_sizes'].items()}  # convert keys and values to int
        topic_sizes = dict(sorted(topic_sizes.items()))  # sort by keys
        print('Topic sizes: ', topic_sizes)

        # Get topic-term matrix
        topic_term_matrix = load_npz(indir + 'topic_term_matrix.npz')
        topic_term_matrices.append(topic_term_matrix.toarray())
        print('Topic term matrix shape: {} '.format(topic_term_matrix.shape))
        print('Type of topic term matrix: {} , Size: {}'.format(type(topic_term_matrix), topic_term_matrix.size))
        print('Topic term matrix: {} '.format(topic_term_matrix))
        df = pd.DataFrame({'Topic': topics['topic_labels'].keys(), 'Name': topics['topic_labels'].values(), 'No. Docs': topic_sizes.values(),
                           'Topic_Embeddings': np.split(one_topic_embedding, one_topic_embedding.shape[0],axis=0)})  # split topic embedding by rows

    return df, topic_term_matrices


if __name__ == '__main__':
    file_list = glob(DATA_DIR+ '*.xlsx')
    print('List of files to evaluate: ', file_list)
    wv = api.load('word2vec-google-news-300')
    #file_list = ['/Users/natal/Desktop/Disaster Analysis/Codes/Output/1121/Kws\\Bertopic-kw-coherence-Post-1121.xlsx']
    for file in file_list:
        print('File: ', file)
        modelname = file.replace("\\", '/').split('/')[-1].split('-')[0]
        print('Model name: ', modelname)
        if '~$' in file:
            continue

        df = pd.read_excel(file, header=0 )
        df.dropna(axis=0, inplace= True)
        df = convert_str_to_list(df, 'Kws')
        print('Calculating diversity score...')
        diversity = diversity_score(df['Kws'].tolist())
        print('Diversity score: ', diversity)
        df['Diversity'] = diversity

        print('Calculating Topic-KW similarity score...')
        similarity = similarity_score(df['Kws'].tolist(), wv)
        print('Similarity score: ', similarity)
        df['KW_Similarity'] = similarity

        print('Calculating Topic-Embedding similarity score...')
        # Get topic embeddings and return them in a df with topic and subtopic number         # Save topic embeddings to df with topic and subtopic number
        topic_df, topic_term_matrices = get_topic_embeddings(modelname)
        print('Topic {}: {} \n shape of topic term matrixes: {}'.format(modelname, topic_df, len(topic_term_matrices)))
        topic_similarity = embedding_similarity_score(topic_df['Topic_Embeddings'].to_numpy())
        print('Topic similarity: ', topic_similarity, topic_similarity[0][0])
        df['Topic_Embedding_Similarity'] = topic_similarity[0][0] # because topic similarity is a list of list
        #Insert columns: topic names and sizes to df
        df.insert(2, 'Topic_Name', topic_df['Name'].tolist())
        df.insert(3, 'No. Docs', topic_df['No. Docs'].tolist())

        # Calculate avg topic significance score and identify the topics with high significance score
        print('Calculating Topic significance score...')
        sigfinicance_scores, avg_significance = significance_score_for_multi_topics(topic_term_matrices)
        print('Avg significance score across {} topics: {}'.format(len(sigfinicance_scores),avg_significance) ,'Topic significance score: ', sigfinicance_scores)
        if len(sigfinicance_scores) == 1:
            df['Topic_Significance_Score'] = sigfinicance_scores[0]
        else:
            df['Topic_Significance_Score'] = sum(sigfinicance_scores, [])
        # Identify topics and subtopic with high significance score
        print('df: ', df.tail(), 'df columns: ', df.columns)

        print('Saving df to dir: ', OUT+modelname+'_eval.xlsx')
        # Save df to excel
        df.to_excel(OUT+modelname+'_eval_{}.xlsx'.format(DATE), index=False)