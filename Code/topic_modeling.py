#import bertopic
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud
import pandas as pd
from matplotlib import pyplot as plt
import json
import Process_Texts
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import Doc2Vec
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
import gc
import Based_BERTopic

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity

import os
import sys
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Train BERTopic func
def tr_BERTopic(texts, topic_list=None, reduced_topics=None, top_kws=20, diverse=0.3,
                   min_topic_sz=10):  # , min_topic_sz
    """Details of parameter tunning: https://maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html#n_gram_range"""

    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=True)  # use this model to reduce the impact of frequent w
    diverse = MaximalMarginalRelevance(diversity=diverse)  # increase kw diversity by increasing param diversity
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")  # Use pre-trained model as an embedding model
    embeddings = sentence_model.encode(texts, show_progress_bar=True)

    model = BERTopic(ctfidf_model=ctfidf_model, representation_model=diverse, top_n_words=top_kws,
                     seed_topic_list=topic_list, \
                     min_topic_size=min_topic_sz, n_gram_range=(1, 3), nr_topics=reduced_topics, verbose=True)
    # Train BERTopic
    topic_model = model.fit(texts, embeddings)

    return topic_model, embeddings


# Train BERTopic func
def train_BERTopic(texts, stopwords=list(Process_Texts.all_stopwords), topic_list=None, reduced_topics="auto", top_kws=20, diverse=0.3,
                   min_topic_sz=10):  # , min_topic_sz
    """Details of parameter tunning: https://maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html#n_gram_range"""

    # Use pre-trained model to extract embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)  # Save embeddings from input

    # Reduce dimentionality
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state= SEED)

    # Cluster embeddings
    hdbscan_model = HDBSCAN(min_cluster_size=min_topic_sz, metric='euclidean', cluster_selection_method='eom',
                            prediction_data=True)

    # Tokenize topics
    vectorizer_model = CountVectorizer(stop_words=stopwords)

    # Create topic representation, use this model to reduce the impact of frequent w
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=False)

    # Fine-tune topic representations with KeyBERT model
    representation_model = KeyBERTInspired(top_n_words=top_kws)

    # Increase kw diversity by increasing param diversity
    # diverse = MaximalMarginalRelevance(diversity= diverse)

    model = BERTopic(embedding_model=embedding_model,  # Embed docs into vec representation
                     umap_model=umap_model,  # Dimensionality reduction
                     hdbscan_model=hdbscan_model,
                     # Clustering alg, which does not force docs into a cluster that they don't belong, but treat them as outlier instead
                     vectorizer_model=vectorizer_model,  # Convert docs in each cluster into BOW
                     ctfidf_model=ctfidf_model,
                     # Extract important words in each cluster that describe/represent the topic using c-TF-IDF
                     representation_model=representation_model,
                     # (or use diverse). Use KeyBERT to fine-tune c-TF-IDF topic representation
                     top_n_words=top_kws,  # Number of kws for topic representation
                     seed_topic_list=topic_list,
                     min_topic_size=min_topic_sz,
                     n_gram_range=(1, 3),
                     nr_topics= "auto", verbose=True) #reduced_topics
    # Train BERTopic     topic_model = model.fit(texts, embeddings)
    topics, probs = model.fit_transform(texts, embeddings)

    print('Before reducing outliers----Topics: ', set(topics))
    # # Reduce topic automatically during sub-topic extraction if the number of subtopics is at least 3
    # if reduced_topics == None and len(set(topics)) > 3:
    #     model = BERTopic(nr_topics="auto")
    #     topics, probs = model.fit_transform(texts, embeddings)
    #     print('After reducing number of topics: ', set(topics))

    # try fit_transform and new_topics = model.reduce_outliers(docs, topics, strategy='embeddings'), then update topic representation model.update_topics(docs, topics=new_topics)
    if set(topics) != {-1}:
        new_topics = model.reduce_outliers(texts, topics, strategy='embeddings', embeddings= embeddings)
        model.update_topics(texts, topics=new_topics)
        print('After reducing outliers----Topics: ', set(new_topics)); print('shape of Probs: {} Probs: {}'.format(probs.shape , probs))
    else:
        print('Cannot reduce outliers as there is only outlier topic')
    return model, embeddings

def plot_topic_dist(model, name):
    topics = model.get_topic_info()
    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(topics['Name'], topics['Count'], width=0.4)
    for i, v in enumerate(topics['Count']):
        plt.text(i, v + 0.5, str(v), ha='center', fontsize=8)

    plt.xlabel("Topic Names")
    plt.ylabel("Number of Documents")
    bottom, top = plt.ylim()
    plt.ylim(top=top + 15)
    plt.xticks(rotation=90)
    plt.title("{}-Pandemic Topic Distribution".format(name))
    plt.tight_layout()
    plt.savefig(OUT_IMG_DIR + 'topic_dist_{}pandemic.jpg'.format(name), bbox_inches='tight', dpi=150)
    plt.show()

# Old create wc
def create_wordcloud0(model, period, name=''):
    for topic in model.get_topics():
        text = {word: value for word, value in model.get_topic(topic)}
        wc = WordCloud(background_color="white", max_words=1000)
        wc.generate_from_frequencies(text)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        if name != '':
            plt.title('{}-Pandemic Topic {} - Subtopic {} '.format(period, name, topic))
            plt.savefig(OUT_IMG_DIR + '{}-pandemic-topic{}-subtopic{}.jpg'.format(period, name, topic),
                        bbox_inches='tight', dpi=150)
        else:
            plt.title('{}-Pandemic Topic {} '.format(period, topic))
            plt.savefig(OUT_IMG_DIR + '{}-pandemic-topic{}.jpg'.format(period, topic), bbox_inches='tight', dpi=150)
        plt.show()

# topic name (if plotting subtopic kws), else (when plotting topic kws) name is '')
def create_wordcloud(model, model_name, period, topic_name=''):
    dir = OUT_IMG_DIR + '{}/'.format(model_name)
    if not os.path.exists(dir):  os.makedirs(dir)
    for topic in model.get_topics():
        text = {word: value for word, value in model.get_topic(topic)}
        print('Kws: ', text)
        wc = WordCloud(background_color="white", max_words=1000)
        wc.generate_from_frequencies(text)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        if topic_name != '':
          #  plt.title('{}-Pandemic Topic {} - Subtopic {} '.format(period, topic_name, topic))
            plt.savefig(dir +'{}-pandemic-topic{}-subtopic{}.jpg'.format(period, topic_name, topic),bbox_inches='tight', dpi=150)
        else:
          #  plt.title('{}-Pandemic Topic {} '.format(period, topic))
            plt.savefig(dir +'{}-pandemic-topic{}.jpg'.format(period, topic),bbox_inches='tight', dpi=150)
    #plt.show()

def create_wordcloud_from_freq(kw_freq, period, model_name, img_name, topic_name='', subtopic_name=''): #embedding
    dir = OUT_IMG_DIR + '{}/'.format(model_name)
    if not os.path.exists(dir):  os.makedirs(dir)
    wc = WordCloud(background_color="white", max_words=100)
    wc.generate_from_frequencies(kw_freq)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if topic_name != '':
        img_name = '{}-Pandemic Topic {} '.format(period, topic_name)
        if subtopic_name != '':
            img_name += ' - Subtopic {} '.format(subtopic_name)
    else:
        print('No topic name provided.')
    #plt.savefig(OUT_IMG_DIR+'{}-pandemic-topic{}.jpg'.format(period, topic),bbox_inches='tight', dpi=150)
    #plt.title(img_name + model_name)
    print('In create_wc_from_freq, img filename is: ', img_name + model_name) # +embedding)
    print('Kws: ', kw_freq.items())
    plt.savefig(dir + '{}-{}.jpg'.format(img_name, model_name), bbox_inches='tight', dpi=150)
   # plt.show()

# Get kw-freq dict from the topic model BERTopic and call func create_wordcloud to generate images of wordcloud
def create_wordcloud_from_topicmodel(model, model_name, period, file_name, topic_name=''):
  for topic in model.get_topics():
    text = {word: value for word, value in model.get_topic(topic)}
    create_wordcloud_from_freq(text, period, model_name= model_name, img_name = file_name, topic_name=topic_name, subtopic_name=topic)


# Plot Coherence per subtopic given x (indices in subtopic_kws df) and y (coherence scores) and the model name (e.g., ''BERTopic-BERTopic')
def plot_coherence0(x, y, label):
    #plt.figure(figsize=())
    plt.bar(x, y, label= label)
    plt.xlabel('Subtopics')
    plt.ylabel('Coherence Score')
    plt.legend()
    plt.title('{} Coherence Score'.format(label))
    plt.tight_layout()
    plt.savefig(OUT_IMG_DIR+'{} Coherence Score.jpg'.format(label), bbox_inches='tight', dpi=150)
    plt.show()

# Plot Coherence per subtopic given x (indices in subtopic_kws df) and y (coherence scores) and the model name (e.g., ''BERTopic-BERTopic')
def plot_coherence(df, name):
    #plt.show()
    #plt.figure(figsize=())
    x = np.arange(0,len(df))
    xlabel = '(Topic, Subtopic)'
    if name=='BERTopic':
      xlabel = 'Subtopic'
    plt.bar(x, df['Coherence'])
    #add horizontal line at mean value of y
    plt.axhline(y=np.nanmean(df['Coherence']), linestyle='dashed', color='grey', label='Avg')
    plt.ylim(top = max(df['Coherence'])+0.1)
    plt.xlabel(xlabel)
    plt.xticks(x, df['Label'], size='small', rotation=45)
    plt.ylabel('Coherence Score')
    plt.legend()
    plt.title('{} Coherence Score'.format(name))
    plt.tight_layout()
    plt.savefig(OUT_IMG_DIR+'{} Coherence Score.jpg'.format(name), bbox_inches='tight', dpi=150)
    #plt.show()

# Get subtopic docs for BERTopic
# Params: data is df, topic is the topic number
def get_texts_from_topic(df, TOPIC):
  filt = df['Topic'] == TOPIC
  return df.loc[filt]['Corpus']

def write_json_to_file(topic, topic_dict, filename, dir):
  file = dir+ filename
  with open(file, "a") as outfile:
    outfile.write('\nTopic {}:\n'.format(topic))
    json.dump(topic_dict, outfile, indent = 4)

# Get subtopic docs for BERTopic
# Params: data is df, topic is the topic number
def get_texts_from_topic(df, TOPIC):
    filt = df['Topic'] == TOPIC
    return df.loc[filt]['Corpus']


def reset_file(dir, filename):
    file_to_delete = open(dir + filename, 'w')
    file_to_delete.close()

""" Params: model - BERTopic model, which has a function get_topics to get a list of topics,
topic_name - the topic number if we're extracting subtopic kws
Functionality: call func get_topics of BERTopic to get a dict of kw and freq of a topic,
and save the kws into subtopic_kw dict.
Return a subtopic-kw dict from a topic with subtopic is key and kw is word freq. """
def get_topic_kws(model, topic_name=''):
  subtopic_kws = {}
  for subtopic in model.get_topics():
    text = {word: value for word, value in model.get_topic(subtopic)}
    subtopic_kws[subtopic] = list(text.keys())
    print('from create wc from topic model: Topic {} - Subtopic {} - kws: {}'.format(topic_name, subtopic, subtopic_kws[subtopic]))
  return subtopic_kws


# Params: df contains Topic column. filename include period name. min_cluster is the min cluster sz
# Functionality: extract subtopics using BERTopic
def extract_subtopics(df, period, filename, min_cluster=3, embed = 'bert'):
    # Create dir to save images
    dir = OUT_IMG_DIR + '/BERTopic-{}'.format(embed)
    if not os.path.exists(dir):  os.makedirs(dir)

    subtopic_doclist = []
    subtopic_kws_df = pd.DataFrame(columns = ['Topic', 'Subtopic', 'Kws'])
    # Reset text files
    reset_file(OUT_KWS, '{}-kws.json'.format(filename))
    reset_file(OUT_KWS, '{}-titles.txt'.format(filename))
    list_of_probabilities = []
    print('Extracting subtopics using BERTopic-{}...'.format(embed))
    for topic in df['Topic'].value_counts().sort_index(ascending=True).index:
        docsbytopic = get_texts_from_topic(df, topic)
        print('Extracting subtopics from topic {} of {} docs...'.format(topic, len(docsbytopic)))
        if topic == -1:
            print('Topic {} is outlier, skip'.format(topic))
            continue
        elif len(docsbytopic) <= min_cluster:
            print('Topic {} has {} docs, less than min_cluster {}, skip'.format(topic, len(docsbytopic), min_cluster))
            continue

        #Bertopic, embeddings = train_BERTopic(docsbytopic.values.tolist(), reduced_topics =None, min_topic_sz=min_cluster)  # , topic_list=['disaster', 'health', 'economic']) remove limit min_topic_sz=10
        Bertopic, embeddings, probabilities = Based_BERTopic.train_BERTopic_w_diff_embeddings(docsbytopic.values.tolist(), embedding_name = embed,
                                        topic_list=None, reduced_topics="auto", top_kws=20, diverse=0.3, min_topic_sz= min_cluster, topic_number=topic, calc_prob = True)
        list_of_probabilities.append(probabilities)

        # save model
        #Bertopic.save(path='{}{}-model'.format(MODELS_DIR, embed))  # save_ctfidf=True, serialization = "pickle"

        print('Topic {} - Subtopics include: '.format(topic), Bertopic.generate_topic_labels())  # -1 means outliers that should be ignored
        # Get kw-freq per subtopic
        subtopic_kws = get_topic_kws(Bertopic, topic_name=topic)
        # Generate wordclouds for each subtopic
        create_wordcloud_from_topicmodel(Bertopic, 'BERTopic-{}'.format(embed), period, topic_name= topic, file_name= filename) #(model, period, file_name, topic_name=''

        # Save subtopic titles within each topic
        write_json_to_file(topic, Bertopic.generate_topic_labels(), '{}-titles.txt'.format(filename), OUT_KWS)
        # Save subtopic kws within each topic
        write_json_to_file(topic, str(Bertopic.get_topics()), '{}-kws.json'.format(filename), OUT_KWS)
        subtopic_kws_df = pd.concat([subtopic_kws_df, pd.DataFrame(
            {'Topic': topic, 'Subtopic': subtopic_kws.keys(), 'Kws': subtopic_kws.values()})])
        # Add topic embeddings of each Topic-Subtopic to df
        #subtopic_kws_df['Topic_Embeddings'] = [i for i in Bertopic.topic_embeddings_.tolist()]
        # Add topic-term matrix to df
        #subtopic_kws_df['Topic_Term_Matrix'] = [i for i in Bertopic.topic_term_matrix_.tolist()]

        # Mark docs with subtopic
        doctopicdf = Bertopic.get_document_info(docsbytopic.values.tolist())
        # Make sure the original index is kept from docsbytopic
        doctopicdf.set_index(docsbytopic.index.values, inplace=True, drop=True)
        subtopic_doclist.append(doctopicdf)
    return pd.concat(subtopic_doclist, ignore_index=False), subtopic_kws_df.reset_index(drop=True), list_of_probabilities


 # check for missing values Abstract in a df, and drop them if exists
def remove_missing_values(df):
    na_ind = df[df['Abstract'].isnull()].index.tolist()
    if len(na_ind) > 0:
        print('Remove {} rows missing abstract from df len {}'.format(len(na_ind), len(df)))
        df.drop(index=na_ind, axis=0, inplace=True)
        print('Len after remove: {}'.format(len(df)))
    return df

# Params: df contains Topic column. filename include period name. min_cluster is the min cluster sz
def extract_topics(df, period, filename, col_name, embed, min_cluster=2, num_kws = 20, calc_prob = True):
    file = OUT + filename
    # Reset text files
    reset_file(OUT, '{}-kws.json'.format(filename))
    reset_file(OUT, '{}-titles.txt'.format(filename))
    print('Extracting topics using BERTopic...')
    #    docsbytopic = get_texts_from_topic(df, topic)
    Bertopic, embeddings, probabilities = Based_BERTopic.train_BERTopic_w_diff_embeddings(df['Corpus'].values.tolist(), topic_list=None,
                                            embedding_name = embed, reduced_topics = "auto", min_topic_sz = min_cluster, topic_number=None,
                                            stopwords = list(Process_Texts.all_stopwords), top_kws= num_kws, calc_prob=calc_prob)

    # Save the model and embeddings
    #Bertopic.save(path = '{}{}-model'.format(MODELS_DIR, col_name) ) #save_ctfidf=True, serialization = "pickle"
    #np.save('{}{}-embeddings.npy'.format(MODELS_DIR,col_name), embeddings) # to load: all_embeddings = np.load('embeddings.npy')

    print('Topics generated: ', Bertopic.generate_topic_labels())  # -1 means outliers that should be ignored
    # Generate wordclouds for each subtopic
    create_wordcloud(Bertopic, 'BERTopic', period)

    # Save subtopic titles within each topic
    write_json_to_file('_', Bertopic.generate_topic_labels(), '{}-titles.txt'.format(filename), OUT_KWS)
    # Save subtopic kws within each topic
    write_json_to_file('_', str(Bertopic.get_topics()), '{}-kws.json'.format(filename), OUT_KWS)

    # Mark docs with Topic
    doctopicdf = Bertopic.get_document_info(df['Corpus'].values.tolist())
    print('In extract topics func for ', filename,', doctopicdf: \n', doctopicdf )

    # Add the Topic column to df. The original index is kept as is
    df[col_name] = [int(i) for i in doctopicdf['Topic']]

    # Get kw-freq per topic or subtopic
    subtopic_kws = get_topic_kws(Bertopic, topic_name='')
    subtopic_kws_df = pd.DataFrame({'Subtopic': subtopic_kws.keys(), 'Kws': subtopic_kws.values()})
    #get topic embeddings by the order of topic number and add to df
    subtopic_kws_df['Topic_Embeddings'] = [i for i in Bertopic.topic_embeddings_.tolist()]
    #get topic term matrix by the order of topic number and add to df
    #subtopic_kws_df['Topic_Term_Matrix'] = [i for i in Bertopic.topic_term_matrix_.tolist()]

    return df, subtopic_kws_df, probabilities

"""- Params: kws: a list of list of kws [[kws, kws], [kws, kws]], each list element contains kws for ea topic,
bow: kw-freq per doc, id2word: gensim dictionary that converts a word to its id, texts: corpus,
metric='c_v': by default, coherence score is calculated by this metric, 
n_top_kws=10: by default, the top 10 kws are used to calculate the coherence score.
- Functionality: use CoherenceModel from gensim to calculate the cohrence score given the params above.
- Return: a list of coherence score associating with a list of topics"""

""" Use the same bow and id2word generated by BERTopic as follows:
    # Extract vectorizer and analyzer from BERTopic: 
    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()
    # Extract features for Topic Coherence evaluation
    words = vectorizer.get_feature_names()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words, _ in topic_model.get_topic(topic)] for topic in range(len(set(topics))-1)]
    # Evaluate
    coherence_model = CoherenceModel(topics=topic_words, texts=tokens, corpus=corpus, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model.get_coherence()"""

def get_coherence_score(kws, bow, id2word, texts, metric='c_v', n_top_kws=10):
    cm = CoherenceModel(topics= kws, corpus= bow, dictionary= id2word, texts= texts,
                      coherence= metric, topn=n_top_kws)
    coherence = cm.get_coherence_per_topic()  # get coherence value
    print('Coherence: ', coherence)
    #subtopic_kws['Coherence'] = coherence
    return coherence

def get_perplexity_score_old(probs):
    log_perplexity = -1 * np.mean(np.log(np.sum(probs, axis=1)))
    perplexity = np.exp(log_perplexity)
    return perplexity

def get_perplexity_score(probs):
    # Add elements of different elements in probs
    sum_arr = []
    for element in probs:
        #print('len: ', len(element.shape))
        if len(element.shape) > 1: #if probs is a 2D array
            res = np.sum(element, axis=1).tolist()
            sum_arr.append(res)
        else:
            sum_arr.append([np.sum(element)])
    # sum_arr is a list of lists, each list contains the sum of probabilities of each doc belonging to each topic
    # Calculate perplexity
    sum_arr = np.array(sum(sum_arr, [])) # concatenate all lists in sum_arr and convert to np array
    print('sum_arr: ', sum_arr)
    log_perplexity = -1 * np.mean(np.log(sum_arr))
    print('log perplexity: ', log_perplexity)
    #log_perplexity = -1 * np.mean(np.log(np.sum(probs, axis=1))) #sum up by row
    perplexity = np.exp(log_perplexity)

    return perplexity

""" Param: probs_list: a list of np arrays of probabilities (probabilities of each doc belonging to each topic), and these arrays
have different sizes.
    This func creates a np array with the number of rows as the number of arrays in probs_list, and the number of columns as the max length of arrays in probs_list.
    Then, it copies the values from probs_list to the new array, and returns it.
"""
def format_prob_array(probs_list):
    # Check if all elements in probs_list have the same size
    if all(len(x) == len(probs_list[0]) for x in probs_list):
        return np.array(probs_list)
    else:
        print('Probs_list does not have the same size, reformatting...')
        # Find the max length of arrays in probs_list
        probs_list =  np.array(probs_list)
        max_len = max(len(arr) for arr in probs_list)
        # Create a new array with the same size
        new_probs_list = np.zeros((len(probs_list), max_len))
        # Copy the values from probs_list to new_probs_list
        for i, arr in enumerate(probs_list):
            new_probs_list[i, :len(arr)] = arr
        return new_probs_list

DATE = '1121' #'1106'1116
MODELS_DIR = '/Users/natal/Desktop/Disaster Analysis/Codes/Saved_models/'
if not os.path.exists(MODELS_DIR):  os.makedirs(MODELS_DIR)
#DATA_DIR = '/Users/natal/Desktop/Disaster Analysis/Data/processed-dfs/parsed-dfs/clean_data/Colab-Output/Topics/'

DATA_DIR = '/Users/natal/Desktop/Disaster Analysis/Codes/Output/06270/'
OUT =  '/Users/natal/Desktop/Disaster Analysis/Codes/Output/{}/'.format(DATE)
if not os.path.exists(OUT):  os.makedirs(OUT)

OUT_IMG_DIR = OUT + 'Imgs/'
if not os.path.exists(OUT_IMG_DIR):  os.makedirs(OUT_IMG_DIR)

OUT_KWS = OUT + 'Kws/'
if not os.path.exists(OUT_KWS):  os.makedirs(OUT_KWS)

OUT_COHERENCE = OUT + 'Coherence/'
if not os.path.exists(OUT_COHERENCE):  os.makedirs(OUT_COHERENCE)

B  = 'BERTopic'
BB = 'BERTopic-BERTopic'
D2V= 'Doc2Vec'
BM = 'BioMedLM'

SEED = 24

if __name__ == '__main__':
    # df19 has the added columns: Topic, and Top10kws
    #dir = '/Users/natal/Desktop/Disaster Analysis/Data/processed-dfs/parsed-dfs/clean_data'
  #  df19 = pd.read_excel(DATA_DIR + 'df19-topics-05-13-23.xlsx' , header=0, index_col=0) # 'clean-parsed_df19-4-13-23.xlsx'
    df19 = pd.read_excel(DATA_DIR + 'df19-subtopics-{}.xlsx'.format('0627'), header=0,index_col=0)  # 0627 'clean-parsed_df19-4-13-23.xlsx'

   # df22 = pd.read_excel(DATA_DIR + 'df22-topics-05-13-23.xlsx', header=0, index_col=0) # 'clean-parsed_df22-4-13-23.xlsx'
    df22 = pd.read_excel(DATA_DIR + 'df22-topics-{}.xlsx'.format('0627'), header=0,index_col=0)  # 'clean-parsed_df19-4-13-23.xlsx' 0604

    df19 = Process_Texts.convert_str_to_list(df19)
    df22 = Process_Texts.convert_str_to_list(df22)

    print('df19: ', df19.head())
    print('df22: ', df22.head())

    # Remove rows with empty abstracts
    print('Removing rows missing abstracts from df19....')
    df19 = remove_missing_values(df19)
    print('Removing rows missing abstracts from df22....')
    df22 = remove_missing_values(df22)

    # Reset index
    df19.reset_index(drop=True, inplace=True)
    df22.reset_index(drop=True, inplace=True)

    # Combine the abstracts and titles to create a corpus for each df
    texts19 = [i + '. ' + j for i, j in zip(df19['Title'].to_list(), df19['Abstract'].to_list())]
    texts22 = [i + '. ' + j for i, j in zip(df22['Title'].to_list(), df22['Abstract'].to_list())]
    #print(texts19)

    # Break sentences into words, remove stopwords, make bigram and lemmatize. The result is [[list of words in doc1],[words in doc2],..]
    corpus19 = Process_Texts.util_process_corpus(texts19)
    corpus22 = Process_Texts.util_process_corpus(texts22)

    df19['Corpus'] = [' '.join(i) for i in corpus19]
    df22['Corpus'] = [' '.join(i) for i in corpus22]

    # Convert the corpus into a dictionary BOW that provide ID to each word
    id2word19 = Process_Texts.create_corpora_dict(corpus19)
    id2word22 = Process_Texts.create_corpora_dict(corpus22)

    # Term Frequency of ea Doc: Convert a list of words into its integer id from the BOW and its freq in each doc.
    # Returns a list of (word id, word freq)
    doc_freq19 = [id2word19.doc2bow(text) for text in corpus19]
    doc_freq22 = [id2word22.doc2bow(text) for text in corpus22]

    #print('\n\n Corpus19:', corpus19, '\nLen: ', len(corpus19))
    print('\n\n Corpus22 len: ', len(corpus22))

    #Define list of topic seed for df19 to get similar topic w df22 for comparison
    seed_topic_list = [["diaster", "crisis", "management", "communication", "response"],
                       ["rumor", 'counterrumor', 'conspiracy', "uncertainty", "misinformation", 'credibility'],
                       ["public health", "disease", "pandemic", "influenza"],
                       ["humanitarian", "response", "volunteer", "digital"]]

    #df22 = pd.read_excel(OUT + 'df22-subtopics-{}.xlsx'.format(DATE), header=0, index_col=0) # added in for shortcut


    print('Extracting Subtopics from Post-Pandemic data...')
    df22, Bertopic_subtopic_kws22, B_probabilities = extract_topics(df22, period='BERTopic Post', embed= 'Bertopic', filename='{}-Df22-Subtopics-Bert'.format(DATE), num_kws=20,
                                                 col_name='Subtopic-B', min_cluster=15, calc_prob = True) #10

    print('Calculating Perplexity Score of the subtopics generated by BERTopic...')
    print('Shape of probabilities as a whole: {}, per unit: {} and prob: {}'.format(len(B_probabilities),B_probabilities[0],B_probabilities))
    B_perplexity = get_perplexity_score(B_probabilities)
    print('Perplexity score generated by BERTopic: {}'.format(B_perplexity))

    # Get coherence score for each subtopic extracted by Bertopic
    print('Coherence Score of the subtopics generated by BERTopic: ')
    Bertopic_subtopic_kws22['Coherence'] = get_coherence_score(Bertopic_subtopic_kws22['Kws'], bow=doc_freq22, id2word=id2word22,
                                                               texts=corpus22, metric='c_v', n_top_kws=10)
    print('Average Coherence score: ', Bertopic_subtopic_kws22['Coherence'].mean())
    Bertopic_subtopic_kws22['Perplexity'] = B_perplexity
    Bertopic_subtopic_kws22['Label'] = Bertopic_subtopic_kws22['Subtopic']
    print('Bertopic_subtopic_kws: \n', Bertopic_subtopic_kws22)
    #plot_coherence(x=Bertopic_subtopic_kws.index, y=Bertopic_subtopic_kws['Coherence'], label='BERTopic')
    plot_coherence(df=Bertopic_subtopic_kws22, name='BERTopic')
    Bertopic_subtopic_kws22.to_excel(OUT_KWS+ 'Bertopic-kw-coherence-Post-{}.xlsx'.format(DATE), index=False)

    #Save dfs
    df22.to_excel(OUT + 'df22-subtopics-{}.xlsx'.format(DATE), index=True)

#    ________________________Post-Pandemic BB________________________
    df22.drop(columns = ['Topic'], inplace=True)
    df22, topic_kws22, round1_probabilities = extract_topics(df22, 'Post', '{}-Df22-Topics-BERTopic'.format(DATE), num_kws=20, col_name='Topic',
                            embed = 'bert-round1', min_cluster= 30, calc_prob= True) #15
    # Save dfs
    df22.to_excel(OUT + 'df22-subtopics-{}.xlsx'.format(DATE), index=True)
    topic_kws22.to_excel(OUT_KWS + 'Round1-Bertopic-kw22-{}.xlsx'.format(DATE), index=False)

    print('df22: ', df22.head())
    df22.reset_index(drop=True, inplace=True)
    subtopicdocs_df22, BB_subtopic_kws22, BB_probabilities = extract_subtopics(df22, 'Post', filename='{}-Df22-Subtopics-BB'.format(DATE),
                                                embed= 'BB', min_cluster=15)
    # checking indices of subtopicdocs_df19 and df19
    print('Checking indices of subtopicdocs_df22 and df22: ')
    print('subtopicdocs_df22:\n', subtopicdocs_df22[['Topic', 'Document']])
    print('df22:\n', df22[['Topic', 'Title']])

    print('Calculating Perplexity Score of the subtopics generated by BERTopic-BERTopic...')
    print('Shape of probabilities as a whole: {}, per unit: {} xx {} and prob: {}'.format(len(BB_probabilities), BB_probabilities[0].shape, BB_probabilities[1].shape, BB_probabilities))
    round1_perplexity = get_perplexity_score(round1_probabilities)
    print('Round 1 perplexity: ', round1_perplexity)
    BB_perplexity = get_perplexity_score(BB_probabilities)
    print('Shape of perplexity {} and Perplexity score: {}'.format(BB_perplexity.shape, BB_perplexity))


    # Get coherence score for each subtopic extracted by Bertopic
    print('Post-Pandemic Coherence Score of the subtopics generated by BERTopic-BERTopic: ')
    BB_subtopic_kws22['Coherence'] = get_coherence_score(BB_subtopic_kws22['Kws'], bow=doc_freq22, id2word=id2word22,
                                                       texts=corpus22, metric='c_v', n_top_kws=10)
    print('Average Coherence score: ', BB_subtopic_kws22['Coherence'].mean())
    BB_subtopic_kws22['Perplexity'] = BB_perplexity
    BB_subtopic_kws22['Round_1_Perplexity'] = round1_perplexity
    BB_subtopic_kws22['Label'] = [(i, j) for i, j in zip(BB_subtopic_kws22['Topic'], BB_subtopic_kws22['Subtopic'])]
    print('BB_subtopic_kws22: \n', BB_subtopic_kws22)
    # plot_coherence(x= BB_subtopic_kws.index, y= BB_subtopic_kws['Coherence'], label='BERTopic-BERTopic')
    plot_coherence(df=BB_subtopic_kws22, name='BERTopic-BERTopic Post-Pandemic')

    BB_subtopic_kws22.to_excel(OUT_KWS + 'BB-kw-coherence-Post-{}.xlsx'.format(DATE), index=False)
    #  df22, _ = extract_topics(df22, 'Post', '{}-Df22-Topics-BERTopic'.format(DATE),col_name='Topic', embed='bert', min_cluster=15)
    # Save dfs
    # Add subtopics column to df
    df22['Subtopic-BB'] = subtopicdocs_df22['Topic']
    df22.to_excel(OUT + 'df22-subtopics-{}.xlsx'.format(DATE), index=True)

# #----------------------------------------------------------------- New Code for Doc2Vec -------------------------------------------------
    subtopicdocs_df22, doc2vec_subtopic_kws22, doc2vec_probabilities = extract_subtopics(df22, 'Post', filename='{}-Df22-Subtopics-D2V'.format(DATE), min_cluster=15, embed='D2V')
    print('############ Finish extracting subtopics with Doc2Vec embedding ############')
    print('suptopicdocs_df22: \n', subtopicdocs_df22)
    print('doc2vec_subtopic_kws22: \n', doc2vec_subtopic_kws22)
     # # checking indices of subtopicdocs_df19 and df19
     # print('Checking indices of subtopicdocs_df22 and df22: ')
     # print('subtopicdocs_df22:\n', subtopicdocs_df22[['Topic', 'Document']])
     # print('df22:\n', df22[['Topic', 'Title']])

    print('Calculating Perplexity Score of the subtopics generated by Doc2Vec...')
    print('Shape of probabilities as a whole: {}, and prob: {}'.format(len(doc2vec_probabilities), doc2vec_probabilities))

    doc2vec_perplexity = get_perplexity_score(doc2vec_probabilities)
    print('Shape of perplexity {} and Perplexity score: {}'.format(doc2vec_perplexity.shape, doc2vec_perplexity))

    # Get coherence score for each subtopic extracted by Bertopic
    print('Post-Pandemic Coherence Score of the subtopics generated by BERTopic-Doc2Vec: ')
    doc2vec_subtopic_kws22['Coherence'] = get_coherence_score(doc2vec_subtopic_kws22['Kws'], bow=doc_freq22, id2word=id2word22,
                                                        texts=corpus22, metric='c_v', n_top_kws=10)
    print('Average Coherence score: ', doc2vec_subtopic_kws22['Coherence'].mean())
    doc2vec_subtopic_kws22['Perplexity'] = doc2vec_perplexity
    #doc2vec_subtopic_kws22['Round_1_perplexity'] = round1_perplexity
    doc2vec_subtopic_kws22['Label'] = [(i, j) for i, j in zip(doc2vec_subtopic_kws22['Topic'], doc2vec_subtopic_kws22['Subtopic'])]
     # # plot_coherence(x= BB_subtopic_kws.index, y= BB_subtopic_kws['Coherence'], label='BERTopic-BERTopic')
    plot_coherence(df= doc2vec_subtopic_kws22, name='BERTopic-Doc2Vec Post-Pandemic')

    doc2vec_subtopic_kws22.to_excel(OUT_KWS + 'D2V-kw-coherence-Post-{}.xlsx'.format(DATE), index=False)
    # Save dfs
    # Add subtopics column to df
    df22['Subtopic-Doc2Vec'] = subtopicdocs_df22['Topic']
    df22.to_excel(OUT + 'df22-subtopics-{}-D2V.xlsx'.format(DATE), index=True)
#
#     #-------------end new code-----------------------------------------------------------------------------------------------------------------------------
#     #-------------------Post-Pandemic BERTopic-BioMedLM Topic Modeling----------
#
    subtopicdocs_df22, biomed_subtopic_kws22, biomed_probabilities = extract_subtopics(df22, 'Post', filename='{}-Df22-Subtopics-Biomed'.format(DATE), min_cluster=15, embed='biomed')
    print('############ Finish extracting subtopics with BioMed embedding ############')
    print('suptopicdocs_df22: \n', subtopicdocs_df22)
    print('biomed_subtopic_kws22: \n', biomed_subtopic_kws22)
    # # checking indices of subtopicdocs_df19 and df19
    # print('Checking indices of subtopicdocs_df22 and df22: ')
    # print('subtopicdocs_df22:\n', subtopicdocs_df22[['Topic', 'Document']])
    # print('df22:\n', df22[['Topic', 'Title']])

    print('Calculating Perplexity Score of the subtopics generated by BioMed...')
    print('Shape of probabilities as a whole: {}, and prob: {}'.format(len(biomed_probabilities), biomed_probabilities))

    biomed_perplexity = get_perplexity_score(biomed_probabilities)
    print('Shape of perplexity {} and Perplexity score: {}'.format(biomed_perplexity.shape, biomed_perplexity))

    # Get coherence score for each subtopic extracted by Bertopic
    print('Post-Pandemic Coherence Score of the subtopics generated by BERTopic-BioMed: ')
    biomed_subtopic_kws22['Coherence'] = get_coherence_score(biomed_subtopic_kws22['Kws'], bow=doc_freq22,
                                                              id2word=id2word22, texts=corpus22, metric='c_v', n_top_kws=10)
    print('Average Coherence score: ', biomed_subtopic_kws22['Coherence'].mean())
    biomed_subtopic_kws22['Perplexity'] = biomed_perplexity
    #biomed_subtopic_kws22['Round_1_perplexity'] = round1_perplexity
    biomed_subtopic_kws22['Label'] = [(i, j) for i, j in
                                       zip(biomed_subtopic_kws22['Topic'], biomed_subtopic_kws22['Subtopic'])]
    # # plot_coherence(x= BB_subtopic_kws.index, y= BB_subtopic_kws['Coherence'], label='BERTopic-BERTopic')
    plot_coherence(df= biomed_subtopic_kws22, name='BERTopic-BioMedLM Post-Pandemic')

    biomed_subtopic_kws22.to_excel(OUT_KWS + 'BioMed-kw-coherence-Post-{}.xlsx'.format(DATE), index=False)
    # Save dfs
    # Add subtopics column to df
    df22['Subtopic-BioMed'] = subtopicdocs_df22['Topic']
    df22.to_excel(OUT + 'df22-subtopics-{}.xlsx'.format(DATE), index=True)

    #-------------------Post-Pandemic BioMedLM representations and cluster them with HDBSCAN----------

    # Create a vector representation of each doc
    scaled_biomed_vecs22 = Doc2Vec.process_vector(corpus22, embeddings= 'BioMedLM')
    # Get a list of docs from ea topic and a topic list
    #docs_indx_per_topic19, topic_list19 = Doc2Vec.get_indx_per_topic(df19) already did above in lin 453
    # Cluster doc represetations within a topic into subtopics using HDBSCAN
    biomed_subtopic_per_topic22, biomed_doc_probabilities = Doc2Vec.get_subtopics_hdbscan(docs_indx_per_topic22, scaled_biomed_vecs22, min_cluster_size=15)

    df22 = Doc2Vec.add_subtopic_to_df(df22, 'Subtopic_BioMedLM', topic_list22, docs_indx_per_topic22, biomed_subtopic_per_topic22)

    # Get doc from each subtopic within a topic for KW extraction
    subtopic_corpus22 = df22.groupby(['Topic', 'Subtopic_BioMedLM'], group_keys=True)\
                            .apply(lambda x: '. '.join(x['Corpus']))
    # Kw extraction from each subtopic
    file_name = 'BERTopic-BioMedLM-{}Pandemic'.format('Post')
    biomed_subtopic_kw22 = Doc2Vec.run_keyBERT(subtopic_corpus22, 'Post', model_name= 'BERTopic-BioMedLM', file_name= file_name)
    print('Kws from subtopic generated by BioMedLM: ', biomed_subtopic_kw22)
    print('Calculating Coherence Score of the subtopics generated by BioMedLM...')
    biomed_subtopic_kw22['Coherence'] = get_coherence_score(biomed_subtopic_kw22['Kws'], bow= doc_freq22,
                                                             id2word=id2word22, texts= corpus22,
                                                             metric='c_v', n_top_kws=10)
    biomed_subtopic_kw22['Label'] = [(i, j) for i, j in zip(biomed_subtopic_kw22['Topic'], biomed_subtopic_kw22['Subtopic'])]
    print('BioMedLM Coherence: ',biomed_subtopic_kw22)
    #plot_coherence(x=biomed_subtopic_kw19.index, y=biomed_subtopic_kw19['Coherence'], label='BERTopic-BioMedLM')
    plot_coherence(df=biomed_subtopic_kw22, name='BERTopic-BioMedLM Post-Pandemic')

    biomed_subtopic_kw22.to_excel(OUT_KWS+ 'biomed-kw-coherence-Post-{}.xlsx'.format(DATE), index=False)

    # Save dfs
    # Add subtopics column to df
    df22.to_excel(OUT+ 'df22-subtopics-{}.xlsx'.format(DATE), index=True)
    gc.collect()
"""
    print('Extracting Subtopics from Pre-Pandemic data...')
    # Extract subtopics from each df using BERTopic
    df19, Bertopic_subtopic_kws = extract_topics(df19, period='BERTopic Pre', filename='{}-Df19-Subtopics-Bert'.format(DATE), num_kws=20,
                                                 col_name='Subtopic-B', embed= 'bert', min_cluster=3)

    #subtopicdocs_df19, Bertopic_subtopic_kws = extract_subtopics(df19, 'Pre', '{}-Df19-Subtopics-BERTopic'.format(DATE),
                                                               #  min_cluster=3)
    # Get coherence score for each subtopic extracted by Bertopic
    print('Coherence Score of the subtopics generated by BERTopic: ')
    Bertopic_subtopic_kws['Coherence'] = get_coherence_score(Bertopic_subtopic_kws['Kws'], bow=doc_freq19,
                                                             id2word=id2word19, texts=corpus19,
                                                             metric='c_v', n_top_kws=10)
    Bertopic_subtopic_kws['Label'] = Bertopic_subtopic_kws['Subtopic']
    print('Bertopic_subtopic_kws: \n', Bertopic_subtopic_kws)
    #plot_coherence(x=Bertopic_subtopic_kws.index, y=Bertopic_subtopic_kws['Coherence'], label='BERTopic')
    plot_coherence(df=Bertopic_subtopic_kws, name='BERTopic')
    Bertopic_subtopic_kws.to_excel(OUT_KWS+ 'Bertopic-kw-coherence-{}.xlsx'.format(DATE), index=False)
    df19.to_excel(OUT + 'df19-subtopics-{}.xlsx'.format(DATE), index=True)


    # Extract topics from each df using BERTopic-BERTopic
    # Uncomment the extract_topics row to get a topic if starting out without col topic
  #  df19, _ = extract_topics(df19, 'Pre', '{}-Df19-Topics'.format(DATE), col_name='Topic', embed= 'Round1', min_cluster=8, num_kws=20)
    subtopicdocs_df19, BB_subtopic_kws = extract_subtopics(df19, 'Pre', filename= '{}-Df19-Subtopics-BB'.format(DATE),min_cluster=3)
    # checking indices of subtopicdocs_df19 and df19
    print('Checking indices of subtopicdocs_df19 and df19: ')
    print('subtopicdocs_df19:\n', subtopicdocs_df19[['Topic', 'Document']])
    print('df19:\n', df19[['Topic', 'Title']])

    # Get coherence score for each subtopic extracted by Bertopic
    print('Coherence Score of the subtopics generated by BERTopic-BERTopic: ')
    BB_subtopic_kws['Coherence'] = get_coherence_score(BB_subtopic_kws['Kws'], bow=doc_freq19, id2word=id2word19,
                                                    texts=corpus19, metric='c_v', n_top_kws=10)
    BB_subtopic_kws['Label'] = [(i, j) for i, j in zip(BB_subtopic_kws['Topic'], BB_subtopic_kws['Subtopic'])]
    print('BB_subtopic_kws: \n', BB_subtopic_kws)
    #plot_coherence(x= BB_subtopic_kws.index, y= BB_subtopic_kws['Coherence'], label='BERTopic-BERTopic')
    plot_coherence(df=BB_subtopic_kws, name='BERTopic-BERTopic ')

    BB_subtopic_kws.to_excel(OUT_KWS+ 'BB-kw-coherence-{}.xlsx'.format(DATE), index=False)
  #  df22, _ = extract_topics(df22, 'Post', '{}-Df22-Topics-BERTopic'.format(DATE),col_name='Topic', embed='bert', min_cluster=15)
    # Save dfs
    # Add subtopics column to df
    df19['Subtopic-BB'] = subtopicdocs_df19['Topic']
    df19.to_excel(OUT+ 'df19-subtopics-{}.xlsx'.format(DATE), index=True)


    #-------------------Get Doc2Vec representations and cluster them with HDBSCAN----------
    # Create a vector representation of each doc
    scaled_doc2vec19 = Doc2Vec.process_vector(corpus19, embeddings= 'Doc2Vec')
    # Get a list of docs from ea topic and a topic list
    docs_indx_per_topic19, topic_list19 = Doc2Vec.get_indx_per_topic(df19)
    # Cluster doc represetations within a topic into subtopics using HDBSCAN
    doc2vec_subtopic_per_topic19, doc2vec_doc_probabilities = Doc2Vec.get_subtopics_hdbscan(docs_indx_per_topic19, scaled_doc2vec19,
                                                                 min_cluster_size=3)
    df19 = Doc2Vec.add_subtopic_to_df(df19, 'Subtopic_Doc2Vec', topic_list19, docs_indx_per_topic19,
                                      doc2vec_subtopic_per_topic19)
    # Get doc from each subtopic within a topic for KW extraction
    subtopic_corpus19 = df19.groupby(['Topic', 'Subtopic_Doc2Vec'], group_keys=True)\
                            .apply(lambda x: '. '.join(x['Corpus']))
    # Kw extraction from each subtopic
    file_name = 'BERTopic-Doc2Vec-{}Pandemic'.format('Pre')
    doc2vec_subtopic_kw19 = Doc2Vec.run_keyBERT(subtopic_corpus19, 'Pre', model_name='BERTopic-Doc2Vec', file_name=file_name)
    print('Kws from subtopic generated by Doc2Vec: ', doc2vec_subtopic_kw19)
    print('Calculating Coherence Score of the subtopics generated by Doc2Vec...')
    doc2vec_subtopic_kw19['Coherence'] = get_coherence_score(doc2vec_subtopic_kw19['Kws'], bow=doc_freq19,
                                                             id2word=id2word19, texts=corpus19,
                                                             metric='c_v', n_top_kws=10)
    doc2vec_subtopic_kw19['Label'] = [(i, j) for i, j in zip(doc2vec_subtopic_kw19['Topic'], doc2vec_subtopic_kw19['Subtopic'])]

    print('Doc2Vec_subtopic_kw19:\n ',doc2vec_subtopic_kw19)
    #plot_coherence(x=doc2vec_subtopic_kw19.index, y=doc2vec_subtopic_kw19['Coherence'], label='BERTopic-Doc2Vec')
    plot_coherence(df=doc2vec_subtopic_kw19, name='BERTopic-Doc2Vec')
    doc2vec_subtopic_kw19.to_excel(OUT_KWS+ 'doc2vec-kw-coherence-{}.xlsx'.format(DATE), index=False)


    #-------------------Get BioMedLM representations and cluster them with HDBSCAN----------
    # Reset the df index so we can grab the embedded doc vector from corpus19_vecs
    #df19.reset_index(drop=True, inplace=True)
    # Create a vector representation of each doc
    scaled_biomed_vecs19 = Doc2Vec.process_vector(corpus19, embeddings= 'BioMedLM')
    # Get a list of docs from ea topic and a topic list
    #docs_indx_per_topic19, topic_list19 = Doc2Vec.get_indx_per_topic(df19) already did above in lin 453
    # Cluster doc represetations within a topic into subtopics using HDBSCAN
    biomed_subtopic_per_topic19, biomed_doc_probabilities = Doc2Vec.get_subtopics_hdbscan(docs_indx_per_topic19, scaled_biomed_vecs19, min_cluster_size=3)

    df19 = Doc2Vec.add_subtopic_to_df(df19, 'Subtopic_BioMedLM', topic_list19, docs_indx_per_topic19, biomed_subtopic_per_topic19)

    # Get doc from each subtopic within a topic for KW extraction
    subtopic_corpus19 = df19.groupby(['Topic', 'Subtopic_BioMedLM'], group_keys=True)\
                            .apply(lambda x: '. '.join(x['Corpus']))
    # Kw extraction from each subtopic
    file_name = 'BERTopic-BioMedLM-{}Pandemic'.format('Pre')
    biomed_subtopic_kw19 = Doc2Vec.run_keyBERT(subtopic_corpus19, 'Pre', model_name= 'BERTopic-BioMedLM', file_name= file_name)
    print('Kws from subtopic generated by BioMedLM: ', biomed_subtopic_kw19)
    print('Calculating Coherence Score of the subtopics generated by BioMedLM...')
    biomed_subtopic_kw19['Coherence'] = get_coherence_score(biomed_subtopic_kw19['Kws'], bow=doc_freq19,
                                                             id2word=id2word19, texts=corpus19,
                                                             metric='c_v', n_top_kws=10)
    biomed_subtopic_kw19['Label'] = [(i, j) for i, j in zip(biomed_subtopic_kw19['Topic'], biomed_subtopic_kw19['Subtopic'])]
    print('BioMedLM Coherence: ',biomed_subtopic_kw19)
    #plot_coherence(x=biomed_subtopic_kw19.index, y=biomed_subtopic_kw19['Coherence'], label='BERTopic-BioMedLM')
    plot_coherence(df=biomed_subtopic_kw19, name='BERTopic-BioMedLM')
    biomed_subtopic_kw19.to_excel(OUT_KWS+ 'biomed-kw-coherence-{}.xlsx'.format(DATE), index=False)

    # Save dfs
    # Add subtopics column to df
    df19.to_excel(OUT+ 'df19-subtopics-{}.xlsx'.format(DATE), index=True)

    # Concludes the script with this line since we use block=False in the above plt.show
    #plt.show()
    """

