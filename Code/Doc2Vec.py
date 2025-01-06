import gensim
import gensim.downloader as api
import os
import sys

import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import numpy as np
import hdbscan
from keybert import KeyBERT
import topic_modeling
import Process_Texts
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import joblib
from joblib import load


# Tag a list of docs
def tagged_document(list_of_list_of_words):
   for i, list_of_words in enumerate(list_of_list_of_words):
      yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

# param: corpus is a list of list of words from each doc
# Train a doc2vec model with tagged list of list of words
# return: a trained doc2vec model
def train_Doc2Vec(corpus, vector_size=50, epochs=30, seed=20, workers=1):
  data_for_training = list(tagged_document(corpus))
  print(data_for_training[:1])
  Doc2Vecmodel = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=30, seed=20, workers=1)
  Doc2Vecmodel.build_vocab(data_for_training)
  Doc2Vecmodel.train(data_for_training, total_examples= Doc2Vecmodel.corpus_count, epochs= Doc2Vecmodel.epochs)
  return Doc2Vecmodel


# Get embedded vector for each doc in corpus
def Doc2Vec_representation(corpus):
  corpus_vecs = [] # make sure the index is consistent w the df
  Doc2Vecmodel =  train_Doc2Vec(corpus)
  for ind, doc in enumerate(corpus):
    corpus_vecs.append(Doc2Vecmodel.infer_vector(corpus[ind]))
  return np.array(corpus_vecs)

# param: corpus
# This func call Doc2Vec_representation to create corpus_vecs, which is a list of vector representations of all docs
# This func scale the list of vector representations and use UMAP for dimension reduction
# return: a list of 2D and scaled vectors
def process_vector0(corpus, n_components=2):
  corpus_vecs = Doc2Vec_representation(corpus)
  print('Shape before scale: ', corpus_vecs.shape)
  # Scale the vector representation of each doc
  scaled_vecs = StandardScaler().fit_transform(corpus_vecs)
  print('Shape after scale: ', scaled_vecs.shape)

  # Reducting the data in 2 dimensions using UMAP
  UMAP_Object = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2, random_state= topic_modeling.SEED)
  scaled_vecs = UMAP_Object.fit_transform(scaled_vecs)
  print('Shape after dim reduction: ', scaled_vecs.shape)
  return scaled_vecs

# Params: list_of_doc is a list of all processed documents. The list is created by concatenating words from corpus19 or corpus22
# Functionality: Use the pretrained BioMedLM as a tokenizer and LM to tokenize the doc and embed it into its vector representation.
## The embedding result is averaged across the columns to obtain a 1D vector representation for the entire doc.
## Repeat the steps for all documents in the list and save the embedding vectors (as numpy array) into a list (of np arrays)
# Return a 2D np array of embeddings. Each row is an embedding of a doc
def BioMedLM_representation(list_of_doc):
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  print('Running BioMedLM with {} '.format(device))
  tokenizer = AutoTokenizer.from_pretrained("stanford-crfm/BioMedLM")
  model_dir = topic_modeling.MODELS_DIR + "BioMedLM/"
  if not os.path.exists(model_dir):  os.makedirs(model_dir)
  model = AutoModelForCausalLM.from_pretrained("stanford-crfm/BioMedLM", device_map= 'auto',
                                               offload_folder= model_dir, offload_state_dict = True)
                                               #torch_dtype=torch.float16)
  embeddings_list = []
  #sentence = "The disaster causes damage to several areas"
  for doc in list_of_doc:
    encoded_input = tokenizer.encode(doc, return_tensors='pt').to(device)
    with torch.no_grad():
        # The output of model(encode_input) has 2 elements: the final hidden state and the attention weights. We will grab the first element as the embedding
        model_output = model(encoded_input)[0] # The final hidden states contain the contextualized representation of the input doc at each position.
    # Take the mean of the final hidden states along the seq length dim to obtain a single vector representation for the entire sdoc
    doc_embedding = model_output.mean(dim=1).squeeze() # output in tensor
    embeddings_list.append(doc_embedding.numpy()) # add the embedding vector in numpy format to a list of embeddings
    print(doc_embedding)
  return np.array(embeddings_list)


# param: corpus is a list of BOW from each doc. embeddings is the embeddings method to create doc representation
# This func call Doc2Vec_representation to create corpus_vecs, which is a list of vector representations of all docs
# This func scale the list of vector representations and use UMAP for dimension reduction
# return: a list of 2D and scaled vectors
def process_vector(corpus, embeddings, n_components=2):
  if embeddings == 'Doc2Vec':
    print('Getting doc embeddings from Doc2Vec....')
    corpus_vecs = Doc2Vec_representation(corpus)
    print('Shape before scale: ', corpus_vecs.shape, type(corpus_vecs))
    # Scale the vector representation of each doc
    scaled_vecs = StandardScaler().fit_transform(corpus_vecs)
    print('Shape after scale: ', scaled_vecs.shape)
    # Reducting the data in 2 dimensions using UMAP
    UMAP_Object = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2, random_state=topic_modeling.SEED)
    scaled_vecs = UMAP_Object.fit_transform(scaled_vecs)
    print('Shape after dim reduction: ', scaled_vecs.shape)

  if embeddings == 'BioMedLM':
    print('Getting doc embeddings from BioMedLM....')
    # concat the list of BOW into
    concat_corpus = [' '.join( i ) for i in  corpus]
    # Get document embeddings for each doc. Corpus_vecs is a list of np arrays of the doc embeddings.
    corpus_vecs = BioMedLM_representation(concat_corpus)
    print('Shape before scale: ', corpus_vecs.shape, type(corpus_vecs))
    # Scale the vector representation of each doc
    scaled_vecs = StandardScaler().fit_transform(corpus_vecs)
    print('Shape after scale: ', scaled_vecs.shape)
    # Reducting the data in 2 dimensions using UMAP
    UMAP_Object = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2, random_state=topic_modeling.SEED)
    scaled_vecs = UMAP_Object.fit_transform(scaled_vecs)
    print('Shape after dim reduction: ', scaled_vecs.shape)
  return scaled_vecs

# Get a list of doc indices for each topic and a list of topics
def get_indx_per_topic(df):
  docs_indx_per_topic = {} # contains topic# as key and the list of doc indices as value
  topic_list = df['Topic'].value_counts().index.to_list()
  for t in topic_list:
    filt = df['Topic'] == t
    docs_indx_per_topic[t] = df.loc[filt].index.to_list()
  return docs_indx_per_topic, topic_list

""" Cluster docs from each topics into smaller groups using HDBSCAN.
Return: - subtopic_per_topic: a dict with topic# as key and a list of subtopics as value
        - probabilities: a list of probabilities of each doc to be in each subtopic cluster 
"""
def get_subtopics_hdbscan(docs_indx_per_topic, scaled_vecs, min_cluster_size=3, min_samples=3):
  print('docs indx in ea topic: \n',docs_indx_per_topic)
  SEED = 20
  np.random.seed(20) # set random seed for reproducible results from HDBSCAN. However, the result is non-deterministic because Doc2Vec works based on randomization
  # Get the embedded doc vector from each topic and cluser them into sub-topics
  subtopic_per_topic = {} # topic: key, a list of subtopics within the topic: value
  probabilities = [] # a list of probabilities of each doc to be in each subtopic cluster
  # Cluster subtopics from each Topic using HDBSCAN
  for topic in docs_indx_per_topic.keys():
    # Run HDBSCAN on the vector representation of each topic to get the subtopics
    data = [ scaled_vecs[ind] for ind in docs_indx_per_topic.get(topic)] # get embed vectors from topic 0
    print('--------------Topic {}-------------'.format(topic))
    print('Len of vecs: {}'.format(len(data)))
    cluster_model = hdbscan.HDBSCAN(algorithm='best', min_cluster_size= min_cluster_size,
                                    approx_min_span_tree=True, allow_single_cluster= False,
                                    metric='euclidean', min_samples= min_samples) # https://hdbscan.readthedocs.io/en/latest/api.html
    cluster_model.fit_predict(data)
    subtopic_per_topic[topic] = cluster_model.labels_
    print('Cluster labels: ', subtopic_per_topic[topic])
    # Get the probability of each doc to be in each subtopic cluster
    prob =  hdbscan.all_points_membership_vectors(cluster_model)# cluster_model.probabilities_ . For a set of new unseen points we could use membership_vector()
    probabilities.append(prob)
    print('Type of probabilities: {} and shape: {}, and values: {}'.format(type(prob), prob.shape, prob))
    # save model
    joblib.dump( cluster_model, topic_modeling.MODELS_DIR + 'hdbscan_model_{}_topic_{}.joblib'.format('22', topic)) # to load: model = load('finalized_model.joblib') # make predictions test_labels, strengths = model.approximate_predict(model, test_points)
  return subtopic_per_topic, probabilities

# params: df: the df that we need to add col subtopic. colname: subtopic column name
# docs_indx_per_topic: a dict of topic (key): list of indx (value)
# subtopic_per_topic: a dict of topic (key): a list of subtopics (value)
# Functionality: update df with a subtopic col from subtopic_per_topic dict based on the the docs in each topic
# Return: updated df with subtopic col
def add_subtopic_to_df(df, colname, topic_list, docs_indx_per_topic, subtopic_per_topic):
  # Add col subtopic to df
  df[colname] = ''
  for topic in topic_list:
    df.loc[docs_indx_per_topic[topic], colname] = subtopic_per_topic.get(topic)
  return df

def run_keyBERT (subtopic_corpus, period, model_name, file_name):
  subtopic_kws_df = pd.DataFrame(columns = ['Topic', 'Subtopic', 'Kws'])
  for topic in subtopic_corpus.index.levels[0]:
    for s_topic in subtopic_corpus[topic].index:
      print('Topic {} - Subtopic {}: '.format(topic, s_topic))
      #print(subtopic_corpus[topic, s_topic])

      kw_model = KeyBERT()
      kws_per_subtopic = dict(kw_model.extract_keywords(subtopic_corpus[topic, s_topic], keyphrase_ngram_range=(1, 1),\
                                            stop_words= list(Process_Texts.all_stopwords), use_mmr=True, diversity=0.3, top_n = 20))
      topic_modeling.create_wordcloud_from_freq(dict(kws_per_subtopic), period, model_name= model_name, topic_name= topic,img_name= file_name, subtopic_name= s_topic)
      # save kws from each subtopic   (text, period, model_name='BERTopic', img_name = file_name, topic_name=topic_name, subtopic_name=topic)
      #filename = '{}-keyBERT-subtopic'.format(model_name)
      topic_modeling.write_json_to_file('{} Subtopic {}'.format(topic, s_topic), dict(kws_per_subtopic), '{}.json'.format(file_name), topic_modeling.OUT_KWS)
      #subtopic_kws_df = pd.concat( [subtopic_kws_df, pd.DataFrame({'Topic':topic, 'Subtopic': s_topic, 'Kws': kws_per_subtopic.values()})] )
      row = [topic, s_topic, list(kws_per_subtopic.keys())]
      print('new row: ', row)
      subtopic_kws_df.loc[len(subtopic_kws_df)] = row
  return subtopic_kws_df

if __name__ == '__main__':
  ...
#
#   subtopic_kws_df = pd.concat([subtopic_kws_df, pd.DataFrame(
#     {'Topic': topic, 'Subtopic': subtopic_kws.keys(), 'Kws': subtopic_kws.values()})])
#
#   # Mark docs with subtopic
#   doctopicdf = Bertopic.get_document_info(docsbytopic.values.tolist())
#   # Make sure the original index is kept from docsbytopic
#   doctopicdf.set_index(docsbytopic.index.values, inplace=True, drop=True)
#   subtopic_doclist.append(doctopicdf)
# return pd.concat(subtopic_doclist, ignore_index=False), subtopic_kws_df.reset_index(drop=True)