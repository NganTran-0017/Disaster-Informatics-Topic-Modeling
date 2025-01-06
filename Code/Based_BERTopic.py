
from bertopic import BERTopic
import gensim

from gensim.models.coherencemodel import CoherenceModel
from bertopic.vectorizers import ClassTfidfTransformer

from sentence_transformers import SentenceTransformer

import Process_Texts
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

import numpy as np
from transformers.pipelines import pipeline, AutoTokenizer
import gensim.downloader as api
import topic_modeling
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity
import torch
import gc
import os
import json
from scipy.sparse import save_npz


# hashseed = os.getenv('PYTHONHASHSEED')
# if not hashseed:
#     os.environ['PYTHONHASHSEED'] = '0'
#     os.execv(sys.executable, [sys.executable] + sys.argv)

# SEED = 24


# Tag a list of docs
def tagged_document(list_of_list_of_words):
   for i, list_of_words in enumerate(list_of_list_of_words):
      yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

# Train BERTopic func
def train_BERTopic_w_diff_embeddings(texts, embedding_name, stopwords=list(Process_Texts.all_stopwords),  topic_list=None,
                   reduced_topics="auto", top_kws=20, diverse=0.3, min_topic_sz=10, topic_number= None, calc_prob = True):  # , min_topic_sz
    """Details of parameter tunning: https://maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html#n_gram_range"""
    model_dir = topic_modeling.MODELS_DIR + "/" + embedding_name + "/"
    if topic_number != None:
        model_dir = model_dir + 'Topic-{}/'.format(str(topic_number))
    if not os.path.exists(model_dir):  os.makedirs(model_dir)

    # Use pre-trained model to extract embeddings
    if embedding_name == 'Bertopic' or embedding_name == 'Round1' or embedding_name == 'BB': ## BB = Bert-round2, bert = 1 stage Bertopic
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedding_model.encode(texts, show_progress_bar=True)  # Save embeddings from input

    elif embedding_name == "biomed":
        print('Devices available: ', torch.cuda.device_count())
        print([(i, torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())])
        if torch.cuda.is_available():
            device = "cuda:0"
            torch.cuda.empty_cache()
    #        del variables
            gc.collect()
        else:
            device = "cpu"
        print('Running BioMedLM with {} '.format(device))
        #model_dir = topic_modeling.MODELS_DIR + "BioMedLM/"
        #if not os.path.exists(model_dir):  os.makedirs(model_dir)
        #Check if the model is saved in the models directory
        #if len(os.listdir(topic_modeling.MODELS_DIR+'BERTopic_w_biomed_embeddings')) == 0: #No model here model_dir
        print('Downloading BioMedLM...')
        tokenizer = AutoTokenizer.from_pretrained("stanford-crfm/BioMedLM")
        tokenizer.pad_token = tokenizer.eos_token
        embedding_model = pipeline("feature-extraction", model="stanford-crfm/BioMedLM", tokenizer=tokenizer) #use cuda:0 device= 0
        print('Finished setting up model ')
        embeddings = None
        #else: #Load model from models directory
            # print('Loading BERTopic-BioMedLM from dir...')
            # tokenizer = AutoTokenizer.from_pretrained("stanford-crfm/BioMedLM")
            # tokenizer.pad_token = tokenizer.eos_token
            # model = BERTopic.load(topic_modeling.MODELS_DIR+'BERTopic_w_biomed_embeddings', embedding_model= pipeline("feature-extraction", model="stanford-crfm/BioMedLM", tokenizer=tokenizer))#model_dir



    elif embedding_name == "D2V":
        embedding_model = api.load("glove-wiki-gigaword-300") #glove-twitter-25"
        embeddings = None

    # Reduce dimentionality
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state= topic_modeling.SEED)

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
    #if model is None:
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
                     nr_topics= reduced_topics,
                     calculate_probabilities= calc_prob,
                     verbose=True)
    #else:
        # Train BERTopic
    topics, probs = model.fit_transform(texts, embeddings)

    print('Before reducing outliers----Topics: ', set(topics)); print('OG Probs: ', probs)
    # Only reduce outliers if the topics are not all outliers and there is at least one outlier doc
    if set(topics) != {-1} and -1 in set(topics):
        try:
            new_topics = model.reduce_outliers(texts, topics, strategy='embeddings', embeddings=embeddings)
            model.update_topics(texts, topics=new_topics)
        except:
            print('Error in reducing outliers')
        print('After reducing outliers----Topics: ', set(new_topics))
        #n_topics, new_probs = model.transform(texts, embeddings)
        #print('shape of old Probs: {} vs new Probs: {}'.format(probs.shape, new_probs.shape))
    else:
        print('Cannot reduce outliers as there is only outlier topic')
        # Set prob of outlier topic to near zero to avoid error in perplexity calculation
        probs = np.ones(probs.shape)*(1e-10)
        print(' New probs: ', probs)
    # Save topic representation, model, embeddings, topic_term matrix, and topic_labels
    np.save('{}topic_embeddings.npy'.format(model_dir), model.topic_embeddings_)  # to load: all_embeddings = np.load('embeddings.npy')
    model.save(model_dir, serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
    np.save('{}embeddings.npy'.format(model_dir), embeddings)  # to load: all_embeddings = np.load('embeddings.npy')
    save_npz('{}topic_term_matrix.npz'.format(model_dir), model.c_tf_idf_)
    # Convert and write JSON object to file
    with open("{}topic_labels.json".format(model_dir), "w") as outfile:
        json.dump(model.topic_labels_, outfile)
    return model, embeddings, probs


