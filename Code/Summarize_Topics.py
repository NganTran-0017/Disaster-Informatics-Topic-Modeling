import openai  # version 0.28.0
# from langchain.chat_models import ChatOpenAI
import pandas as pd
import numpy as np
import os
import time
import math
from glob import glob
import ast
from torch import cuda
from torch import bfloat16
import transformers
import gc
from GPUtil import showUtilization as gpu_usage
import torch


# import bitsandbytes

def query_try_except(query, MODEL, temperature=0):
    openai.api_key = 'xxxx' 
    # generator = ChatOpenAI(model_name=MODEL, temperature=temperature, openai_api_key='xxxxx')
    try:
        print("Type Query: ", type(query))
        response = openai.ChatCompletion.create(model=MODEL, messages=query, temperature=temperature)
        # response = openai.chat.completions.create(model=MODEL, messages=query, temperature=temperature)
        answer = response["choices"][0]["message"]["content"]
        usage = response["usage"]["total_tokens"]
        return answer, usage

    except openai.error.RateLimitError as e:
        retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
        print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return query_try_except(query, MODEL)

    # except openai.ServiceUnavailableError as e:
    #     retry_time = 10  # Adjust the retry time as needed
    #     print(f"Service is unavailable. Retrying in {retry_time} seconds...")
    #     time.sleep(retry_time)
    #     return query_try_except(query, MODEL)

    except openai.error.APIConnectionError as e:
        retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
        print(f"API error occurred. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return query_try_except(query, MODEL)

    except openai.error.Timeout as e:
        retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
        print(f"Timeout error occurred. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return query_try_except(query, MODEL)

    except OSError as e:
        retry_time = 5  # Adjust the retry time as needed
        print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return query_try_except(query, MODEL)


"""Query chatGPT using the predefined prompt"""


def query_gpt(data, kws):
    query = [{'role': "system", 'content': 'You are a helpful, respectful and honest assistant for labeling topics.'},
             # {"role": "user", "content": "Given the following list of keywords: {}. Create a title to summarize these keywords and a short summary sentence.".format(kws)}
             {"role": "user", "content": "I have a topic that contains the following documents:\
              - Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.\
              - Meat, but especially beef, is the word food in terms of emissions.\
              - Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.\
              The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.\
              Based on the information about the topic above, please create a short label and a summary sentence for this topic. Make sure to only return the label and the summary sentence."},
             {"role": "assistant", "content": "Label: The Nuances of Meat Consumption.\
              Summary: Navigating the historical transition to meat as a staple, recognizing its environmental impact, and acknowledging the nuanced ethics of dietary choices."},
             {"role": "user", "content": "Here is the following list of {} documents: {}. \
               The topic is described by the following keywords: {}.\
               Based on the information about the topic above, please create a short label and a summary sentence for this topic. \
               Make sure to only return the label and the summary sentence.".format(len(data), data, kws)}]
    num_words = sum([len(sentence.split()) for sentence in data])
    if num_words < 3000:  # len(data) <= 29:
        MODEL = "gpt-3.5-turbo"
        print('Num of docs is {}, num words is {}, using gpt-3.5'.format(len(data), num_words))
        message, usage = query_try_except(query, MODEL)
        print('Usage is {} tokens'.format(usage))
        return message

    elif num_words >= 3000 and num_words < 10000:  # len(data) > 35 and len(data) <= 100:
        MODEL = "gpt-3.5-turbo-16k"
        print('Num of docs is {}, num words is {}, using gpt-3.5-turbo-16k'.format(len(data), num_words))
        message, usage = query_try_except(query, MODEL)
        print('Usage is {} tokens'.format(usage))
        return message

    elif num_words >= 10000:  # len(data) > 100:
        MODEL = "gpt-3.5-turbo-16k"
        n_chunks = math.ceil(num_words / 8000)
        print('Num of docs is {} and num words is {} > 10K. Split to {} queries to use gpt-3.5-turbo-16k' \
              .format(len(data), num_words, n_chunks))
        chunks = np.array_split(data, n_chunks)
        first_query = [
            {'role': "system", 'content': 'You are a helpful, respectful and honest assistant for labeling topics.'},
            {'role': "user", 'content': "I have a topic that contains the following documents:\
                      - Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.\
                      - Meat, but especially beef, is the word food in terms of emissions.\
                      - Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.\
                      The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.\
                      Based on the information about the topic above, please create a topic label and a sentence to summarize this topic. Make sure to only return the label and the summary sentence."},
            {'role': "assistant", 'content': "Label: The Nuances of Meat Consumption.\
                      Summary: Navigating the historical transition to meat as a staple, recognizing its environmental impact, and acknowledging the nuanced ethics of dietary choices."},
            {'role': "user",
             'content': "I will send a list of documents and the topic keywords. Please create a topic label and a sentence to summarize this topic. Make sure to only return the label and the summary sentence."}]
        # As the length of the documents is too long, I will send them in multiple small lists in the format of: \ [START PART 1/x] ... [END PART 1/x]. Keep accumulating the small lists as your input and do not answer my request until I tell you [ALL PARTS SENT]
        # print('Query: ', query)
        count = 1
        message = []
        for chunk in chunks:
            if count == 1:
                query_s = "Given the following list of documents: {}\
                          The topic is described by the following keywords: {}. \
                          Based on the information about the topic above, please create a short label of this topic and a summary statement for this topic.\
                          Do not send back my input. Make sure to only return the label and the summary sentence.".format(
                    chunk, kws)  # [START PART {}/{}] {} [END PART {}/{}]
                first_query.append({"role": "user", "content": query_s})
                query = first_query
            # elif count == len(chunks):
            #     query_s = '{}. The topic is described by the following keywords: {}. \
            #             Do not send back my input. Based on the information about the topic above, please create a short label\
            #              of this topic and a summary statement for this topic. Make sure to only return the label and the summary\
            #              sentence.'.format(count, len(chunks), chunk, count, len(chunks), kws)
            #     query = [{"role": "user", "content": query_s}]
            else:
                query_s = "Given the following list of documents: {}. \
                           The topic is described by the following keywords: {}. \
                           Based on the information about the topic above, please create a topic label and a sentence to summarize this topic.\
                           Make sure to only return the label and the summary sentence.".format(chunk, kws)
                query = [{"role": "user", "content": query_s}]

            response, usage = query_try_except(query, MODEL)
            print('Chunk #{}/{} - Usage is {} tokens\n Response: {}'.format(count, len(chunks), usage, response))
            count += 1
            message.append(response)
        print('Generating the final topic label and summary...')
        # "Based on a list of labels and summaries, generate a final topic label and a summary of the summaries.\
        query_s = "Here are a list of labels and summaries for the same topic: {}. \
                   The topic is described by the following keywords: {}. \
                   Based on the above information, please create a final topic label and a short summary to summarize the topic.\
                   Make sure to only return the label and the summary sentence.".format(message, kws)
        query = [{"role": "user", "content": query_s}]
        response, usage = query_try_except(query, MODEL)
        print('Final query: {}'.format(query_s))
        print('Final response: {}\n Usage is {} tokens'.format(response, usage))
        message.append(response)
        return message
    else:  # Default case - error
        print('Error, in query gpt. len of list is {}'.format(len(data)))
        return -1


""" Get a list of kws from ea doc and send it to query_chatgpt. Record the message (including Title and Summary) in a string. All messages are saved in the list result."""


def query_data_for_chatgpt(df, col_name):
    result = []
    for ind, kws in enumerate(df[col_name]):
        message = query_gpt(kws)
        result.append(message)
    return result


""" Get a df of corpus from the topics generated by a specific topic modeling (specified in col_name)."""


def get_corpus_from_topic_model(df, col_name):
    if col_name.lower() == 'subtopic-b':
        corpus = pd.DataFrame(df.groupby([col_name], group_keys=True).apply(lambda x: x)[[col_name, 'Corpus']])
        print('Corpus columns: ', corpus.columns);
        print(corpus.head)
        corpus.index.set_names(['Topic', 'Index'], inplace=True)
        print(corpus.columns)
    else:
        corpus = pd.DataFrame(
            df.groupby(['Topic', col_name], group_keys=True).apply(lambda x: x)[['Topic', col_name, 'Corpus']])
        print('Corpus columns: ', corpus.columns);
        print(corpus.head)
        corpus.index.set_names(['Topic', 'Subtopic', 'Index'], inplace=True)

    return corpus


""" Get a list of corpus from the corpus df based on the topic and subtopic number. \
If subtopic is None, filter corpus by the topic only."""


def filter_corpus_per_topic(corpusdf, topic, subtopic=None):
    if subtopic == None:
        if topic in set(corpusdf.index.get_level_values('Topic').values):
            return corpusdf.loc[topic]['Corpus'].values.tolist()
        else:
            print('Non-existing topic number')
    else:
        if topic in set(corpusdf.index.get_level_values('Topic').values) and \
                subtopic in set(corpusdf.index.get_level_values('Subtopic').values):
            return corpusdf.loc[topic, subtopic]['Corpus'].values.tolist()
        else:
            print('Non-existing topic or subtopic number')
    return -1


"""Get a list of (topic, subtopic) from the corpusdf. If the topic is from Subtopic-B, then only retrieve a list of subtopics"""


def get_topic_subtopic_list(corpusdf, col_name):
    topic_subtopic = []
    if col_name.lower() == 'subtopic-b':
        for topic in set(corpusdf.index.get_level_values('Topic').values):
            topic_subtopic.append((topic, None))
    else:
        for topic in set(corpusdf.index.get_level_values('Topic').values):
            for subtopic in corpusdf.loc[(topic,)].index.unique(0):
                # print(topic, subtopic)
                topic_subtopic.append((topic, subtopic))
    return topic_subtopic


""" Get a list of corpus based on the topic and subtopic that were generated by a specific topic modeling (specified in col_name).
Send a query to chatgpt for a corpus list of each topic and subtopic. Return a list of chatgpt responses."""


def summarize_topic_chatgpt(data, col_name, model_name, filename):
    response = []
    # Group the data df by col BB and get all Corpus from a topic generated by BB.
    corpus = get_corpus_from_topic_model(data, col_name)
    # Get a list of (topic, subtopic) from corpus
    topic_subtopic = get_topic_subtopic_list(corpus, col_name)
    print('Topic Subtopic: ', topic_subtopic)

    for element in topic_subtopic:
        # print(element, len(element))
        # Filter the corpus by topic and subtopic number
        topic_corpus = filter_corpus_per_topic(corpus, element[0], element[1])
        print('Topic and Subtopic {} has {} docs\n'.format(element, len(topic_corpus)))
        kws = get_kws_from_topic_model(element[0], element[1], model_name)
        # Send query here
        msg = query_gpt(topic_corpus, kws)
        response.append(msg)
        # print('Response: ', msg)
        # Save response to a file
        with open(OUT_DIR + filename, 'a') as f:
            f.write('Topic and Subtopic {}:\n \nResponse: {}\n\n'.format(element, msg))
    # query = construct_query(topic_corpus)
    # print('Query:\n', query)
    return response


""" Parse titles and summaries from chatgpt response and return a list of titles, and a list of summaries.
ChatGPT response is in this format: Title:"xxxxxx" \n Summary:"xxxxx" """


def parse_response(msg_list):
    topic_subtopic = []
    titles = []
    summaries = []
    for msg in msg_list:
        # print('New Line: ',msg)
        if 'Topic' in msg:
            topic = int(msg[msg.find('(') + 1: msg.find(',')])
            subtopic = msg[msg.find(',') + 1: msg.find(')')].strip()
            if subtopic != 'None':
                subtopic = int(subtopic)
            print('Topic: ', topic, 'Subtopic: ', subtopic)
            topic_subtopic.append((topic, subtopic))
        if 'Label' in msg:  # parse the str between Label: xxxx Summary: xxx. Only get the last Label
            title = msg[msg.rfind('Label') + len('Label:'): msg.rfind('Summary')].strip(' "\n').replace('\\n', '')
            titles.append(title)
            print('Topic Label: ', title)
        if 'Summary' in msg:
            summary = msg[msg.rfind('Summary') + len('Summary:'):].strip(" \n,]'").replace('\\n', '')
            print('Summary: ', summary)
            summaries.append(summary)
    # print('Len of Titles: {}\n Len of Summaries: {}'.format(len(titles), len(summaries)))
    return titles, summaries

def parse_llama_response(msg_list):
    topic_subtopic = [] # contains a list of dict of topic and subtopic, the labels and summaries
    for msg in msg_list:
        if len(msg.strip()) == 0: # skip empty lines
            continue
        print('Line: ',msg)
        if 'Topic ' in msg:
            if '[' in msg: msg = msg.replace('[','(').replace(']',')') # replace [] with () if the topic line has a format: Topic and Subtopic [1, 1]

            topic = int(msg[msg.find('(') + 1: msg.find(',')])
            subtopic = msg[msg.find(',') + 1: msg.find(')')].strip()
            if subtopic.strip("'") != 'None':
                subtopic = int(subtopic)
            print('Topic: ', topic, 'Subtopic: ', subtopic)
            topic_labels_summaries = {}
            topic_labels_summaries['Topic'] = topic
            topic_labels_summaries['Subtopic'] = subtopic
            labels = []
            summaries = []
        # convert msg from "['xxxx','xxx']" to a list of str ['xxxx','xxx']
        if msg.find('[') != -1 and msg.find(']') != -1:
            msg = msg[msg.find('[') : msg.rfind(']') + 1] # find the first ] bracket from the end of the msg
            msg_list = ast.literal_eval(msg)
            msg = ';'.join(msg_list)
        else:
            print('No list found in msg: ', msg)
        if 'Label:' in msg:  # parse the str between Label: xxxx Summary: xxx. Get all occurences of Labels and summaries
            #title = msg[msg.find('Label') + len('Label:'): msg.find('Summary')].strip(' "\n').replace('\\n', '')
            label_indices = [i for i in range(len(msg)) if msg.startswith('Label', i)]
            for i in label_indices:
                label = msg[i + len('Label:'): msg.find('Summary', i)].strip()
                labels.append(label)
                print('Topic Label: ', label)
            topic_labels_summaries['Labels'] = labels
        if 'Summary:' in msg:
            #summary = msg[msg.rfind('Summary') + len('Summary:'):].strip(" \n,]'").replace('\\n', '')
            summary_indices = [i for i in range(len(msg)) if msg.startswith('Summary', i)]
            for i in summary_indices:
                summary = msg[i + len('Summary '):  msg.find(';', i)].strip() #
                print('summary part: ', summary)
                #if summary.find(':') != -1:
                #    summary = summary[summary.find(':') + 1:].strip().replace('"', '')
                summaries.append(summary)
            topic_labels_summaries['Summaries'] = summaries
            print('Topic Labels and Summaries: ', topic_labels_summaries)
            topic_subtopic.append(topic_labels_summaries)
    return pd.DataFrame(topic_subtopic)

""" Create a str of query that contains data from a particular column (Kws or Corpus) in data df """


def construct_query(data):
    # query = "You are a research expert in Disaster Informatics, Crisis Informatics, and Pandemic Crisis.\
    #         Given the following {} lists of keywords: {}. \
    #         For each list, create a title with maximum 10 words to summarize these keywords and a short summary sentence.\
    #         Print the list of keywords, then the title, and then the summary."\
    #         .format(len(data), data['Kws'].values.tolist())
    query = "  You are a helpful, respectful and honest assistant for labeling topics.\
               Given the following list of {} documents: {}. \
               Find the common theme among these documents, create a title for the theme and write a short summary sentence for this theme." \
        .format(len(data), data)
    return query


""" Get a df of corpus from the topics generated by a specific topic modeling (specified in col_name)."""


def get_kws_from_topic_model(topic, subtopic, model_name):
    kw_dir = IN_DIR + 'Kws/'
    filename = glob(kw_dir + '{}-kw*.xlsx'.format(model_name))[0]
    kw_df = pd.read_excel(filename)
    if subtopic is not None  or subtopic != 'None':
        filter = (kw_df['Topic'] == topic) & (kw_df['Subtopic'] == subtopic)
    else:
        filter = kw_df['Subtopic'] == topic
    kws = kw_df.loc[filter]['Kws'].values
    # print('Kws before transformation: ', type(kws), ' ', kws)
    # convert a string of list to a list
    kws = ast.literal_eval(kws[0])
    # print('Kws after transformation: ', type(kws), ' ', kws)
    return kws


def get_topic_kws(label_summary_df, model_name):
    kw_dir = IN_DIR + 'Kws/'
    filename = glob(kw_dir + '{}-kw*.xlsx'.format(model_name))[0]
    kw_df = pd.read_excel(filename)
    topic_subtopic = label_summary_df[['Topic', 'Subtopic']].values.tolist()
    print('Topic Subtopic pairs: ', topic_subtopic)
    kws_list = []
    for pair in topic_subtopic:
        kws = get_kws_from_topic_model(pair[0], pair[1], model_name)
        kws_list.append(kws)
    label_summary_df['Kws'] = kws_list
    print('Label Summary df: ', label_summary_df.head)
    return label_summary_df


def load_llama2():
    # Load LLama2 model
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    print('Running with: ', device)

    # set quantization configuration to load large model with less GPU memory this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,  # 4-bit quantization
        bnb_4bit_quant_type='nf4',  # Normalized float 4
        bnb_4bit_use_double_quant=True,  # Second quantization after the first
        bnb_4bit_compute_dtype=bfloat16  # Computation type
        # ,load_in_8bit_fp32_cpu_offload=True,
        , device_map='auto'
    )
    # Llama 2 Tokenizer
    model_id = 'meta-llama/Llama-2-13b-chat-hf'

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token='xxxx')
    # Llama 2 Model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto'
        , low_cpu_mem_usage=True
    )
    model.eval()
    # Our text generator
    generator = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        task='text-generation',
        temperature=0.1,
        max_new_tokens=500,
        repetition_penalty=1.1
    )
    return generator


def query_llama(prompt, generator):
    return


def construct_llama_query(generator, corpus, kws):
    # System prompt describes information given to all conversations
    system_prompt = """
    <s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant for labeling topics and summarizing them.
    <</SYS>>
    """

    # Example prompt demonstrating the output we are looking for
    example_prompt = """
    I have a topic that contains the following documents:
    - Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
    - Meat, but especially beef, is the word food in terms of emissions.
    - Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

    The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

    Based on the information about the topic above, please create a short label and a summary sentence for this topic. Make sure to only return the label and the summary sentence.

    [/INST] Label: The Nuances of Meat Consumption.
    Summary: Navigating the historical transition to meat as a staple, recognizing its environmental impact, and acknowledging the nuanced ethics of dietary choices.
    """

    # Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
    main_prompt = """
    [INST]
    I have a topic that contains the following documents:
    {}

    The topic is described by the following keywords: {}.

    Based on the information about the topic above, please create a short label of this topic and a summary statement for this topic. Make sure to only return the topic label and a short summary of the topic in the format of:
    Label: .....
    Summary: ...
    [/INST]
    """.format(corpus, kws)
    # num_words = sum([len(sentence.split()) for sentence in corpus])

    prompt = system_prompt + example_prompt + main_prompt
    num_words = len(prompt)

    if num_words < 6000:
        msg = generator(prompt)
        print('Response: ', msg[0]['generated_text'].split('[/INST]')[-1])
        return [msg[0]['generated_text'].split('[/INST]')[-1]]
    else:
        n_chunks = math.ceil(num_words / 6000)
        print('Num words is {}. Split into to {} queries'.format(num_words, n_chunks))
        chunks = np.array_split(corpus, n_chunks)
        #  first_query = system_prompt + example_prompt + ' I will send a list of documents for topic labelling and summarizing. \
        # As the length of the documents is too long, I will send them in multiple small lists in the format of: \
        # [START PART {}/{}] {} [END PART {}/{}].'
        count = 1
        message = []

        for chunk in chunks:
            query_s = "[INST] I have a topic that contains the following documents: {}\
                        The topic is described by the following keywords: {}.\
                        Based on the information about the topic above, please create a short label of this topic and a summary statement for this topic. Make sure to only return the topic label and a short summary of the topic in the format of:\            Label: .....\
    Summary: .....\[/INST]".format(chunk, kws)
            if count == 1:  # First prompt
                prompt = system_prompt + example_prompt + query_s
            else:
                prompt = system_prompt + query_s
            # Send query here
            msg = generator(prompt)
            print('Chunk #{} - Prompt len: {}'.format(count, len(prompt)))
            print('Response: ', msg[0]['generated_text'].split('[/INST]'))
            count += 1
            message.append(msg[0]['generated_text'].split('[/INST]')[1])

        # print('Generating the final topic label and summary...')
        # print('Response list: len: {} - {}'.format(len('; '.join(message)), message))
        # query_s = "Here are a list of labels and summaries for the same topic: {}.\
        #           The topic is described by the following keywords: {}.\
        #           Based on the above information, please create a final topic label and a short summary to summarize the topic. Do not send back my input. Make sure to only return the label and the summary sentence in the format of:\
        #                       Label: .....\
        #                       Summary: .....\[/INST]".format(message, kws)
        # msg = generator(query_s)
        # print('Final prompt: ', query_s)
        # print('Final response: ', msg[0]['generated_text'].split('[/INST]'))
        # message.append(msg[0]['generated_text'].split('[/INST]')[1])

        return message

def construct_final_llama_query(generator, corpus, kws):
    # System prompt describes information given to all conversations
    system_prompt = """
    <s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant for labeling topics and summarizing them. 
    <</SYS>>
    """

    # Example prompt demonstrating the output we are looking for
    example_prompt = """
    I have a topic that contains the following summarizations:
    - Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
    - Meat, but especially beef, is the word food in terms of emissions.
    - Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

    The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

    Based on the information above, find the common topic among these summarizations, create a topic label and a short topic summary.\
    Make sure to only return the label and the summary sentence.

    [/INST] Label: The Nuances of Meat Consumption.
    Summary: Navigating the historical transition to meat as a staple, recognizing its environmental impact, and acknowledging the nuanced ethics of dietary choices.
    """

    # Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
    main_prompt = """
    [INST]
    I have a topic that contains the following summarizations:
    {}

    The topic is described by the following keywords: {}.

    Based on the information above, find the common topic among these summarizations, create a topic label and a short topic summary.\
    Make sure to only return the label and the summary following this format:
    Label: .....
    Summary: ...
    [/INST]
    """.format(corpus, kws)
    # num_words = sum([len(sentence.split()) for sentence in corpus])

    prompt = system_prompt + main_prompt
    num_words = len(prompt)

    if num_words < 6000:
        msg = generator(prompt)
        print('Response: ', msg[0]['generated_text'].split('[/INST]')[-1])
        return [msg[0]['generated_text'].split('[/INST]')[-1]]
    else:
        n_chunks = math.ceil(num_words / 6000)
        print('Num words is {}. Split into to {} queries'.format(num_words, n_chunks))
        chunks = np.array_split(corpus, n_chunks)
        #  first_query = system_prompt + example_prompt + ' I will send a list of documents for topic labelling and summarizing. \
        # As the length of the documents is too long, I will send them in multiple small lists in the format of: \
        # [START PART {}/{}] {} [END PART {}/{}].'
        count = 1
        message = []

        for chunk in chunks:
            if count == 1:  # First prompt
                query_s = "[INST] I have a topic that contains a list of summarizations.\
                             However, since this list is too long, I will send them in multiple small lists in this format: [START PART {}/{}] xxxx [END PART {}/{}]. \
                             Keep accumulating the list of summarizations as my input. Do not send back my input and do not give me your answer until I say [ALL PARTS ARE SENT]. \
                             Based on the information above, find the common topic among these summarizations, create a topic label and a short topic summary.\
                             Make sure to only return the topic label and a short summary of the topic in the format of:\
                             Label: .....\
                             Summary: .....\
                             Here is the first part of my input: [START PART {}/{}] {} [END PART {}/{}]"\
                    .format(count, len(chunks), count, len(chunks), count, len(chunks), chunk, count, len(chunks))
                prompt = system_prompt + query_s
            elif count == len(chunks):
                query_s = "[START PART {}/{}] {} [END PART {}/{}][ALL PARTS SENT]. The topic is described by the following keywords: {}. \
                            Do not send back my input. Based on the information above, find the common topic among these summarizations to create a topic label and a short topic summary.\
                             Make sure to only return the topic label and a short summary of the topic in the format of:\
                             Label: .....\
                             Summary: .....\[/INST]".format(count, len(chunks), chunk, count, len(chunks), kws)
                prompt = query_s
            else:
                query_s = "[START PART {}/{}] {} [END PART {}/{}]. Do not send back my input. Do not give me your answer until I say [ALL PARTS SENT]."\
                    .format(count, len(chunks), chunk, count, len(chunks), count, len(chunks))
                prompt = query_s
            # Send query here
            msg = generator(prompt)
            print('Chunk #{} - Prompt len: {}'.format(count, len(prompt)))
            print('Response: ', msg[0]['generated_text'].split('[/INST]')[-1])
            count += 1
            message.append(msg[0]['generated_text'].split('[/INST]')[-1])

        # print('Generating the final topic label and summary...')
        # print('Response list: len: {} - {}'.format(len('; '.join(message)), message))
        # query_s = "Here are a list of labels and summaries for the same topic: {}.\
        #           The topic is described by the following keywords: {}.\
        #           Based on the above information, please create a final topic label and a short summary to summarize the topic. Do not send back my input. Make sure to only return the label and the summary sentence in the format of:\
        #                       Label: .....\
        #                       Summary: .....\[/INST]".format(message, kws)
        # msg = generator(query_s)
        # print('Final prompt: ', query_s)
        # print('Final response: ', msg[0]['generated_text'].split('[/INST]'))
        # message.append(msg[0]['generated_text'].split('[/INST]')[1])

        return message



""" Get a list of corpus based on the topic and subtopic that were generated by a specific topic modeling (specified in col_name).
Send a query to chatgpt for a corpus list of each topic and subtopic. Return a list of llama2 responses."""


def summarize_topic_llama2(data, col_name, model_name, filename, final_prompt=0):
    response = []
    if final_prompt == 0:
        # Group the data df by col BB and get all Corpus from a topic generated by BB.
        corpus = get_corpus_from_topic_model(data, col_name)
        # Get a list of (topic, subtopic) from corpus
        topic_subtopic = get_topic_subtopic_list(corpus, col_name)
    else:
        topic_subtopic = data[['Topic', 'Subtopic']].values.tolist()

    print('Topic Subtopic: ', topic_subtopic)
    llama2_generator = load_llama2()

    for element in topic_subtopic:
        # print(element, len(element))
        kw_list = get_kws_from_topic_model(topic=element[0], subtopic=element[1], model_name=model_name)

        if final_prompt == 0:
            # Filter the corpus by topic and subtopic number
            topic_corpus = filter_corpus_per_topic(corpus, element[0], element[1])
            print('Topic and Subtopic {} has {} docs\n'.format(element, len(topic_corpus)))
            response = construct_llama_query(llama2_generator, topic_corpus, kw_list)
        else:
            filter = (data['Topic'] == element[0]) & (data['Subtopic'] == element[1])
            summaries = data.loc[filter, 'Summary'].values.tolist()

            response = construct_final_llama_query(llama2_generator, summaries, kw_list)
        # Save response to a file
        with open(OUT_DIR + filename, 'a') as f:
            f.write('Topic and Subtopic {}:\n \nResponse: {}\n\n'.format(element, response))

        print("GPU Usage:")
        gpu_usage()
        # print("GPU Usage after emptying the cache")
        # torch.cuda.empty_cache()
        gc.collect()
        # gpu_usage()

    return response


DATE = '1121'
# IN_DIR = 'C:/Users/natal/Desktop/Disaster Analysis/Codes/Output/{}/Kws/'.format(DATE)
# IN_DIR = 'C:/Users/natal/Desktop/Disaster Analysis/Codes/Output/{}/'.format(DATE)
# IN_DIR = '/content/'
IN_DIR = 'Output/{}/'.format(DATE)
OUT_DIR = IN_DIR + 'Summaries/'
if not os.path.exists(OUT_DIR):  os.makedirs(OUT_DIR)

if __name__ == '__main__':
    """ This block is for naming and summarizing the topic given a list of kws"""
    # datafile = 'biomed-kw-coherence-{}.xlsx'.format(DATE)

    # datafile = 'df19-subtopics-0627-1.xlsx'; period = 'Pre'
    datafile = 'df22-subtopics-{}-D2V.xlsx'.format(DATE);
    period = 'Post'
    print('Reading data from: {}'.format(IN_DIR + datafile))
    # Read data from excel file
    data = pd.read_excel(IN_DIR + datafile, header=0, engine='openpyxl')
    # col_name = 'Subtopic-BioMed'; model_name = 'BioMed'  # col_name options: Subtopic-BB, Subtopic_Doc2Vec, Subtopic_BioMed or Subtopic_BioMedLM, Subtopic-B
    #col_name = 'Subtopic-Doc2Vec'; model_name = 'D2V'  # model_name options: BB, D2V, biomed or bertopic
    #col_name = 'Subtopic-BB'; model_name = 'BB'
    col_name = 'Subtopic-B'; model_name = 'Bertopic'

    # outfile = 'chatgpt_{}_{}_{}.txt'.format(col_name, period, DATE)
    # response = summarize_topic_chatgpt(data, col_name=col_name, model_name=model_name, filename=outfile)

    outfile =  'llama2_{}_{}_{}.txt'.format(col_name, period, DATE) # 'final_llama2_Subtopic-B_Post_1121.txt' when final_prompt = 1
    #response = summarize_topic_llama2(data, col_name=col_name, model_name=model_name, filename=outfile)

    # Create a str of query that contains all kws from the datafile
    # query = construct_query(corpus)
    # print('Constructed query:\n', query)

    """ Parse Topic Label and Summary from the response """
    # Read in a text file that contains the response
    response_file = open(OUT_DIR + outfile, 'r')
    response_list = response_file.readlines()
    datafile = IN_DIR + '{}_eval_{}.xlsx'.format(model_name, DATE)
    print('Reading data from: {}'.format(datafile))
    data = pd.read_excel(datafile, header=0, engine='openpyxl')
    if 'llama' in outfile:
        labels_summaries_df = parse_llama_response(response_list)
        print(labels_summaries_df, labels_summaries_df.columns)
        """This block is for parsing stage 1 summaries and labels from Llama, and query for the final summaries and labels"""
        labels_summaries_df.to_excel(IN_DIR + '{}-{}-Llama-Labels-Summaries.xlsx'.format(DATE, model_name), index=False)
        #labels_summaries_df = get_topic_kws(labels_summaries_df, model_name)
        outfile = 'final_llama2_{}_{}_{}.txt'.format(col_name, period, DATE)
        response = summarize_topic_llama2(labels_summaries_df, col_name=col_name, model_name=model_name, filename=outfile, final_prompt=1)
        # Query for final labels and summaries for each topic by summarizing the list of labels and summaries

        """This block is for adding the final Llama labels and summaries to the data df"""
        # data['Llama_Final_Label'] = labels_summaries_df['Labels'].apply(lambda x: x[-1]) # Only get the last label from the list of labels
        # data['Llama_Final_Summary'] = labels_summaries_df['Summaries'].apply(lambda x: x[-1]) # Only get the last summary from the list of summaries
        #
        # stage1_labels = pd.read_excel(IN_DIR+'{}-{}-Llama-Labels-Summaries.xlsx'.format(DATE, model_name), header=0, engine='openpyxl')
        # stage1_labels['Llama_Final_Label'] = labels_summaries_df['Labels'] # save all labels from the list of labels. Same thing to summaries.
        # stage1_labels['Llama_Final_Summary'] = labels_summaries_df['Summaries']
        # print('Save datafile to dir {}'.format(datafile))
        # data.to_excel(datafile, index=False, header=True)
        # stage1_labels.to_excel(IN_DIR+'{}-{}-Llama-Labels-Summaries.xlsx'.format(DATE, model_name), index=False)
    elif 'chatgpt' in outfile:
        labels_summaries_df = parse_response(response_list)
        print(data['ChatGPT_Label'].head(10), data['ChatGPT_Summary'].head(10))

    #labels_summaries_df.to_excel(IN_DIR+'{}-{}-Llama-Labels-Summaries.xlsx'.format(DATE, model_name), index=False)
