""" Topic_Evaluation.py
Functionalities:
    1. Calculate the number of Yes in agreement per topic by both raters. (ChatGPT summary and Llama2 summary)
    2. Calculate the Cohen's Kappa score of the topic evaluation results from 2 different raters
    3. Find out which LLM (GPT or Llama2) generates better topic summary
    """

import pandas as pd
import numpy as np
import re
from sklearn.metrics import cohen_kappa_score

DATA_DIR = 'C:/Users/natal/Desktop/Disaster Analysis/Codes/Output/1121/Comprehensive Eval/'
DATE = '0229'

""" Calculate the number of Yes in agreement, and No in agreement per topic by both raters. (ChatGPT summary and Llama2 summary) """
def calculate_agreement(column1, column2):
    if len(column1) != len(column2):
        raise ValueError("Columns must have the same length")

    total_rows = len(column1)
    agreement_1 = sum(1 for a, b in zip(column1, column2) if a == b == 1)
    agreement_0 = sum(1 for a, b in zip(column1, column2) if a == b == 0)

    return agreement_1, agreement_0 # Return the number of Yes in agreement, and No in agreement

# Load the topic evaluation results
def main():
    # Read in topic eval excel with multiple sheets
    topic_list = [27, 31, 23, 22, 29, 30, 25, 20, 19, 24, 26, 32]
    eval_list = []  # List of topic evaluation results. Each element is a dataframe that contains the evaluation results for a topic
    for topic_number in topic_list:
        topic_eval = pd.read_excel(DATA_DIR + 'Topic-Docs-Eval.xlsx', sheet_name='Topic-{}'.format(topic_number))
        print('read in:\n', topic_eval)
        # Read in topic number
        eval = pd.DataFrame(columns=['Topic', 'GPT-A', 'Llama-A', 'GPT-B', 'Llama-B'])
        print('Length of topic eval:', len(topic_eval))
        print('Topic number:', topic_number)
        eval['Topic'] = np.ones(len(topic_eval), dtype=int) * topic_number

        # Read in rating from both raters for GPT summary
        eval['GPT-A'] = topic_eval['GPT-Summary (Yes/No/Maybe) - Sai'].apply(
            lambda x: 1 if re.sub("[^0-9a-zA-Z\s]+", "", x.lower().strip()) == 'yes' else 0)
        eval['GPT-B'] = topic_eval['GPT-Summary (Yes/No/Maybe) - Illa'].apply(
            lambda x: 1 if re.sub("[^0-9a-zA-Z\s]+", "", x.lower().strip()) == 'yes' else 0)

        # Read in rating from both raters for Llama summary
        eval['Llama-A'] = topic_eval['Llama-Summary (Yes/No/Maybe) - Sai'].apply(
            lambda x: 1 if re.sub("[^0-9a-zA-Z\s]+", "", x.lower().strip()) == 'yes' else 0)
        eval['Llama-B'] = topic_eval['Llama-Summary (Yes/No/Maybe) - Illa'].apply(
            lambda x: 1 if re.sub("[^0-9a-zA-Z\s]+", "", x.lower().strip()) == 'yes' else 0)

        print('eval:\n', eval)
        eval_list.append(eval)

    topic_eval_df = pd.concat(eval_list)
    print('topic eval df:\n', topic_eval_df)
    topic_eval_df = pd.concat(eval_list)
    print('topic eval df:\n', topic_eval_df)

    #result_eval_df = pd.DataFrame(columns=['Topic','Yes-GPT', 'No-GPT', 'Yes-Llama', 'No-Llama', 'Kappa_score'])
    #result_eval_df['Topic'] = topic_list
    # Count number of Yes in agreement per topic by both raters in GPT
    result_gpt_series = topic_eval_df.groupby('Topic').apply(lambda x: calculate_agreement(x['GPT-A'], x['GPT-B']))
    result_gpt = pd.DataFrame(result_gpt_series.values.tolist(), columns=['Yes-GPT', 'No-GPT'])
    result_gpt['Kappa_GPT'] = topic_eval_df.groupby('Topic').apply(lambda x: round(cohen_kappa_score(x['GPT-A'], x['GPT-B']), 2)).values.tolist()
    result_gpt['Topic'] = result_gpt_series.index
    # Number of documents per topic
    result_gpt['Num_Docs'] = topic_eval_df.groupby('Topic').size().reset_index(name='Num_Docs')['Num_Docs']
    print('result gpt:\n', result_gpt)

    # Count number of Yes in agreement per topic by both raters in Llama
    result_llama_series = topic_eval_df.groupby('Topic').apply(lambda x: calculate_agreement(x['Llama-A'], x['Llama-B']))
    result_llama = pd.DataFrame(result_llama_series.values.tolist(), columns=['Yes-Llama', 'No-Llama'])
    result_llama['Kappa_Llama'] = topic_eval_df.groupby('Topic').apply(lambda x: round(cohen_kappa_score(x['Llama-A'], x['Llama-B']), 2)).values.tolist()
    result_llama['Topic'] = result_llama_series.index
    print('result llama:\n', result_llama)

    result_eval_df = result_gpt.join(result_llama.set_index('Topic'), on='Topic')

    # Calculate the probability of Yes in agreement and No in agreement in GPT
    result_eval_df['Prob-Yes-GPT'] = round(result_eval_df['Yes-GPT'] / result_eval_df['Num_Docs'], 2)
    result_eval_df['Prob-No-GPT'] = round(result_eval_df['No-GPT'] / result_eval_df['Num_Docs'], 2)

    # Calculate the probability of Yes in agreement and No in agreement in Llama
    result_eval_df['Prob-Yes-Llama'] = round(result_eval_df['Yes-Llama'] / result_eval_df['Num_Docs'], 2)
    result_eval_df['Prob-No-Llama'] = round(result_eval_df['No-Llama'] / result_eval_df['Num_Docs'], 2)

    result_eval_df = result_eval_df[['Topic','Num_Docs', 'Yes-GPT','Prob-Yes-GPT', 'No-GPT','Prob-No-GPT', 'Kappa_GPT','Yes-Llama','Prob-Yes-Llama','No-Llama','Prob-No-Llama','Kappa_Llama']]
    print('result eval df:\n', result_eval_df.to_string())

    # Save the results to an excel file
    print('Saving results to excel: Topic-Eval-Qual-Results.xlsx')
    result_eval_df.to_excel(DATA_DIR + 'Topic-Eval-Qual-Results-{}.xlsx'.format(DATE), index=False)

if __name__ == '__main__':
    main()