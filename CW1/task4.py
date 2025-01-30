import os
import pandas as pd
import numpy as np
from timeit import default_timer as timer
import math
import json

def query_likelihood_computation(query_df, pid_qid_df, processed_passage_dict, processed_query_dict, inverted_index, smoothing='laplace'):
    """
    Computes query likelihood scores for given queries and passages using different smoothing techniques.

    Parameters:
    - query_df: DataFrame containing query IDs and text
    - pid_qid_df: DataFrame containing mappings of passage IDs to queries
    - processed_passage_dict: Dictionary of preprocessed passages
    - processed_query_dict: Dictionary of preprocessed queries
    - inverted_index: Inverted index storing term occurrences in passages
    - smoothing: Smoothing method ('laplace', 'lidstone', or 'dirichlet')

    Returns:
    - DataFrame containing ranked passage scores for each query
    """

    df = pd.DataFrame()
    qids = query_df['qid'].tolist()

    # Smoothing parameters
    epsilon = 0.1  # Used for Lidstone Smoothing
    mu = 50  # Used for Dirichlet Smoothing
    min_prob = 1e-10  # Prevents log(0) error

    # Compute vocabulary size (V) for Laplace and Lidstone
    if smoothing in ['laplace', 'lidstone']:
        v = len(inverted_index)
        if v == 0:
            raise ValueError("Vocabulary size is zero. Check the inverted index.")

    # Compute total number of words in the collection (C) for Dirichlet Smoothing
    elif smoothing == 'dirichlet':
        c = sum(len(passage_tokens) for passage_tokens in processed_passage_dict.values())
        if c == 0:
            raise ValueError("Corpus size is zero. Check passage processing.")

    else:
        raise ValueError(f"Invalid smoothing method: {smoothing}")

    # Iterate through each query
    for qid in qids:
        output = []

        # Retrieve query tokens
        query_tokens = processed_query_dict.get(qid, [])

        # Retrieve candidate passages for the query
        pids = pid_qid_df.loc[pid_qid_df['qid'] == qid, 'pid'].tolist()

        for pid in pids:
            passage_tokens = processed_passage_dict.get(pid, [])
            doc_length = len(passage_tokens)  # Number of words in passage

            # Skip empty passages
            if doc_length == 0:
                continue

            score = 0  # Initialize log probability score

            for token in query_tokens:
                token_freq = passage_tokens.count(token)  # Frequency of token in passage

                if smoothing == 'laplace':
                    # Laplace Smoothing: (f(w, d) + 1) / (|d| + V)
                    prob = (token_freq + 1) / (doc_length + v)

                elif smoothing == 'lidstone':
                    # Lidstone Smoothing: (f(w, d) + ε) / (|d| + εV)
                    prob = (token_freq + epsilon) / (doc_length + epsilon * v)

                elif smoothing == 'dirichlet':
                    # Dirichlet Smoothing: ((f(w, d) + μ P(w|C)) / (|d| + μ))
                    collection_freq = sum(inverted_index.get(token, {}).values())  # Token occurrences in corpus
                    prob = (token_freq + mu * (collection_freq / c)) / (doc_length + mu)

                # Ensure probability is never zero or negative
                prob = max(prob, min_prob)

                # Accumulate log probability
                score += math.log(prob)

            output.append((qid, pid, score))

        # Convert results into a DataFrame
        df_result = pd.DataFrame(data=output, columns=['qid', 'pid', 'score'])

        # Keep only the top 100 results per query
        df_top_100 = df_result.sort_values(by='score', ascending=False).head(100)
        df = pd.concat([df, df_top_100], ignore_index=True)

    return df


if __name__ == "__main__":
    start = timer()

    # Load preprocessed passage data
    with open("processed_passage.json", 'r') as f:
        processed_passage_dict = json.load(f)

    # Load preprocessed query data
    with open("processed_query.json", 'r') as f:
        processed_query_dict = json.load(f)

    # Load inverted index
    with open("inverted_index.json", 'r') as f:
        inverted_index = json.load(f)

    # Read test queries
    query_df = pd.read_csv("test-queries.tsv", sep='\t', names=['qid', 'query'], dtype='string')

    # Read candidate passages
    pid_qid_df = pd.read_csv("candidate-passages-top1000.tsv", sep='\t', names=['qid', 'pid', 'query', 'passage'],
                             dtype={'qid': 'string', 'pid': 'string'}).loc[:, ['qid', 'pid']]

    # Compute query likelihood scores for each smoothing method and save results
    smoothing_methods = ['laplace', 'lidstone', 'dirichlet']
    filenames = ["laplace.csv", "lidstone.csv", "dirichlet.csv"]

    for smoothing, filename in zip(smoothing_methods, filenames):
        df_results = query_likelihood_computation(query_df, pid_qid_df, processed_passage_dict, processed_query_dict, inverted_index, smoothing)
        df_results.to_csv(filename, index=False, header=False)

    end = timer()
    print(f"Task 4 Process time: {((end - start)/60):.2f} minutes")
