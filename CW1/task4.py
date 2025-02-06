import os
import pandas as pd
import numpy as np
from timeit import default_timer as timer
import math
import json

def compute_query_likelihood_scores(query_df, candidate_df, passages, queries, inv_index, smoothing='laplace'):
    """
    Computes the query likelihood scores (i.e. log-probabilities) for each query-passage pair
    using the specified smoothing method.

    Parameters:
      - query_df: DataFrame with columns ['qid', 'query'].
      - candidate_df: DataFrame with columns ['qid', 'pid'] linking queries with candidate passages.
      - passages: Dictionary mapping passage IDs to lists of tokens.
      - queries: Dictionary mapping query IDs to lists of tokens.
      - inv_index: Inverted index (mapping each token to a dict of {pid: { "freq": frequency, "positions": [...] }}).
      - smoothing: Smoothing method ('laplace', 'lidstone', or 'dirichlet').

    Returns:
      A DataFrame with columns ['qid', 'pid', 'score'] containing the top 100 ranked passages for each query.
    """
    output_rows = []
    # Smoothing parameters:
    epsilon = 0.1  # for Lidstone smoothing
    mu = 50        # for Dirichlet smoothing
    min_prob = 1e-10  # to avoid log(0)
    
    if smoothing in ['laplace', 'lidstone']:
        V = len(inv_index)  # vocabulary size
        if V == 0:
            raise ValueError("Vocabulary size is zero; check the inverted index.")
    elif smoothing == 'dirichlet':
        corpus_size = sum(len(tokens) for tokens in passages.values())
        if corpus_size == 0:
            raise ValueError("Corpus size is zero; check processed passages.")
    else:
        raise ValueError(f"Unsupported smoothing method: {smoothing}")
    
    # Process each query in the order given by query_df.
    for qid in query_df['qid'].tolist():
        q_tokens = queries.get(qid, [])
        # Retrieve candidate passage IDs for this query.
        cand_pids = candidate_df[candidate_df['qid'] == qid]['pid'].tolist()
        query_results = []
        for pid in cand_pids:
            p_tokens = passages.get(pid, [])
            doc_len = len(p_tokens)
            if doc_len == 0:
                continue  # Skip passages with no tokens.
            log_probability = 0.0
            for token in q_tokens:
                freq = p_tokens.count(token)
                if smoothing == 'laplace':
                    prob = (freq + 1) / (doc_len + V)
                elif smoothing == 'lidstone':
                    prob = (freq + epsilon) / (doc_len + epsilon * V)
                elif smoothing == 'dirichlet':
                    token_postings = inv_index.get(token, {})
                    coll_freq = sum(posting.get("freq", 0) for posting in token_postings.values())
                    prob = (freq + mu * (coll_freq / corpus_size)) / (doc_len + mu)
                # Avoid zero probabilities.
                prob = max(prob, min_prob)
                log_probability += math.log(prob)
            query_results.append((qid, pid, log_probability))
        # For this query, sort the candidate passages by score in descending order
        # and retain only the top 100.
        query_results_sorted = sorted(query_results, key=lambda x: x[2], reverse=True)[:100]
        output_rows.extend(query_results_sorted)
    
    return pd.DataFrame(output_rows, columns=['qid', 'pid', 'score'])


if __name__ == "__main__":
    start = timer()
    
    # Load preprocessed passage data from task2.
    with open("processed_passage.json", "r") as f:
        processed_passages = json.load(f)
    
    # Load processed query data (produced in task3).
    with open("processed_query.json", "r") as f:
        processed_queries = json.load(f)
    
    # Load the inverted index from task2.
    with open("inverted_index.json", "r") as f:
        inverted_index = json.load(f)
    
    # Read test queries.
    query_df = pd.read_csv("test-queries.tsv", sep='\t', names=['qid', 'query'], dtype="string")
    
    # Read candidate passage mappings (qid and pid).
    candidate_df = pd.read_csv("candidate-passages-top1000.tsv", sep='\t',
                               names=['qid', 'pid', 'query', 'passage'],
                               dtype={'qid': 'string', 'pid': 'string'}).loc[:, ['qid', 'pid']]
    
    # Define smoothing methods and corresponding output filenames.
    smoothing_list = ['laplace', 'lidstone', 'dirichlet']
    out_files = ["laplace.csv", "lidstone.csv", "dirichlet.csv"]
    
    # Compute and save query likelihood scores for each smoothing method.
    for method, out_file in zip(smoothing_list, out_files):
        scores_df = compute_query_likelihood_scores(query_df, candidate_df, processed_passages, processed_queries, inverted_index, smoothing=method)
        # Added empirical outcome summary: print average log-probability score.
        avg_score = scores_df['score'].mean()
        print(f"Smoothing method: {method}, Average log-probability score: {avg_score:.6f}")
        scores_df.to_csv(out_file, index=False, header=False)
    
    end = timer()
    print(f"Task 4 Process time: {((end - start)/60):.2f} minutes")
