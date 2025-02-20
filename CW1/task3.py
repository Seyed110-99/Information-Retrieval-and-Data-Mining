import os
import pandas as pd
import numpy as np
import math
import json
from timeit import default_timer as timer
import task1 as t1  # Use the preprocessing function from task1.py
from collections import Counter

def compute_passage_idf(passages, inv_index):
    """
    Computes the inverse document frequency (IDF) for each term in the passage collection.
    The formula applied is: idf = log10(total_docs / (1 + document_frequency)).
    """
    idf_values = {}
    total_docs = len(passages)
    for term, doc_dict in inv_index.items():
        df = len(doc_dict)  # number of passages containing the term
        idf_values[term] = math.log10(total_docs / (df))
    return idf_values


def compute_passage_tf_idf(passages, inv_index, idf_values):
    """
    Computes TF-IDF vectors for each passage.
    For each passage, the term frequency (TF) is calculated as count(term) / total terms,
    then multiplied by the term's IDF value.
    """
    tf_idf_passages = {}
    for pid, tokens in passages.items():
        total_terms = len(tokens)
        if total_terms == 0:
            continue  # Skip empty passages
        counts = Counter(tokens)
        tf_idf_passages[pid] = {term: (count / total_terms) * idf_values.get(term, 0)
                                for term, count in counts.items()}
    return tf_idf_passages


def compute_query_tf_idf(query_dict, idf_values):
    """
    Computes TF-IDF vectors for queries using the IDF values computed from passages.
    """
    tf_idf_queries = {}
    for qid, tokens in query_dict.items():
        total = len(tokens)
        if total == 0:
            continue
        counts = Counter(tokens)
        tf_idf_queries[qid] = {term: (count / total) * idf_values.get(term, 0)
                               for term, count in counts.items()}
    return tf_idf_queries


def rank_by_cosine_similarity(query_tf_idf, passage_tf_idf, candidate_df, query_df):
    """
    For each query, computes the cosine similarity between its TF-IDF vector and those of candidate passages.
    Returns a DataFrame of the top 100 (qid, pid, similarity) rows per query.
    """
    results = []
    for qid in query_df['qid']:
        q_vec = query_tf_idf.get(qid, {})
        q_terms = set(q_vec.keys())
        candidate_pids = candidate_df[candidate_df['qid'] == qid]['pid'].tolist()
        sim_list = []
        for pid in candidate_pids:
            p_vec = passage_tf_idf.get(pid, {})
            common = q_terms.intersection(p_vec.keys())
            dot = sum(q_vec[t] * p_vec[t] for t in common)
            norm_q = np.linalg.norm(list(q_vec.values()))
            norm_p = np.linalg.norm(list(p_vec.values()))
            similarity = dot / (norm_q * norm_p) if norm_q and norm_p else 0
            sim_list.append((qid, pid, similarity))
        # Keep top 100 for the query
        sim_list = sorted(sim_list, key=lambda x: x[2], reverse=True)[:100]
        results.extend(sim_list)
    return pd.DataFrame(results, columns=['qid', 'pid', 'similarity'])


def rank_by_bm25(query_tf_idf, passages, inv_index, candidate_df):
    """
    For each query, computes BM25 scores for candidate passages.
    BM25 parameters: k1 = 1.2, k2 = 100, b = 0.75.
    Returns a DataFrame of the top 100 ranked (qid, pid, score) rows per query.
    """
    total_docs = len(passages)
    avg_doc_length = np.mean([len(tokens) for tokens in passages.values()])
    k1 = 1.2
    k2 = 100
    b = 0.75
    results = []
    for qid, q_vec in query_tf_idf.items():
        q_terms = set(q_vec.keys())
        candidate_pids = candidate_df[candidate_df['qid'] == qid]['pid'].tolist()
        for pid in candidate_pids:
            p_tokens = passages.get(pid, [])
            doc_len = len(p_tokens)
            bm25_score = 0
            for term in q_terms:
                # Retrieve term frequency from the inverted index structure
                tf = inv_index.get(term, {}).get(pid, {}).get('freq', 0)
                df = len(inv_index.get(term, {}))
                idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
                # qfi is set to 1 assuming unique query terms
                qfi = 1
                K = k1 * ((1 - b) + b * (doc_len / avg_doc_length))
                bm25_score += idf * ((tf * (k1 + 1)) / (tf + K)) * ((k2 + 1) * qfi / (k2 + qfi))
            results.append((qid, pid, bm25_score))
    # For each query, select the top 100 ranked passages.
    df_scores = pd.DataFrame(results, columns=['qid', 'pid', 'score'])
    final_scores = pd.DataFrame()
    for q in df_scores['qid'].unique():
        top_q = df_scores[df_scores['qid'] == q].sort_values(by='score', ascending=False).head(100)
        final_scores = pd.concat([final_scores, top_q])
    return final_scores


if __name__ == "__main__":
    start = timer()

    # Load preprocessed passages and inverted index from task2 output files.
    with open("processed_passage.json", "r") as f:
        processed_passages = json.load(f)
    with open("inverted_index.json", "r") as f:
        inverted_index = json.load(f)

    # Compute passage IDF using the new function.
    passage_idf = compute_passage_idf(processed_passages, inverted_index)

    # Read test queries from file.
    query_df = pd.read_csv("test-queries.tsv", sep="\t", names=["qid", "query"], dtype="string")
    query_dict = {}
    for _, row in query_df.iterrows():
        qid = row["qid"]
        query_text = row["query"]
        # Process queries: remove stopwords and apply stemming (without lemmatization) to mimic the original.
        tokens = t1.preprocess_text(query_text, remove_stop=True, do_lemmatize=False, do_stem=True)
        query_dict[qid] = tokens

    # Load candidate mapping: a file with columns qid, pid, query, passage.
    candidate_df = pd.read_csv("candidate-passages-top1000.tsv", sep="\t",
                               names=["qid", "pid", "query", "passage"],
                               dtype={"qid": "string", "pid": "string"}).loc[:, ["qid", "pid"]]

    # Compute TF-IDF representations.
    passage_tf_idf = compute_passage_tf_idf(processed_passages, inverted_index, passage_idf)
    query_tf_idf = compute_query_tf_idf(query_dict, passage_idf)

    # Compute cosine similarity-based ranking and save the results to tfidf.csv.
    tfidf_results = rank_by_cosine_similarity(query_tf_idf, passage_tf_idf, candidate_df, query_df)
    tfidf_results.to_csv("tfidf.csv", index=False, header=False)

    # Compute BM25 ranking and save the results to bm25.csv.
    bm25_results = rank_by_bm25(query_tf_idf, processed_passages, inverted_index, candidate_df)
    bm25_results.to_csv("bm25.csv", index=False, header=False)

    # Save the processed queries for later use.
    with open("processed_query.json", "w") as f:
        json.dump(query_dict, f, indent=4)

    end = timer()
    print(f"Process completed in {(end - start) / 60:.2f} minutes.")
