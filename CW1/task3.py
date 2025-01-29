import os
import pandas as pd
import numpy as np
import math
import json
from timeit import default_timer as timer
import task1 as t1
from collections import Counter

def passage_idf_gen(processed_passage_dict, inverted_index):
    """
    Compute Inverse Document Frequency (IDF) for each term in the passage collection.
    """
    idf_passage = {}
    num_passages = len(processed_passage_dict)

    for word in inverted_index:
        df_t = len(inverted_index[word])  # Number of passages containing the word
        idf_passage[word] = math.log10(num_passages / (1 + df_t))  # Apply IDF formula

    return idf_passage


def passage_tf_idf_gen(processed_passage_dict, inverted_index, idf_passage): 
    """
    Compute TF-IDF for passages.
    """
    tf_idf = {}

    for pid, tokens in processed_passage_dict.items():
        tf_idf[pid] = {}
        total_terms = len(tokens)

        if total_terms == 0:
            continue  # Skip empty passages

        term_counts = Counter(tokens)  # Count occurrences of terms

        for token, count in term_counts.items():
            tf = count / total_terms  # Compute Term Frequency (TF)
            idf = idf_passage.get(token, 0)  # Use precomputed IDF, default to 0
            tf_idf[pid][token] = tf * idf  # Compute TF-IDF

    return tf_idf


def query_idf_tf_gen(qid_query_dict, idf_passage):
    """
    Compute TF-IDF for queries using IDF from passages.
    """
    tf_idf_query = {}

    for qid, tokens in qid_query_dict.items():
        tf_idf_query[qid] = {}  
        total_terms = len(tokens)

        if total_terms == 0:
            continue  

        term_counts = Counter(tokens)  

        for token, count in term_counts.items():
            tf = count / total_terms  # Compute TF
            idf = idf_passage.get(token, 0)  # Use IDF from passages
            tf_idf_query[qid][token] = tf * idf  # Compute TF-IDF

    return tf_idf_query


def cosine_similarity(query_idf_tf, passage_idf_tf, pid_qid_df, query_df):
    """
    Compute cosine similarity between query TF-IDF vectors and passage TF-IDF vectors.
    """
    cosine_df = pd.DataFrame()
    all_qid = query_df['qid'].tolist()

    for qid in all_qid:
        query_tfidf = query_idf_tf.get(qid, {})  # Get query TF-IDF vector
        query_tokens = set(query_tfidf.keys())  # Extract query terms
        output = []

        all_pid = pid_qid_df.loc[pid_qid_df['qid'] == qid, 'pid'].tolist()
        for pid in all_pid:
            passage_tfidf = passage_idf_tf.get(pid, {})  # Get passage TF-IDF vector
            passage_tokens = set(passage_tfidf.keys())

            # Find common terms
            common_tokens = query_tokens & passage_tokens

            # Initialize numerator for dot product calculation
            numerator = 0
            for token in common_tokens:
                passage_tf_idf = passage_tfidf.get(token, 0)  
                query_tfidf_value = query_tfidf.get(token, 0)  
                numerator += passage_tf_idf * query_tfidf_value  

            # Compute vector norms
            query_norm = np.linalg.norm(list(query_tfidf.values()))
            passage_norm = np.linalg.norm(list(passage_tfidf.values()))

            # Compute cosine similarity (avoid division by zero)
            similarity = numerator / (query_norm * passage_norm) if query_norm and passage_norm else 0

            output.append((qid, pid, similarity))

        # Convert results to DataFrame
        cosine_df_result = pd.DataFrame(output, columns=['qid', 'pid', 'similarity'])

        # Keep only top 100 ranked passages per query
        cosine_df_100 = cosine_df_result.sort_values(by=['similarity'], ascending=False).head(100)

        cosine_df = pd.concat([cosine_df, cosine_df_100])

    return cosine_df

def bm25_ranking(query_idf_tf, passage_idf_tf, pid_qid_df, processed_passage_dict, inverted_index):
    """
    Compute BM25 ranking for each query-passage pair using the inverted index and passage_idf_tf.
    """
    avgdl = sum(len(tokens) for tokens in processed_passage_dict.values()) / len(processed_passage_dict)
    
    bm25_df = pd.DataFrame()
    k1 = 1.2  # BM25 term frequency parameter
    k2 = 100  # BM25 query frequency parameter
    b = 0.75  # BM25 document length normalization parameter
    n = len(processed_passage_dict)  # Total number of passages

    for qid, query_tfidf in query_idf_tf.items():
        query_tokens = set(query_tfidf.keys())  # Extract query terms
        output = []

        all_pid = pid_qid_df.loc[pid_qid_df['qid'] == qid, 'pid'].tolist()
        for pid in all_pid:
            passage_tokens = processed_passage_dict.get(pid, [])  # Get passage words
            passage_length = len(passage_tokens)  # Document length

            bm25_score = 0
            for token in query_tokens:
                tf = inverted_index.get(token, {}).get(pid, 0)  # Get TF from inverted index
                ni = len(inverted_index.get(token, {}))  # Number of passages containing the term
                idf = math.log((n - ni + 0.5) / (ni + 0.5) + 1)  # Compute IDF
                qfi = list(query_tfidf.keys()).count(token)  # Convert set to list before counting

                # Compute BM25 denominator
                k = k1 * ((1 - b) + b * (passage_length / avgdl))

                # Compute BM25 score contribution
                bm25_score += idf * ((tf * (k1 + 1)) / (tf + k)) * ((k2 + 1) * qfi / (k2 + qfi))

            output.append((qid, pid, bm25_score))

        # Convert results to DataFrame
        bm25_df_result = pd.DataFrame(output, columns=['qid', 'pid', 'score'])

        # Keep only top 100 ranked passages per query
        bm25_df_100 = bm25_df_result.sort_values(by=['score'], ascending=False).head(100)

        bm25_df = pd.concat([bm25_df, bm25_df_100])

    return bm25_df

if __name__ == "__main__":
    start = timer()

    # Load processed passage file
    with open("processed_passage.json", 'r') as f:
        processed_passage_dict = json.load(f)

    # Load inverted index
    with open("inverted_index.json", 'r') as f:
        inverted_index = json.load(f)

    # Compute IDF for passages
    passageidf = passage_idf_gen(processed_passage_dict, inverted_index)

    # Load query file and tokenize queries individually
    query_df = pd.read_csv("test-queries.tsv", sep='\t', names=['qid', 'query'], dtype='string')

    qid_query_dict = {}
    for _, row in query_df.iterrows():
        qid = row['qid']
        query_text = row['query']
        tokens = t1.tokenise_txt(query_text, remove_stopwords=True, stem=True)
        qid_query_dict[qid] = tokens

    # Load query-to-passage mappings
    pid_qid_df = pd.read_csv("candidate-passages-top1000.tsv", sep='\t', names=['qid', 'pid', 'query', 'passage'],
                             dtype={'qid': 'string', 'pid': 'string'}).loc[:, ['qid', 'pid']]

    # Compute TF-IDF for passages and queries
    passage_tfidf = passage_tf_idf_gen(processed_passage_dict, inverted_index, passageidf)
    query_tfidf = query_idf_tf_gen(qid_query_dict, passageidf)

    # Compute cosine similarity
    tfidf = cosine_similarity(query_tfidf, passage_tfidf, pid_qid_df, query_df)

    # Save top-ranked passages in tfidf.csv
    tfidf.to_csv("tfidf.csv", mode='w', index=False, header=False)

    # Compute BM25 ranking
    bm25_df = bm25_ranking(query_tfidf, passageidf, pid_qid_df, processed_passage_dict, inverted_index)

    # Save top-ranked passages in bm25.csv
    bm25_df.to_csv("bm25.csv", mode='w', index=False, header=False)

    end = timer()
    print(f"Process completed in {(((end - start))/60):.2f} minutes.")
