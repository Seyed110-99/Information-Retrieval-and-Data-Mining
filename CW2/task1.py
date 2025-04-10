import re  
import math  
import json  
import nltk 
import pandas as pd 
from timeit import default_timer as timer  
from nltk.stem import WordNetLemmatizer, SnowballStemmer 
import ssl  

# Download required NLTK resources if not already available.
nltk.download('stopwords')
nltk.download('punkt')

#############################################
# Text Processing Functions
#############################################

def process_raw_text(lines, remove_stopwords=False, lemmatisation=False, stemming=False):
    """
    Tokenizes and preprocesses raw text lines into clean tokens. 
    Supports optional stopword removal, lemmatisation, and stemming.
    
    Args:
        lines (list of str): Input raw text lines to process.
        remove_stopwords (bool): Whether to remove stopwords.
        lemmatisation (bool): Whether to apply lemmatisation.
        stemming (bool): Whether to apply stemming.
    
    Returns:
        list[list[str]]: List of tokenized and processed words per input line.
    """
    tokens = []

    if remove_stopwords:
        stop_words = nltk.corpus.stopwords.words('english')

    if lemmatisation:
        nltk.download('wordnet')
        lemmatiser = WordNetLemmatizer()

    if stemming:
        stemmer = SnowballStemmer("english")

    for line in lines:
        line = line.strip().lower()
        line = re.sub(r"[^a-zA-Z\s]", " ", line)
        words = nltk.word_tokenize(line)

        if remove_stopwords:
            words = [w for w in words if w not in stop_words]
        if lemmatisation:
            words = [lemmatiser.lemmatize(w) for w in words]
        if stemming:
            words = [stemmer.stem(w) for w in words]

        tokens.append(words)
    return tokens

def generate_inverted_index(passage_dict):
    """
    Builds an inverted index from passage tokens.
    
    Args:
        passage_dict (dict): Maps pid to list of tokens.
    
    Returns:
        dict: Inverted index mapping tokens to {pid: frequency}.
    """
    inverted_index_dict = {}
    for pid, tokens in passage_dict.items():
        unique_tokens = set(tokens)
        for token in unique_tokens:
            occurrence = tokens.count(token)
            if token not in inverted_index_dict:
                inverted_index_dict[token] = {pid: occurrence}
            else:
                inverted_index_dict[token][pid] = occurrence
    return inverted_index_dict

#############################################
# BM25 Scoring Function
#############################################

def bm25(qid_pid_df, inv_index_dict, query_dict, passage_dict):
    """
    Calculates BM25 scores for query-passage pairs.
    
    Args:
        qid_pid_df (pd.DataFrame): DataFrame with ['qid', 'pid'] pairs.
        inv_index_dict (dict): Inverted index.
        query_dict (dict): Maps qid to tokenized queries.
        passage_dict (dict): Maps pid to tokenized passages.
    
    Returns:
        pd.DataFrame: ['qid', 'pid', 'relevance_score'], sorted by score.
    """
    k1, k2, b = 1.2, 100, 0.75
    n = len(passage_dict)
    total_dl = sum(len(tokens) for tokens in passage_dict.values())
    avdl = total_dl / n

    output_all = []

    qid_pid_df['qid'] = qid_pid_df['qid'].str.strip()
    qid_pid_df['pid'] = qid_pid_df['pid'].str.strip()

    qids = qid_pid_df['qid'].unique()
    for qid in qids:
        query_tokens = query_dict.get(qid)
        if query_tokens is None:
            continue
        pids = qid_pid_df.loc[qid_pid_df['qid'] == qid, 'pid'].tolist()
        for pid in pids:
            passage_tokens = passage_dict.get(pid)
            if passage_tokens is None:
                continue
            dl = len(passage_tokens)
            K = k1 * ((1 - b) + b * (dl / avdl))
            score = 0.0

            common_tokens = set(query_tokens) & set(passage_tokens)
            for token in common_tokens:
                pid_token_count_dict = inv_index_dict.get(token)
                if not pid_token_count_dict:
                    continue
                ni = len(pid_token_count_dict)
                fi = pid_token_count_dict.get(pid, 0)
                qfi = query_tokens.count(token)
                idf = math.log(1 / ((ni + 0.5) / (n - ni + 0.5)))
                term_score = idf * ((k1 + 1) * fi / (K + fi)) * ((k2 + 1) * qfi / (k2 + qfi))
                score += term_score

            output_all.append((qid, pid, score))

    df_bm25 = pd.DataFrame(output_all, columns=['qid', 'pid', 'score'])
    df_bm25.rename(columns={'score': 'relevance_score'}, inplace=True)
    df_bm25.sort_values(by=['qid', 'relevance_score'], ascending=[True, False], inplace=True)
    return df_bm25

#############################################
# Evaluation Metrics
#############################################

def compute_map(pred_dict, relevance_dict):
    """
    Computes Mean Average Precision (MAP).
    
    Args:
        pred_dict (dict): Predicted scores: {qid: {pid: score}}.
        relevance_dict (dict): Ground truth: {qid: {pid: "1.0"}}.
    
    Returns:
        float: MAP score.
    """
    ap_sum = 0
    num_queries = 0

    for qid, scores in pred_dict.items():
        if qid not in relevance_dict:
            continue
        ranked_pids = [pid for pid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        num_relevant = 0
        ap = 0
        total_relevant = len(relevance_dict[qid])

        for rank, pid in enumerate(ranked_pids, start=1):
            if relevance_dict[qid].get(pid) == "1.0":
                num_relevant += 1
                ap += num_relevant / rank
        if total_relevant > 0:
            ap /= total_relevant
            ap_sum += ap
            num_queries += 1
    return ap_sum / num_queries if num_queries > 0 else 0

def compute_ndcg(pred_dict, relevance_dict):
    """
    Computes Normalized Discounted Cumulative Gain (NDCG).
    
    Args:
        pred_dict (dict): Predicted scores: {qid: {pid: score}}.
        relevance_dict (dict): Ground truth: {qid: {pid: "1.0"}}.
    
    Returns:
        float: NDCG score.
    """
    ndcg_sum = 0
    num_queries = 0

    for qid, scores in pred_dict.items():
        if qid not in relevance_dict:
            continue
        ranked_pids = [pid for pid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        dcg = 0
        for rank, pid in enumerate(ranked_pids, start=1):
            if relevance_dict[qid].get(pid) == "1.0":
                dcg += 1 / math.log2(rank + 1)
        total_relevant = len(relevance_dict[qid])
        idcg = sum(1 / math.log2(rank + 1) for rank in range(1, total_relevant + 1))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_sum += ndcg
        num_queries += 1
    return ndcg_sum / num_queries if num_queries > 0 else 0

#############################################
# Main Processing and Evaluation
#############################################

if __name__ == "__main__":
    # Handle SSL context for nltk downloads if necessary.
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    start_time = timer()
    pd.options.mode.copy_on_write = True

    # Load validation data
    raw_validation_df = pd.read_csv("validation_data.tsv", sep="\t", dtype="string")
    raw_validation_df['qid'] = raw_validation_df['qid'].str.strip()
    raw_validation_df['pid'] = raw_validation_df['pid'].str.strip()

    # Process passages
    raw_validation_passages = raw_validation_df['passage']
    validation_passage_tokens_list = process_raw_text(raw_validation_passages, remove_stopwords=True, stemming=True)
    validation_pid_passage_dict = dict(zip(raw_validation_df['pid'].tolist(), validation_passage_tokens_list))
    print("Finished processing validation passages.")

    # Build inverted index
    validation_inverted_index = generate_inverted_index(validation_pid_passage_dict)

    # Process queries
    raw_validation_queries = raw_validation_df["queries"]
    validation_query_tokens_list = process_raw_text(raw_validation_queries, remove_stopwords=True, stemming=True)
    validation_qid_query_dict = dict(zip(raw_validation_df['qid'].tolist(), validation_query_tokens_list))
    print("Finished processing validation queries.")

    # Get query-passage pairs
    df_qid_pid = raw_validation_df.loc[:, ["qid", "pid"]].drop_duplicates()

    # Run BM25 scoring
    df_bm25 = bm25(df_qid_pid, validation_inverted_index, validation_qid_query_dict, validation_pid_passage_dict)
    print("BM25 scoring completed. Sample results:")
    print(df_bm25.head())

    # Convert BM25 results to dict
    bm25_dict = (
        df_bm25.groupby("qid")
        .apply(lambda x: dict(zip(x["pid"], x["relevance_score"])), include_groups=False)
        .to_dict()
    )

    # Get ground truth relevance
    validation_relevance_dict = (
        raw_validation_df.loc[raw_validation_df["relevancy"] == "1.0"]
        .groupby("qid")
        .apply(lambda x: dict(zip(x["pid"], x["relevancy"])), include_groups=False)
        .to_dict()
    )

    # Compute metrics
    bm25_map = compute_map(bm25_dict, validation_relevance_dict)
    bm25_ndcg = compute_ndcg(bm25_dict, validation_relevance_dict)

    end_time = timer()
    time_taken = end_time - start_time

    print("BM25 MAP:", bm25_map)
    print("BM25 NDCG:", bm25_ndcg)
    print(f"Task 1 Process time: {round(time_taken / 60, 1)} minutes")


