
import os
import random
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import xgboost as xgb

# Import required functions from Task 1 and Task 2.
from task1 import process_raw_text, compute_map, compute_ndcg, bm25, generate_inverted_index
from task2 import down_sample_by_query, load_glove_embeddings, compute_average_embedding

# -------------------------------
# Base Feature Functions (Unchanged)
# -------------------------------
def create_feature_vector_simple(query_emb, passage_emb):
    """
    Create a simple feature vector for a query–passage pair by concatenating:
      - A bias term (1)
      - Query embedding
      - Passage embedding
    """
    if isinstance(query_emb, tuple):
        query_emb = query_emb[0]
    if isinstance(passage_emb, tuple):
        passage_emb = passage_emb[0]
    return np.concatenate(([1], query_emb, passage_emb))

def build_feature_matrix(df, glove_embeddings, embedding_dim=100, fallback=True):
    """
    Build a design matrix X and label vector y from a DataFrame using simple features.
    Uses a simple feature vector consisting of a bias term, query embedding, and passage embedding.
    """
    X_list = []
    y_list = []
    for idx, row in df.iterrows():
        q_tokens = process_raw_text([row["queries"]], remove_stopwords=True, stemming=True)[0]
        p_tokens = process_raw_text([row["passage"]], remove_stopwords=True, stemming=True)[0]
        q_emb, _, _ = compute_average_embedding(q_tokens, glove_embeddings, embedding_dim, fallback=fallback)
        p_emb, _, _ = compute_average_embedding(p_tokens, glove_embeddings, embedding_dim, fallback=fallback)
        feats = create_feature_vector_simple(q_emb, p_emb)
        X_list.append(feats)
        label = 1 if row["relevancy"] == "1.0" else 0
        y_list.append(label)
    return np.array(X_list), np.array(y_list)

# -------------------------------
# Main Pipeline for Task 3
# -------------------------------
if __name__ == "__main__":
    start_time = timer()

    # 1. Load GloVe embeddings.
    glove_path = "glove.6B.100d.txt"
    print("Loading GloVe embeddings...")
    glove_embeddings = load_glove_embeddings(glove_path)
    embedding_dim = 100

    # 2. Load and downsample training data.
    print("\nLoading training data...")
    train_df = pd.read_csv("train_data.tsv", sep="\t", dtype="string").dropna()
    train_df["qid"] = train_df["qid"].str.strip()
    train_df["pid"] = train_df["pid"].str.strip()
    train_df = down_sample_by_query(train_df, 10, seed=42)
    print(f"Training data after down sampling: {len(train_df)}")

    # Build the simple feature matrix.
    X_train_simple, y_train_simple = build_feature_matrix(train_df, glove_embeddings, embedding_dim, fallback=True)

    # --- Compute BM25 scores for training data ---
    qid_pid_df_train = train_df[["qid", "pid"]].drop_duplicates()
    train_qid_tokens = {}
    train_passage_tokens = {}
    for idx, row in train_df.iterrows():
        if row["qid"] not in train_qid_tokens:
            train_qid_tokens[row["qid"]] = process_raw_text([row["queries"]], remove_stopwords=True, stemming=True)[0]
        if row["pid"] not in train_passage_tokens:
            train_passage_tokens[row["pid"]] = process_raw_text([row["passage"]], remove_stopwords=True, stemming=True)[0]
    inv_index_train = generate_inverted_index(train_passage_tokens)
    bm25_df_train = bm25(qid_pid_df_train, inv_index_train, train_qid_tokens, train_passage_tokens)
    bm25_dict_train = {(row["qid"], row["pid"]): row["relevance_score"] for idx, row in bm25_df_train.iterrows()}
    bm25_train_feats = []
    for idx, row in train_df.iterrows():
        score = bm25_dict_train.get((row["qid"], row["pid"]), 0)
        bm25_train_feats.append(score)
    bm25_train_feats = np.array(bm25_train_feats).reshape(-1, 1)

    # --- Compute extra features inline for training (query length, passage length, common token count, overlap) ---
    extra_train_feats = []
    for idx, row in train_df.iterrows():
        q_tokens = process_raw_text([row["queries"]], remove_stopwords=True, stemming=True)[0]
        p_tokens = process_raw_text([row["passage"]], remove_stopwords=True, stemming=True)[0]
        q_len = len(q_tokens)
        p_len = len(p_tokens)
        common = len(set(q_tokens) & set(p_tokens))
        overlap = common / q_len if q_len > 0 else 0
        extra_train_feats.append([q_len, p_len, common, overlap])
    extra_train_feats = np.array(extra_train_feats)

    # Enhanced training features: original simple features, BM25 score, and extra features.
    X_train_enhanced = np.hstack((X_train_simple, bm25_train_feats, extra_train_feats))

    # Group sizes (number of samples per query) for training.
    train_group_sizes = train_df.groupby("qid").size().tolist()

    # 3. Load and preprocess validation data.
    print("\nLoading validation data...")
    val_df = pd.read_csv("validation_data.tsv", sep="\t", dtype="string").dropna()
    val_df["qid"] = val_df["qid"].str.strip()
    val_df["pid"] = val_df["pid"].str.strip()
    print(f"Validation data: {len(val_df)} samples")

    # Build simple feature matrix for validation.
    X_val_simple, y_val_simple = build_feature_matrix(val_df, glove_embeddings, embedding_dim, fallback=True)

    # --- Compute BM25 scores for validation data ---
    qid_pid_df_val = val_df[["qid", "pid"]].drop_duplicates()
    val_qid_tokens = {}
    val_passage_tokens = {}
    for idx, row in val_df.iterrows():
        if row["qid"] not in val_qid_tokens:
            val_qid_tokens[row["qid"]] = process_raw_text([row["queries"]], remove_stopwords=True, stemming=True)[0]
        if row["pid"] not in val_passage_tokens:
            val_passage_tokens[row["pid"]] = process_raw_text([row["passage"]], remove_stopwords=True, stemming=True)[0]
    inv_index_val = generate_inverted_index(val_passage_tokens)
    bm25_df_val = bm25(qid_pid_df_val, inv_index_val, val_qid_tokens, val_passage_tokens)
    bm25_dict_val = {(row["qid"], row["pid"]): row["relevance_score"] for idx, row in bm25_df_val.iterrows()}
    bm25_val_feats = []
    for idx, row in val_df.iterrows():
        score = bm25_dict_val.get((row["qid"], row["pid"]), 0)
        bm25_val_feats.append(score)
    bm25_val_feats = np.array(bm25_val_feats).reshape(-1, 1)

    # --- Compute extra features inline for validation data ---
    extra_val_feats = []
    for idx, row in val_df.iterrows():
        q_tokens = process_raw_text([row["queries"]], remove_stopwords=True, stemming=True)[0]
        p_tokens = process_raw_text([row["passage"]], remove_stopwords=True, stemming=True)[0]
        q_len = len(q_tokens)
        p_len = len(p_tokens)
        common = len(set(q_tokens) & set(p_tokens))
        overlap = common / q_len if q_len > 0 else 0
        extra_val_feats.append([q_len, p_len, common, overlap])
    extra_val_feats = np.array(extra_val_feats)

    # Enhanced validation features.
    X_val_enhanced = np.hstack((X_val_simple, bm25_val_feats, extra_val_feats))
    val_group_sizes = val_df.groupby("qid").size().tolist()

    # Utility: Build ground truth relevance dictionary.
    def df_to_relevance_dict(df):
        rel_dict = {}
        grouped = df[df['relevancy'] == '1.0'].groupby('qid')
        for qid, group in grouped:
            rel_dict[qid] = dict(zip(group['pid'], group['relevancy']))
        return rel_dict

    val_relevance_dict = df_to_relevance_dict(val_df)

    # Reduced hyperparameter grid (search over learning_rate, max_depth, n_estimators)
    param_space = {
        'learning_rate': [0.1, 1],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200]
    }

    # -------------------------------
    # Experiment 1: Hyperparameter Search for Simple Features
    # -------------------------------
    best_ndcg_simple = -1
    best_map_simple = -1
    best_params_simple = None
    best_model_simple = None
    results_list_simple = []

    print("\nHyperparameter Search for Simple Features:")
    for lr in param_space['learning_rate']:
        for depth in param_space['max_depth']:
            for n_est in param_space['n_estimators']:
                params = {'learning_rate': lr, 'max_depth': depth, 'n_estimators': n_est}
                print(f"Testing parameters: {params}")
                model = xgb.XGBRanker(
                    objective="rank:ndcg",
                    tree_method="auto",
                    random_state=42,
                    **params
                )
                model.fit(
                    X_train_simple, y_train_simple,
                    group=train_group_sizes,
                    eval_set=[(X_val_simple, y_val_simple)],
                    eval_group=[val_group_sizes],
                    verbose=False
                )
                predictions = model.predict(X_val_simple)
                pred_dict = {}
                for (qid, pid, score) in zip(val_df["qid"], val_df["pid"], predictions):
                    pred_dict.setdefault(qid, {})[pid] = score
                current_map = compute_map(pred_dict, val_relevance_dict)
                current_ndcg = compute_ndcg(pred_dict, val_relevance_dict)
                print(f"Parameters: {params} - MAP: {current_map:.4f}, NDCG: {current_ndcg:.4f}")
                results_list_simple.append({"params": params, "MAP": current_map, "NDCG": current_ndcg})
                if current_ndcg > best_ndcg_simple or (current_ndcg == best_ndcg_simple and current_map > best_map_simple):
                    best_ndcg_simple = current_ndcg
                    best_map_simple = current_map
                    best_params_simple = params
                    best_model_simple = model

    print("\nSimple Features Hyperparameter Results:")
    df_results_simple = pd.DataFrame(results_list_simple)
    df_results_simple.sort_values(by=["NDCG", "MAP"], ascending=False, inplace=True)
    print(df_results_simple)
    print(f"Best Simple Params: {best_params_simple}")
    print(f"Best Simple Validation MAP: {best_map_simple:.4f}, NDCG: {best_ndcg_simple:.4f}")

    # -------------------------------
    # Experiment 2: Hyperparameter Search for Enhanced Features
    # (Enhanced features include: simple features + BM25 score + extra inline features)
    # -------------------------------
    best_ndcg_enhanced = -1
    best_map_enhanced = -1
    best_params_enhanced = None
    best_model_enhanced = None
    results_list_enhanced = []

    print("\nHyperparameter Search for Enhanced Features:")
    for lr in param_space['learning_rate']:
        for depth in param_space['max_depth']:
            for n_est in param_space['n_estimators']:
                params = {'learning_rate': lr, 'max_depth': depth, 'n_estimators': n_est}
                print(f"Testing parameters: {params}")
                model = xgb.XGBRanker(
                    objective="rank:ndcg",
                    tree_method="auto",
                    random_state=42,
                    **params
                )
                model.fit(
                    X_train_enhanced, y_train_simple,
                    group=train_group_sizes,
                    eval_set=[(X_val_enhanced, y_val_simple)],
                    eval_group=[val_group_sizes],
                    verbose=False
                )
                predictions = model.predict(X_val_enhanced)
                pred_dict = {}
                for (qid, pid, score) in zip(val_df["qid"], val_df["pid"], predictions):
                    pred_dict.setdefault(qid, {})[pid] = score
                current_map = compute_map(pred_dict, val_relevance_dict)
                current_ndcg = compute_ndcg(pred_dict, val_relevance_dict)
                print(f"Parameters: {params} - MAP: {current_map:.4f}, NDCG: {current_ndcg:.4f}")
                results_list_enhanced.append({"params": params, "MAP": current_map, "NDCG": current_ndcg})
                if current_ndcg > best_ndcg_enhanced or (current_ndcg == best_ndcg_enhanced and current_map > best_map_enhanced):
                    best_ndcg_enhanced = current_ndcg
                    best_map_enhanced = current_map
                    best_params_enhanced = params
                    best_model_enhanced = model

    print("\nEnhanced Features Hyperparameter Results:")
    df_results_enhanced = pd.DataFrame(results_list_enhanced)
    df_results_enhanced.sort_values(by=["NDCG", "MAP"], ascending=False, inplace=True)
    print(df_results_enhanced)
    print(f"Best Enhanced Params: {best_params_enhanced}")
    print(f"Best Enhanced Validation MAP: {best_map_enhanced:.4f}, NDCG: {best_ndcg_enhanced:.4f}")

    # -------------------------------
    # Choose Overall Best Model and Rerank Test Data
    # -------------------------------
    print("\nSelecting the overall best model for test ranking:")
    if (best_ndcg_enhanced > best_ndcg_simple) or (best_ndcg_enhanced == best_ndcg_simple and best_map_enhanced > best_map_simple):
        overall_best_model = best_model_enhanced
        use_enhanced = True
        print("Enhanced Features model selected.")
    else:
        overall_best_model = best_model_simple
        use_enhanced = False
        print("Simple Features model selected.")

    # Verify test files exist.
    if not os.path.exists("test-queries.tsv"):
        raise FileNotFoundError("test-queries.tsv not found. This file is required for test re-ranking.")
    if not os.path.exists("candidate_passages_top1000.tsv"):
        raise FileNotFoundError("candidate_passages_top1000.tsv not found. This file is required for test re-ranking.")

    print("\nReranking test queries using the overall best model...")
    # Load test queries.
    test_queries_df = pd.read_csv("test-queries.tsv", sep="\t", header=None, names=["qid", "query"], dtype="string")
    test_queries_df["qid"] = test_queries_df["qid"].str.strip()
    test_queries_df["query"] = test_queries_df["query"].str.strip()

    # Load candidate passages.
    candidates_df = pd.read_csv("candidate_passages_top1000.tsv", sep="\t", header=None,
                                  names=["qid", "pid", "query", "passage"], dtype="string")
    candidates_df["qid"] = candidates_df["qid"].str.strip()
    candidates_df["pid"] = candidates_df["pid"].str.strip()

    # --- For test data, also compute BM25 scores.
    candidate_qid_tokens = {}
    candidate_passage_tokens = {}
    for idx, row in candidates_df.iterrows():
        if row["qid"] not in candidate_qid_tokens:
            candidate_qid_tokens[row["qid"]] = process_raw_text([row["query"]], remove_stopwords=True, stemming=True)[0]
        if row["pid"] not in candidate_passage_tokens:
            candidate_passage_tokens[row["pid"]] = process_raw_text([row["passage"]], remove_stopwords=True, stemming=True)[0]
    inv_index_cand = generate_inverted_index(candidate_passage_tokens)
    qid_pid_df_cand = candidates_df[["qid", "pid"]].drop_duplicates()
    bm25_df_cand = bm25(qid_pid_df_cand, inv_index_cand, candidate_qid_tokens, candidate_passage_tokens)
    bm25_dict_cand = {(row["qid"], row["pid"]): row["relevance_score"] for idx, row in bm25_df_cand.iterrows()}

    candidate_scores = []
    for idx, row in candidates_df.iterrows():
        qid = row["qid"]
        pid = row["pid"]
        q_tokens = process_raw_text([row["query"]], remove_stopwords=True, stemming=True)[0]
        p_tokens = process_raw_text([row["passage"]], remove_stopwords=True, stemming=True)[0]
        q_emb, _, _ = compute_average_embedding(q_tokens, glove_embeddings, embedding_dim, fallback=True)
        p_emb, _, _ = compute_average_embedding(p_tokens, glove_embeddings, embedding_dim, fallback=True)
        base_feat = create_feature_vector_simple(q_emb, p_emb)
        if use_enhanced:
            bm25_score = bm25_dict_cand.get((qid, pid), 0)
            q_len = len(q_tokens)
            p_len = len(p_tokens)
            common = len(set(q_tokens) & set(p_tokens))
            overlap = common / q_len if q_len > 0 else 0
            extra_feat = np.array([q_len, p_len, common, overlap])
            final_feat = np.hstack((base_feat, [bm25_score], extra_feat))
        else:
            final_feat = base_feat
        score = overall_best_model.predict(final_feat.reshape(1, -1))[0]
        candidate_scores.append((qid, pid, score))

    ranking = {}
    for qid, pid, score in candidate_scores:
        ranking.setdefault(qid, []).append((pid, score))
    with open("LM.txt", "w") as out_file:
        for qid in ranking:
            sorted_passages = sorted(ranking[qid], key=lambda x: x[1], reverse=True)
            for rank, (pid, score) in enumerate(sorted_passages, start=1):
                out_file.write(f"{qid} A2 {pid} {rank} {score:.4f} LM\n")

    end_time = timer()
    print(f"\nTask 3 completed in {round((end_time - start_time) / 60, 2)} minutes.")
