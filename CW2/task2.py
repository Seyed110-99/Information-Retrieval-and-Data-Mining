import os  # For file and path operations
import re  # Regular expressions for text processing
import math  # Mathematical functions
import json  # JSON file reading/writing
import nltk  # Natural Language Toolkit for text processing
import numpy as np  # For numerical operations and array handling
import pandas as pd  # For data manipulation and analysis
from timeit import default_timer as timer  # For timing execution
import ssl  # For handling SSL certificate issues during downloads
import urllib.request  # For downloading files from the web
import zipfile  # For extracting zip files
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.preprocessing import StandardScaler  # For feature scaling

# Import Task 1 functions.
from task1 import process_raw_text, compute_map, compute_ndcg

# Download required NLTK resources.
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

#############################################
# Download and Prepare GloVe Embeddings (100d)
#############################################
# Define file names and URL for GloVe embeddings.
GLOVE_FILE = "glove.6B.100d.txt"
ZIP_FILE = "glove.6B.zip"
GLOVE_URL = "http://nlp.stanford.edu/data/glove.6B.zip"

# Check if the GloVe file exists; if not, download and extract it.
if not os.path.exists(GLOVE_FILE):
    print(f"{GLOVE_FILE} not found. Downloading {ZIP_FILE}...")
    urllib.request.urlretrieve(GLOVE_URL, ZIP_FILE)  # Download the zip file
    print(f"Downloaded {ZIP_FILE}. Extracting {GLOVE_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zf:
        zf.extract(GLOVE_FILE)  # Extract the required GloVe file from the zip
    print("Extraction complete.")
    os.remove(ZIP_FILE)  # Clean up by removing the zip file

#############################################
# Embedding Functions
#############################################
def load_glove_embeddings(glove_path):
    """
    Load GloVe embeddings from the specified file.
    
    Reads the GloVe file line by line and builds a dictionary mapping words to their corresponding embedding vectors.
    
    Args:
        glove_path (str): Path to the GloVe embeddings file.
        
    Returns:
        dict: A dictionary where each key is a word and each value is a NumPy array representing its embedding vector.
    """
    embeddings = {}
    with open(glove_path, "r", encoding="utf8") as f:
        # Process each line in the file
        for line in f:
            parts = line.strip().split()
            word = parts[0]  # The first part is the word
            # The rest of the parts form the vector
            vec = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vec
    return embeddings

def compute_average_embedding(tokens, embeddings, embedding_dim=100, fallback=False):
    """
    Compute the average embedding vector for a list of tokens.
    
    For each token in the list, if the token exists in the provided embeddings, its vector is included.
    If fallback is True, a random vector is generated for tokens not found in the embeddings.
    The function returns the average vector over all tokens, along with the number of matches and total tokens.
    
    Args:
        tokens (list of str): List of tokenized words.
        embeddings (dict): Dictionary mapping words to their embedding vectors.
        embedding_dim (int, optional): Dimensionality of the embeddings (default is 100).
        fallback (bool, optional): If True, generates a random vector for missing tokens. Defaults to False.
        
    Returns:
        tuple: (average_vector (np.ndarray), match_count (int), total_tokens (int))
    """
    vectors = []  # List to store embedding vectors of tokens
    match_count = 0  # Counter for tokens found in embeddings
    for token in tokens:
        if token in embeddings:
            vectors.append(embeddings[token])
            match_count += 1
        elif fallback:
            # Generate a random vector if token is not found and fallback is enabled
            vectors.append(np.random.uniform(low=-1, high=1, size=embedding_dim))
    if vectors:
        # Compute mean of vectors if at least one vector is present
        return np.mean(vectors, axis=0), match_count, len(tokens)
    else:
        # If no vectors found, return a zero vector
        return np.zeros(embedding_dim), match_count, len(tokens)

#############################################
# Feature Engineering (Simplified)
#############################################
def create_feature_vector_simple(query_emb, passage_emb):
    """
    Create a simple feature vector for a query–passage pair.
    
    The feature vector is built by concatenating a constant bias (1), the query embedding, and the passage embedding.
    If the inputs are tuples (as returned by compute_average_embedding), only the vector part is extracted.
    
    Args:
        query_emb (np.ndarray or tuple): The embedding for the query or a tuple where the first element is the embedding.
        passage_emb (np.ndarray or tuple): The embedding for the passage or a tuple where the first element is the embedding.
        
    Returns:
        np.ndarray: The concatenated feature vector.
    """
    # If embeddings are returned as tuples, extract the first element (the vector)
    if isinstance(query_emb, tuple):
        query_emb = query_emb[0]
    if isinstance(passage_emb, tuple):
        passage_emb = passage_emb[0]
    # Concatenate bias term, query embedding, and passage embedding
    return np.concatenate(([1], query_emb, passage_emb))

#############################################
# Negative Downsampling
#############################################
def down_sample_by_query(df, max_negatives=100, seed=42):
    """
    Downsample negative samples in a DataFrame for each query.
    
    For each query in the DataFrame, all positive examples are kept, and a random sample of negative examples
    is selected such that the total number of negatives does not exceed max_negatives per query.
    
    Args:
        df (pandas.DataFrame): DataFrame containing at least the columns 'qid' (query ID) and 'relevancy'.
        max_negatives (int, optional): Maximum number of negative examples per query. Defaults to 100.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        
    Returns:
        pandas.DataFrame: DataFrame containing all positive examples and the sampled negatives.
    """
    positives = df[df['relevancy'] == "1.0"]  # All positive samples
    negatives = df[df['relevancy'] != "1.0"]  # All negative samples
    sampled_negatives = []
    # Group negatives by query id and sample from each group
    for qid, group in negatives.groupby('qid'):
        num_pos = positives[positives['qid'] == qid].shape[0]
        available = max(0, max_negatives - num_pos)
        if available > 0 and len(group) > 0:
            sampled_negatives.append(group.sample(n=min(len(group), available), random_state=seed))
    # Combine positives with sampled negatives and shuffle
    if sampled_negatives:
        negatives_sampled = pd.concat(sampled_negatives)
        combined = pd.concat([positives, negatives_sampled]).sample(frac=1, random_state=seed).reset_index(drop=True)
        return combined
    else:
        return df

#############################################
# Logistic Regression (Mini-Batch Gradient Descent)
#############################################
def sigmoid(z):
    """
    Compute the sigmoid function.
    
    Args:
        z (np.ndarray or float): Input value or array.
        
    Returns:
        np.ndarray or float: The sigmoid of the input.
    """
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, learning_rate=0.01, num_epochs=100, batch_size=None):
    """
    Train a logistic regression model using mini-batch gradient descent.
    
    The function updates the weights and bias to minimize the logistic loss over the training data.
    
    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Binary target vector of shape (n_samples,).
        learning_rate (float, optional): The learning rate for gradient descent. Defaults to 0.01.
        num_epochs (int, optional): Number of training epochs. Defaults to 100.
        batch_size (int, optional): Size of mini-batches; if None or larger than n_samples, full batch is used.
        
    Returns:
        tuple: (weights (np.ndarray), bias (float), loss_history (list of float))
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)  # Initialize weights to zeros
    bias = 0.0  # Initialize bias
    losses = []  # List to record loss at each epoch
    
    # If batch size is not specified or is greater than number of samples, use full batch
    if batch_size is None or batch_size > n_samples:
        batch_size = n_samples

    # Training loop over epochs
    for epoch in range(num_epochs):
        # Shuffle the training data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        batch_losses = []
        # Process mini-batches
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            # Compute model predictions
            logits = np.dot(X_batch, weights) + bias
            predictions = sigmoid(logits)
            error = predictions - y_batch  # Compute error (difference between prediction and true label)
            # Compute gradients for weights and bias
            grad_w = np.dot(X_batch.T, error) / len(y_batch)
            grad_b = np.sum(error) / len(y_batch)
            # Update parameters using gradient descent step
            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b
            # Compute logistic loss for the batch (with small constant added to avoid log(0))
            loss = -np.mean(y_batch * np.log(predictions + 1e-10) +
                            (1 - y_batch) * np.log(1 - predictions + 1e-10))
            batch_losses.append(loss)
        # Compute average loss for the epoch
        epoch_loss = np.mean(batch_losses)
        losses.append(epoch_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} (lr={learning_rate}): loss = {epoch_loss:.4f}")
    return weights, bias, losses

def predict_logistic_regression(X, weights, bias):
    """
    Predict probabilities using the trained logistic regression model.
    
    Args:
        X (np.ndarray): Feature matrix.
        weights (np.ndarray): Trained weight vector.
        bias (float): Trained bias term.
        
    Returns:
        np.ndarray: Predicted probabilities for the positive class.
    """
    logits = np.dot(X, weights) + bias
    return sigmoid(logits)

#############################################
# Main Pipeline for Task 2
#############################################
if __name__ == "__main__":
    # Handle SSL context for NLTK downloads if necessary.
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    overall_start = timer()  # Start timer for overall pipeline

    # 1. Load GloVe embeddings.
    print("Loading GloVe embeddings...")
    glove_embeddings = load_glove_embeddings(GLOVE_FILE)
    embedding_dim = 100  # Dimensionality for GloVe embeddings

    # 2. Load and preprocess training data.
    print("Loading training data...")
    train_df = pd.read_csv("train_data.tsv", sep="\t", dtype="string")
    # Remove any extra whitespace from query and passage IDs.
    train_df['qid'] = train_df['qid'].str.strip()
    train_df['pid'] = train_df['pid'].str.strip()
    # Optionally downsample negatives; here we use the entire training set.
    train_sample = down_sample_by_query(train_df, max_negatives=10, seed=42)
    # train_sample = train_df
    print(f"Training samples after downsampling: {len(train_sample)}")

    # Tokenize queries and passages.
    # Note: stemming is enabled here to match Task 1 processing.
    train_query_tokens = process_raw_text(train_sample["queries"].tolist(), remove_stopwords=True, stemming=True)
    train_passage_tokens = process_raw_text(train_sample["passage"].tolist(), remove_stopwords=True, stemming=True)

    # Compute average embeddings for queries and passages.
    query_embeddings = []
    for tokens in train_query_tokens:
        emb, matches, total = compute_average_embedding(tokens, glove_embeddings, embedding_dim, fallback=True)
        query_embeddings.append(emb)
    passage_embeddings = []
    for tokens in train_passage_tokens:
        emb, matches, total = compute_average_embedding(tokens, glove_embeddings, embedding_dim, fallback=True)
        passage_embeddings.append(emb)

    # 3. Build feature vectors using a simplified representation.
    # The feature vector is the concatenation of a constant 1, the query embedding, and the passage embedding.
    features = [create_feature_vector_simple(q_emb, p_emb) for q_emb, p_emb in zip(query_embeddings, passage_embeddings)]
    X_train = np.vstack(features)  # Convert list of feature vectors to a NumPy array
    # Convert relevancy labels to binary: 1 for "1.0", else 0.
    y_train = np.array([1 if rel == "1.0" else 0 for rel in train_sample["relevancy"].tolist()])

    # Scale features using StandardScaler.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # 4. Train logistic regression models using different learning rates.
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    models = {}  # Dictionary to store trained models for each learning rate
    loss_histories = {}  # Dictionary to store loss history for each learning rate
    num_epochs = 100  # Number of training epochs
    batch_size = 5000  # Mini-batch size

    print("Training logistic regression models with different learning rates:")
    for lr in learning_rates:
        print(f"\nTraining model with learning rate = {lr}")
        w, b, losses = train_logistic_regression(X_train, y_train, learning_rate=lr, num_epochs=num_epochs, batch_size=batch_size)
        models[lr] = (w, b)
        loss_histories[lr] = losses

    # Plot training loss curves for each learning rate.
    plt.figure(figsize=(10, 6))
    for lr in learning_rates:
        plt.plot(range(num_epochs), loss_histories[lr], label=f"lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss for Different Learning Rates")
    plt.legend()
    plt.grid(True)
    plt.savefig("lr_training_loss.png")  # Save the plot as an image

    # 5. Load and preprocess validation data.
    print("\nLoading validation data...")
    val_df = pd.read_csv("validation_data.tsv", sep="\t", dtype="string")
    val_df['qid'] = val_df['qid'].str.strip()
    val_df['pid'] = val_df['pid'].str.strip()
    

    print(f"\nValidation data: {len(val_df)} samples")
    # Tokenize validation queries and passages.
    # Note: Stemming is disabled here to better match GloVe's vocabulary.
    val_query_tokens = process_raw_text(val_df["queries"].tolist(), remove_stopwords=True, stemming=False)
    val_passage_tokens = process_raw_text(val_df["passage"].tolist(), remove_stopwords=True, stemming=False)
    # Compute average embeddings without fallback for validation data.
    val_query_embeddings = [compute_average_embedding(tokens, glove_embeddings, embedding_dim, fallback=False)[0]
                            for tokens in val_query_tokens]
    val_passage_embeddings = [compute_average_embedding(tokens, glove_embeddings, embedding_dim, fallback=False)[0]
                              for tokens in val_passage_tokens]
    # Build feature vectors for validation data.
    X_val = np.vstack([create_feature_vector_simple(q_emb, p_emb)
                        for q_emb, p_emb in zip(val_query_embeddings, val_passage_embeddings)])
    X_val = scaler.transform(X_val)
    y_val = np.array([1 if rel=="1.0" else 0 for rel in val_df["relevancy"].tolist()])

    # 6. Evaluate each model on the validation data.
    best_lr = learning_rates[0]
    best_map = -1
    best_ndcg = -1
    best_model = None
    val_map = None
    val_ndcg = None

    for lr, (w, b) in models.items():
        # Compute predicted probabilities on validation set.
        val_probs = predict_logistic_regression(X_val, w, b)
        val_df["score"] = val_probs
        # Build prediction dictionary: each query id maps to a dictionary of {pid: score}
        pred_dict = {qid: group.set_index("pid")["score"].to_dict() for qid, group in val_df.groupby("qid")}
        # Build ground truth relevance dictionary for queries with relevancy "1.0"
        relevance_dict = {qid: group.set_index("pid")["relevancy"].to_dict()
                          for qid, group in val_df[val_df["relevancy"]=="1.0"].groupby("qid")}
        # Compute MAP and NDCG using Task 1 evaluation functions.
        current_map = compute_map(pred_dict, relevance_dict)
        current_ndcg = compute_ndcg(pred_dict, relevance_dict)
        print(f"Learning rate {lr} - Validation MAP: {current_map:.4f}, NDCG: {current_ndcg:.4f}")
        # Select the model with the highest MAP.
        if current_map >= best_map and current_ndcg >= best_ndcg:
            best_map = current_map
            best_lr = lr
            best_model = (w, b)
            val_map = current_map
            val_ndcg = current_ndcg

    print(f"\nSelected best learning rate: {best_lr} with MAP: {val_map:.4f} and NDCG: {val_ndcg:.4f}")

    # 7. Re-rank candidate passages for test queries.
    # Check if test queries file exists.
    if not os.path.exists("test-queries.tsv"):
        raise FileNotFoundError("test-queries.tsv not found. This file is required for test re-ranking.")
    print("Ranking candidate passages for test queries...")
    # Load test queries file.
    test_queries_df = pd.read_csv("test-queries.tsv", sep="\t", header=None, names=['qid', 'query'], dtype="string")
    test_queries_df['qid'] = test_queries_df['qid'].str.strip()
    test_queries_df["query"] = test_queries_df["query"].str.strip()
    # Tokenize test queries.
    test_query_tokens = process_raw_text(test_queries_df["query"].tolist(), remove_stopwords=True, stemming=False)
    # Compute average embeddings for test queries without fallback.
    test_query_embeddings = [compute_average_embedding(tokens, glove_embeddings, embedding_dim, fallback=False)[0]
                             for tokens in test_query_tokens]
    # Create a mapping from query id to its embedding.
    qid_to_emb = dict(zip(test_queries_df["qid"], test_query_embeddings))

    # Load candidate passages for test queries.
    candidates_df = pd.read_csv("candidate_passages_top1000.tsv", sep="\t", header=None,
                                  names=["qid", "pid", "query", "passage"], dtype="string")
    candidates_df['qid'] = candidates_df['qid'].str.strip()
    candidates_df['pid'] = candidates_df['pid'].str.strip()

    candidate_scores = []
    w, b = best_model  # Best logistic regression model parameters
    # Iterate over each candidate passage.
    for _, row in candidates_df.iterrows():
        qid = row["qid"]
        pid = row["pid"]
        query_text = row["query"]
        passage_text = row["passage"]
        # Tokenize query and passage text.
        q_tokens = process_raw_text([query_text], remove_stopwords=True, stemming=False)[0]
        p_tokens = process_raw_text([passage_text], remove_stopwords=True, stemming=False)[0]
        # Compute embeddings for the query and passage.
        q_emb = compute_average_embedding(q_tokens, glove_embeddings, embedding_dim, fallback=False)[0]
        p_emb = compute_average_embedding(p_tokens, glove_embeddings, embedding_dim, fallback=False)[0]
        # Create feature vector and scale it.
        feat = create_feature_vector_simple(q_emb, p_emb)
        feat = scaler.transform(feat.reshape(1, -1))
        # Compute score using the trained logistic regression model.
        score = predict_logistic_regression(feat, w, b)[0]
        candidate_scores.append((qid, pid, score))
    
    # Build ranking dictionary: each query maps to a list of (pid, score) tuples.
    ranking = {}
    for qid, pid, score in candidate_scores:
        ranking.setdefault(qid, []).append((pid, score))
    
    # Write the final ranking to an output file.
    with open("LR.txt", "w") as out_file:
        for qid in ranking:
            sorted_passages = sorted(ranking[qid], key=lambda x: x[1], reverse=True)
            for rank, (pid, score) in enumerate(sorted_passages, start=1):
                out_file.write(f"{qid} A2 {pid} {rank} {score:.4f} LR\n")

    overall_end = timer()  # End timer for overall pipeline
    print(f"\nTask completed in {round((overall_end - overall_start) / 60, 1)} minutes.")



