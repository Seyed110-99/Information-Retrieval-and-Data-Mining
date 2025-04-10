import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import nltk
from collections import Counter
import re
import os
import urllib.request
import zipfile
from task1 import process_raw_text, compute_map, compute_ndcg
from task2 import down_sample_by_query

# Download necessary NLTK resources for preprocessing.
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Ensure the required GloVe file is present. If not, download and extract it.
GLOVE_FILE = "glove.6B.100d.txt"
ZIP_FILE = "glove.6B.zip"
GLOVE_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
if not os.path.exists(GLOVE_FILE):
    print(f"{GLOVE_FILE} not found. Downloading...")
    urllib.request.urlretrieve(GLOVE_URL, ZIP_FILE)
    with zipfile.ZipFile(ZIP_FILE, "r") as zf:
        zf.extract(GLOVE_FILE)
    os.remove(ZIP_FILE)

#############################################
# Text Preprocessing and Data Preparation Functions
#############################################

def build_vocab(tokenized_texts, min_freq=1):
    """
    Build a vocabulary dictionary with token frequencies.
    Tokens with frequency >= min_freq are included.
    Special tokens '<PAD>' and '<UNK>' are reserved.

    Args:
        tokenized_texts (list of list of str): Tokenized texts.
        min_freq (int): Minimum frequency to include a token.

    Returns:
        dict: Token to index mapping.
    """
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def tokens_to_indices(tokenized_texts, vocab):
    """
    Convert a list of token lists into a list of index lists based on vocab.
    Unknown tokens are mapped to index of '<UNK>'.

    Args:
        tokenized_texts (list of list of str): Tokenized texts.
        vocab (dict): Token to index mapping.

    Returns:
        list of list of int: Indexed token sequences.
    """
    return [[vocab.get(token, vocab['<UNK>']) for token in tokens] for tokens in tokenized_texts]

def pad_sequences(sequences, max_len=200):
    """
    Pad or truncate sequences to a fixed max_len for input to neural networks.

    Args:
        sequences (list of list of int): Token index sequences.
        max_len (int): Maximum sequence length.

    Returns:
        torch.Tensor: LongTensor of shape (batch_size, max_len).
    """
    padded = []
    for seq in sequences:
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [0] * (max_len - len(seq))
        padded.append(seq)
    return torch.tensor(padded, dtype=torch.long)

def combine_text(query, passage, sep_token="[SEP]"):
    """
    Preprocess and combine query and passage using a separator token.

    Args:
        query (str): Query text.
        passage (str): Passage text.
        sep_token (str): Separator token.

    Returns:
        list of str: Combined and tokenized query + [SEP] + passage.
    """
    query_tokens = process_raw_text([query], remove_stopwords=True, stemming=True)[0]
    passage_tokens = process_raw_text([passage], remove_stopwords=True, stemming=True)[0]
    return query_tokens + [sep_token] + passage_tokens

class CombinedIRDataset(Dataset):
    """
    PyTorch Dataset wrapper for combined query-passage sequences.
    Returns input tensor and corresponding binary label.
    """
    def __init__(self, combined_tensor, labels):
        self.data = combined_tensor
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

#############################################
# Neural Ranking Models
#############################################

class ImprovedRankingModel(nn.Module):
    """
    Bi-directional LSTM-based ranking model.
    Uses stacked BiLSTM layers and a fully connected layer with dropout and ReLU.

    Args:
        vocab_size (int): Size of vocabulary.
        embedding_dim (int): Dimensionality of embeddings.
        hidden_dim (int): Hidden size for LSTM.
        num_layers (int): Number of LSTM layers.
        fc_dim (int): Size of fully connected hidden layer.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64, num_layers=2, fc_dim=32, dropout_rate=0.5):
        super(ImprovedRankingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.fc1 = nn.Linear(hidden_dim * 2, fc_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_dim, 1)

    def forward(self, x):
        x = self.embedding(x)                        # (batch, seq_len, emb_dim)
        x = self.dropout(x)
        lstm_out, (h_n, _) = self.lstm(x)
        h = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)  # Concatenate final hidden states
        fc_out = self.relu(self.fc1(h))
        score = torch.sigmoid(self.fc2(fc_out))      # Output relevance score
        return score

class SimpleCNNRankingModel(nn.Module):
    """
    Faster CNN-based ranking model for passage retrieval.
    Uses convolution, pooling, dropout, and dense layers to compute score.

    Args:
        vocab_size (int): Vocabulary size.
        embedding_dim (int): Embedding dimensionality.
        num_filters (int): Number of convolution filters.
        kernel_size (int): Convolution kernel size.
        fc_dim (int): Fully connected layer size.
        dropout_rate (float): Dropout probability.
    """
    def __init__(self, vocab_size, embedding_dim=100, num_filters=64, kernel_size=3, fc_dim=32, dropout_rate=0.5):
        super(SimpleCNNRankingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters,
                              kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters, fc_dim)
        self.out = nn.Linear(fc_dim, 1)

    def forward(self, x):
        x = self.embedding(x)           # (batch, seq_len, emb_dim)
        x = x.transpose(1, 2)           # (batch, emb_dim, seq_len)
        x = self.relu(self.conv(x))     # (batch, num_filters, seq_len)
        x = self.pool(x).squeeze(2)     # (batch, num_filters)
        x = self.dropout(x)
        x = self.relu(self.fc(x))       # Fully connected layer
        score = torch.sigmoid(self.out(x))
        return score

#############################################
# Helper Functions
#############################################

def generate_nn_txt(model, vocab, test_queries_file="test-queries.tsv",
                    candidate_file="candidate_passages_top1000.tsv", max_len=200):
    """
    Generates rankings for the test queries using the trained neural network model.
    Saves results in TREC-style format to 'NN.txt'.
    """
    model.eval()
    device_local = next(model.parameters()).device
    test_queries_df = pd.read_csv(test_queries_file, sep="\t", header=None, names=["qid", "query"], dtype="string")
    test_queries_df["qid"] = test_queries_df["qid"].str.strip()
    test_queries_df["query"] = test_queries_df["query"].str.strip()
    query_dict = dict(zip(test_queries_df["qid"], test_queries_df["query"]))

    candidate_df = pd.read_csv(candidate_file, sep="\t", header=None,
                               names=["qid", "pid", "query", "passage"], dtype="string")
    candidate_df["qid"] = candidate_df["qid"].str.strip()
    candidate_df["pid"] = candidate_df["pid"].str.strip()

    results = []
    with torch.no_grad():
        for _, row in candidate_df.iterrows():
            qid = row["qid"]
            pid = row["pid"]
            if qid not in query_dict:
                continue
            combined = combine_text(query_dict[qid], row["passage"])
            indices = tokens_to_indices([combined], vocab)
            padded = pad_sequences(indices, max_len=max_len).to(device_local)
            score = model(padded).item()
            results.append((qid, pid, score))

    ranking = {}
    for qid, pid, score in results:
        ranking.setdefault(qid, []).append((pid, score))

    with open("NN.txt", "w") as f:
        for qid in ranking:
            sorted_results = sorted(ranking[qid], key=lambda x: x[1], reverse=True)
            for rank, (pid, score) in enumerate(sorted_results, start=1):
                f.write(f"{qid} A2 {pid} {rank} {score:.4f} NN\n")
    print("Test rankings saved to NN.txt")

def evaluate_model(model, loader):
    """
    Evaluate the model on the given DataLoader and return list of predicted scores.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            predictions.extend(outputs.cpu().numpy())
    return predictions

def df_to_relevance_dict(df):
    """
    Convert DataFrame into relevance dict: {qid: {pid: '1.0'}}.

    Args:
        df (pd.DataFrame): DataFrame with qid, pid, relevancy columns.

    Returns:
        dict: Mapping from qid to {pid: '1.0'} for relevant passages.
    """
    rel_dict = {}
    grouped = df[df['relevancy'] == '1.0'].groupby('qid')
    for qid, group in grouped:
        rel_dict[qid] = dict(zip(group['pid'], group['relevancy']))
    return rel_dict


#############################################
# Main Pipeline for Task 4 with Model Comparison
#############################################

if __name__ == "__main__":
    start = timer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # STEP 1: Load and preprocess training data
    print("Loading training data...")
    train_df = pd.read_csv("train_data.tsv", sep="\t", dtype="string")
    train_df['qid'] = train_df['qid'].str.strip()
    train_df['pid'] = train_df['pid'].str.strip()
    train_df = down_sample_by_query(train_df, 10, seed=10)
    print(f"Training data after down sampling: {len(train_df)}")

    # Tokenize and combine query-passage pairs
    combined_texts = [combine_text(row['queries'], row['passage']) for _, row in train_df.iterrows()]
    vocab = build_vocab(combined_texts, min_freq=1)
    print(f"Vocabulary size: {len(vocab)}")

    # Convert tokens to padded sequences and prepare dataset
    combined_indices = tokens_to_indices(combined_texts, vocab)
    combined_tensor = pad_sequences(combined_indices, max_len=200)
    labels = [1 if rel == "1.0" else 0 for rel in train_df["relevancy"]]
    train_dataset = CombinedIRDataset(combined_tensor, labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # STEP 2: Initialize Bi-LSTM and CNN models
    model_lstm = ImprovedRankingModel(vocab_size=len(vocab)).to(device)
    model_cnn = SimpleCNNRankingModel(vocab_size=len(vocab)).to(device)

    # STEP 3: Load and assign GloVe embeddings
    glove_embeddings = {}
    with open(GLOVE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word, vec = parts[0], list(map(float, parts[1:]))
            glove_embeddings[word] = np.array(vec)

    embedding_matrix = torch.randn(len(vocab), 100) * 0.01
    for token, idx in vocab.items():
        if token in glove_embeddings:
            embedding_matrix[idx] = torch.tensor(glove_embeddings[token], dtype=torch.float)

    model_lstm.embedding.weight.data.copy_(embedding_matrix)
    model_cnn.embedding.weight.data.copy_(embedding_matrix)

    # STEP 4: Train both models
    num_epochs = 5
    lr = 1e-2
    criterion = nn.BCELoss()
    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=lr)
    optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=lr)

    print("\nTraining Bi-LSTM Model...")
    lstm_losses = []
    lstm_start = timer()
    model_lstm.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        for inputs, labels_batch in train_loader:
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            optimizer_lstm.zero_grad()
            outputs = model_lstm(inputs).squeeze()
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer_lstm.step()
            epoch_losses.append(loss.item())
        lstm_losses.append(np.mean(epoch_losses))
        print(f"LSTM Epoch {epoch+1}/{num_epochs}, Loss: {lstm_losses[-1]:.4f}")
    lstm_time = timer() - lstm_start

    print("\nTraining Simple CNN Model...")
    cnn_losses = []
    cnn_start = timer()
    model_cnn.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        for inputs, labels_batch in train_loader:
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            optimizer_cnn.zero_grad()
            outputs = model_cnn(inputs).squeeze()
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer_cnn.step()
            epoch_losses.append(loss.item())
        cnn_losses.append(np.mean(epoch_losses))
        print(f"CNN Epoch {epoch+1}/{num_epochs}, Loss: {cnn_losses[-1]:.4f}")
    cnn_time = timer() - cnn_start

    # STEP 5: Compare training loss curves
    plt.figure(figsize=(8,6))
    plt.plot(range(1, num_epochs+1), lstm_losses, marker='o', label="Bi-LSTM")
    plt.plot(range(1, num_epochs+1), cnn_losses, marker='o', label="Simple CNN")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("nn_training_loss_comparison.png")

    # STEP 6: Load and preprocess validation data
    print("\nLoading validation data...")
    val_df = pd.read_csv("validation_data.tsv", sep="\t", dtype="string")
    val_df['qid'] = val_df['qid'].str.strip()
    val_df['pid'] = val_df['pid'].str.strip()
    print(f"\nValidation data: {len(val_df)} samples")

    combined_texts_val = [combine_text(row['queries'], row['passage']) for _, row in val_df.iterrows()]
    combined_indices_val = tokens_to_indices(combined_texts_val, vocab)
    combined_tensor_val = pad_sequences(combined_indices_val, max_len=200)
    labels_val = [1 if rel == "1.0" else 0 for rel in val_df["relevancy"]]
    val_dataset = CombinedIRDataset(combined_tensor_val, labels_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # STEP 7: Evaluate both models on validation set
    lstm_val_scores = evaluate_model(model_lstm, val_loader)
    cnn_val_scores = evaluate_model(model_cnn, val_loader)

    pred_dict_lstm = {}
    pred_dict_cnn = {}
    for idx, row in val_df.iterrows():
        qid, pid = row['qid'], row['pid']
        pred_dict_lstm.setdefault(qid, {})[pid] = float(lstm_val_scores[idx])
        pred_dict_cnn.setdefault(qid, {})[pid] = float(cnn_val_scores[idx])

    relevance_dict_val = df_to_relevance_dict(val_df)
    map_lstm = compute_map(pred_dict_lstm, relevance_dict_val)
    ndcg_lstm = compute_ndcg(pred_dict_lstm, relevance_dict_val)
    map_cnn = compute_map(pred_dict_cnn, relevance_dict_val)
    ndcg_cnn = compute_ndcg(pred_dict_cnn, relevance_dict_val)

    print(f"\nBi-LSTM Validation MAP: {map_lstm:.4f}, NDCG: {ndcg_lstm:.4f}")
    print(f"Simple CNN Validation MAP: {map_cnn:.4f}, NDCG: {ndcg_cnn:.4f}")
    print(f"Bi-LSTM training time: {lstm_time/60:.2f} minutes")
    print(f"Simple CNN training time: {cnn_time/60:.2f} minutes")

    # STEP 8: Choose best model and produce final test predictions
    best_nn_model = model_lstm if (map_lstm > map_cnn or (map_lstm == map_cnn and ndcg_lstm >= ndcg_cnn)) else model_cnn
    print(f"\nSelected Best NN Model: {'Bi-LSTM' if best_nn_model is model_lstm else 'Simple CNN'}")
    generate_nn_txt(best_nn_model, vocab)

    end = timer()
    print(f"\nTask 4 completed in {(end - start)/60:.2f} minutes.")
