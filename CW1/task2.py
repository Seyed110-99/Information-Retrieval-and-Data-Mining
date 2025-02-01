import os
import pandas as pd
from timeit import default_timer as timer
import task1 as t1  # Ensure task1.py (with preprocess_text) is in the same directory
import json
from collections import Counter

def build_inverted_index(documents):
    """
    Constructs an inverted index from an iterable of (pid, token_list) pairs.
    For each token, the index stores a dictionary mapping each document ID (pid)
    to the term's frequency in that document.
    """
    inverted = {}
    for doc_id, tokens in documents:
        counts = Counter(tokens)
        for term, freq in counts.items():
            if term not in inverted:
                inverted[term] = {}
            inverted[term][doc_id] = freq
    return inverted

if __name__ == "__main__":
    start = timer()

    # Read the candidate passages file.
    # The file is expected to have four tab-separated columns: qid, pid, query, and passage.
    df = pd.read_csv('candidate-passages-top1000.tsv', sep='\t',
                     names=['qid', 'pid', 'query', 'passage'], dtype='str')

    # Tokenize each passage using the preprocess_text function from task1.py.
    # The parameters below remove stopwords and apply both lemmatization and stemming.
    df['tokenised_passage'] = df['passage'].apply(
        lambda x: t1.preprocess_text(x, remove_stop=True, do_lemmatize=True, do_stem=True)
    )

    # Create an iterator over (pid, tokenised_passage) pairs.
    pid_tokens = zip(df['pid'], df['tokenised_passage'])

    # Build the inverted index.
    inv_index = build_inverted_index(pid_tokens)

    # Save the processed passages to "processed_passage.json".
    proc_passages = dict(zip(df['pid'].tolist(), df['tokenised_passage'].tolist()))
    with open("processed_passage.json", "w") as fout:
        json.dump(proc_passages, fout, indent=2)

    # Save the inverted index to "inverted_index.json".
    with open("inverted_index.json", "w") as fout:
        json.dump(inv_index, fout, indent=2)

    end = timer()
    print(f"Task 2 Number of vocabulary: {len(inv_index)}")
    print(f"Time taken: {((end - start) / 60):.2f} minutes")
