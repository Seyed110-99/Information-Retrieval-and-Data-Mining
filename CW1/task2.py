import os
import pandas as pd
from timeit import default_timer as timer
import task1 as t1
import json
from collections import Counter

def find_inverted_index(pid_dict):
    """
    Create an inverted index from the tokenised text.
    """
    inverted_index = {}

    for pid, tokens in pid_dict:
        term_frequencies = Counter(tokens)
        for token, freq in term_frequencies.items():
            if token not in inverted_index:
                inverted_index[token] = {}
            inverted_index[token][pid] = freq
    
    return inverted_index


if __name__ == "__main__":
    start = timer()
    # Load data
    data_raw_txt_df = pd.read_csv('candidate-passages-top1000.tsv', sep='\t', 
                                  names=['qid', 'pid', 'query', 'passage'], dtype='str')
    # Tokenise the 'passage' column
    data_raw_txt_df['tokenised_passage'] = data_raw_txt_df['passage'].apply(
    lambda x: t1.tokenise_txt(x, remove_stopwords=True, lemmatise=True, stem=True))
    
    # Create an inverted index
    pid_and_tokens = zip(data_raw_txt_df['pid'], data_raw_txt_df['tokenised_passage'])
    inverted_index = find_inverted_index(pid_and_tokens)

    # Pid file save to json
    pid_and_tokens_lists = zip(data_raw_txt_df['pid'].tolist(), data_raw_txt_df['tokenised_passage'].tolist())

    with open("processed_passage.json", "w") as f:
        json.dump(dict(pid_and_tokens_lists), f, indent=2)
    # Save the inverted index to a JSON file
    with open("inverted_index.json", "w") as f:
        json.dump(inverted_index, f, indent=2)


    end = timer()
    print(f"Task 2 Number of vocabulary: {len(inverted_index)}")
    print(f"Time taken: {((end-start)/60):.2f} minutes")
