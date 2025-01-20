import pandas as pd
import re 
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
def load_data(file_path):
    # Load the data using pandas
    data = pd.read_csv(file_path, delimiter='\t', header=None)
    # Rename the column
    data.columns = ['passage'] 
    return data

data = load_data('passage-collection.txt')


# Clean the text and tokenise it
def tokenise_txt(text):
    print("inside tokenise text")
    # Remove punctuation
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    # Tokenise the text
    #print(clean_text)
    tokens = clean_text.split()
    return tokens
all_data = " ".join(data['passage'])
tokens = tokenise_txt(all_data) 

# Count the frequency of each word
def calculate_word_frequencies(tokens):
    print("inside calculate word frequencies")
    # Count the frequency of each word
    term_counts = Counter(tokens)
    # Normalise the frequency of each word
    total_terms = sum(term_counts.values())
    normalised_freq = {term: count/total_terms for term, count in term_counts.items()}
    return normalised_freq

term_counts = calculate_word_frequencies(tokens)

term_counts_np = np.array(list(term_counts.values()))

ranks = np.arange(1, len(term_counts_np) + 1)
s = 1  # Zipf's parameter
zipf_frequencies = 1 / ranks**s
zipf_frequencies /= zipf_frequencies.sum() 

vocabulary_size = len(term_counts)
print(f"Vocabulary size: {vocabulary_size}")

# Plot empirical and theoretical distributions
plt.plot(ranks, np.sort(term_counts_np)[::-1], label='Empirical')
plt.plot(ranks, zipf_frequencies, label='Theoretical (Zipf)', linestyle='--')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.legend()
plt.title('Empirical vs. Theoretical Frequencies')
plt.savefig('Task1_fig.pdf', format='pdf')
plt.show()
