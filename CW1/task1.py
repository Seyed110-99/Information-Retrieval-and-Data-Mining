import pandas as pd
import re 
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# Load the data
def load_data(file_path):
    # Load the data using pandas
    data = pd.read_csv(file_path, delimiter='\t', header=None)
    # Rename the column
    data.columns = ['passage'] 
    return data

# Clean the text and tokenise it
def tokenise_txt(text):
    print("inside tokenise text")
    # Remove punctuation
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    # Tokenise the text
    #print(clean_text)
    tokens = clean_text.split()
    return tokens

# Count the frequency of each word
def calculate_word_frequencies(tokens):
    print("inside calculate word frequencies")
    # Count the frequency of each word
    term_counts = Counter(tokens)
    # Normalise the frequency of each word
    total_terms = sum(term_counts.values())
    normalised_freq = {term: count/total_terms for term, count in term_counts.items()}
    return normalised_freq



def plot(term_counts,num):
   term_counts_np = np.array(list(term_counts.values()))
   ranks = np.arange(1, len(term_counts_np) + 1)
   s = 1  # Zipf's parameter
   zipf_frequencies = 1 / ranks**s
   zipf_frequencies /= zipf_frequencies.sum() 
   vocabulary_size = len(term_counts)
   print(f"Vocabulary size: {vocabulary_size}")
   # Plot empirical and theoretical distributions

   # Plot the empirical distribution
   plt.plot(ranks, np.sort(term_counts_np)[::-1], label='Empirical')
   # Plot the theoretical Zipf distribution
   plt.plot(ranks, zipf_frequencies, label='Theoretical (Zipf)', linestyle='--')
   # Set the scale to log-log
   plt.xscale('log')
   plt.yscale('log')
   plt.xlabel('Rank')
   plt.ylabel('Frequency')
   plt.legend()
   plt.title('Empirical vs. Theoretical Frequencies')
   plt.savefig('Task_1_'+str(num)+'_fig.pdf', format='pdf')
   plt.show()

data = load_data('passage-collection.txt')
all_data = " ".join(data['passage'])
tokens = tokenise_txt(all_data) 
term_counts = calculate_word_frequencies(tokens)
print(f"Vocabulary size (with stop words): {len(term_counts)}")
plot(term_counts,1)

stop_words = set(stopwords.words('english'))
tokens_no_stopwords = [word for word in tokens if word not in stop_words]
term_counts_no_stopwords = calculate_word_frequencies(tokens_no_stopwords)
print(f"Vocabulary size (without stop words): {len(term_counts_no_stopwords)}")

plot(term_counts_no_stopwords,2)