import os
from timeit import default_timer as timer
import pandas as pd
import re 
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Load the data
def load_data(file_path):
    try:
        # Load the data using pandas
        data = pd.read_csv(file_path, delimiter='\t', header=None)
        # Rename the column
        data.columns = ['passage'] 
        return data
    except FileNotFoundError:
        print("Error loading data")
        return pd.DataFrame(columns=['passage'])
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(columns=['passage'])
 

# Clean the text and tokenise it
def tokenise_txt(text, remove_stopwords=False, lemmatisation=False, stemming=False):
    print("inside tokenise text")
    # Remove punctuation
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    # Tokenise the text
    tokens = clean_text.split()
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    if lemmatisation:
        lemmatiser = WordNetLemmatizer()
        tokens = [lemmatiser.lemmatize(word) for word in tokens]
    if stemming:
        stemmer = SnowballStemmer("english")
        tokens = [stemmer.stem(word) for word in tokens]

    return tokens

# Count the frequency of each word
def calculate_word_frequencies(tokens):
    print("inside calculate word frequencies")
    # Count the frequency of each word
    df = pd.Series(tokens).value_counts().reset_index()
    # Normalise the frequency of each word
    df.columns = ['word', 'count']
    df['normalised_freq'] = df['count'] / df['count'].sum()
    return df



def plot(df, num, remove_stopwords=False):
    ranks = np.arange(1, len(df) + 1)
    zipf_frequencies = 1 / ranks
    zipf_frequencies /= zipf_frequencies.sum()
    plt.figure()
    plt.loglog(ranks, zipf_frequencies, '--', label="Zipf's Distribution")
    plt.loglog(ranks, df['normalised_freq'], '-', label="Empirical Distribution")
    plt.xlabel("Rank (log)")
    plt.ylabel("Frequency (log)")
    title = "Empirical vs. Zipf's Law (No Stopwords)" if remove_stopwords else "Empirical vs. Zipf's Law (With Stopwords)"
    plt.title(title)
    plt.legend()
    plt.savefig(f'Task_1_{num}_fig.pdf', format='pdf')

def plot_batch(df, df_no_stopwords):

    plot(df, num=1, remove_stopwords=False)
    plot(df_no_stopwords, num=2, remove_stopwords=True)

if __name__ == "__main__":
    start = timer()
    data = load_data('passage-collection.txt')
    all_data = " ".join(data['passage'])
    tokens = tokenise_txt(all_data) 
    term_counts_df = calculate_word_frequencies(tokens)
    print(f"Vocabulary size (with stop words): {len(term_counts_df)}")


    tokens_no_stopwords = tokenise_txt(all_data, remove_stopwords=True)
    term_counts_no_stopwords = calculate_word_frequencies(tokens_no_stopwords)
    print(f"Vocabulary size (without stop words): {len(term_counts_no_stopwords)}")
    end = timer()
    plot_batch(term_counts_df, term_counts_no_stopwords)

    print(f"Time taken: {end-start} seconds")