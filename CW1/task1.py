import os
from timeit import default_timer as timer
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import ssl

# Setup NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Ignore SSL certificate errors
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# **1. Load the Data**
def load_data(file_path):
    """
    Load the data from the given file path.
    """
    # Check if the file exists
    try:
        data = pd.read_csv(file_path, delimiter='\t', header=None)
        data.columns = ['passage']
        return data
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return pd.DataFrame(columns=['passage'])
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(columns=['passage'])

# **2. Tokenize and Preprocess Text**
def tokenise_txt(text, remove_stopwords=False, lemmatise=False, stem=False):
    """
    Tokenize and preprocess the text.
    """
    # Clean the text
    clean_text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    tokens = nltk.word_tokenize(clean_text)

    # Remove stopwords, lemmatise and/or stem the tokens
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

    if lemmatise:
        lemmatiser = WordNetLemmatizer()
        tokens = [lemmatiser.lemmatize(word) for word in tokens]

    if stem:
        stemmer = SnowballStemmer('english')
        tokens = [stemmer.stem(word) for word in tokens]

    return tokens

# **3. Calculate Word Frequencies**
def calculate_word_frequencies(tokens):
    """
    Calculate the frequency of each word in the tokens list.
    """
    # Create a DataFrame with word frequencies
    df = pd.Series(tokens).value_counts().reset_index()
    df.columns = ['word', 'count']
    df['normalised_freq'] = df['count'] / df['count'].sum()
    return df

# **4. Calculate Zipf's Frequencies**
def calculate_zipf_frequencies(df):
    """
    Calculate theoretical Zipf's frequencies for the given DataFrame.
    """
    ranks = np.arange(1, len(df) + 1)
    zipf_frequencies = 1 / ranks
    zipf_frequencies /= zipf_frequencies.sum()
    return zipf_frequencies, ranks

# **5. Plot Distributions**
def plot(df, num, remove_stopwords=False, scale="log-log"):
    """
    Plot and save a distribution comparing empirical and Zipf's frequencies.
    """
    zipf_frequencies, ranks = calculate_zipf_frequencies(df)
    plt.figure()
    if scale == "linear":
        plt.plot(ranks, zipf_frequencies, '-', label="Zipf's Distribution")
        plt.plot(ranks, df['normalised_freq'], '--', label="Empirical Distribution")
    else:
        plt.loglog(ranks, zipf_frequencies, '-', label="Zipf's Distribution")
        plt.loglog(ranks, df['normalised_freq'], '--', label="Empirical Distribution")
    plt.xlabel("Rank (log)")
    plt.ylabel("Frequency (log)")
    title = "Empirical vs. Zipf's Law (No Stopwords)" if remove_stopwords else "Empirical vs. Zipf's Law (With Stopwords)"
    plt.title(title)
    plt.legend()
    plt.savefig(f'Task_1_{num}_fig.pdf', format='pdf')
    plt.close()

# **6. Batch Plot**
def plot_batch(df, df_no_stopwords):
    """
    Generate and save plots for datasets with and without stopwords.
    """
    plot(df, num=1, remove_stopwords=False)
    plot(df_no_stopwords, num=2, remove_stopwords=True)
    plot(df, num=3, remove_stopwords=False, scale="linear")

# **7. Main Function**
if __name__ == "__main__":
    start = timer()

    # Load data
    file_path = "passage-collection.txt"
    data = load_data(file_path)
    all_data = " ".join(data['passage'])

    # Tokenize and process data
    tokens = tokenise_txt(all_data)
    term_counts_df = calculate_word_frequencies(tokens)

    tokens_no_stopwords = tokenise_txt(all_data, remove_stopwords=True)
    term_counts_no_stopwords = calculate_word_frequencies(tokens_no_stopwords)
    
    # Print insights
    print(f"Total number of tokens: {term_counts_df['count'].sum()}")
    print(f"Total number of tokens (no stop words): {term_counts_no_stopwords['count'].sum()}")
    print(f"Vocabulary size (with stopwords): {len(term_counts_df)}")
    print(f"Vocabulary size (without stopwords): {len(term_counts_no_stopwords)}")
    # Generate plots
    plot_batch(term_counts_df, term_counts_no_stopwords)

    end = timer()
    print(f"Process completed in {end - start:.2f} seconds.")
