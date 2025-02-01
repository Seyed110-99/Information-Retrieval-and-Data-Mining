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

# Ensure required NLTK resources are available
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Handle SSL certificate issues if present
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass


# --- 1. Data Loading ---
def read_passage_file(path):
    """
    Reads the file at the given path using tab as a delimiter.
    Expects one passage per row and returns a DataFrame with column 'passage'.
    """
    try:
        df = pd.read_csv(path, delimiter='\t', header=None)
        df.columns = ['passage']
        return df
    except FileNotFoundError:
        print(f"Error: File not found - {path}")
        return pd.DataFrame(columns=['passage'])
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(columns=['passage'])


# --- 2. Text Tokenization and Preprocessing ---
def preprocess_text(text, remove_stop=False, do_lemmatize=False, do_stem=False):
    """
    Lowercases and cleans the input text by removing non-alphabet characters.
    Then tokenizes the text using NLTK.
    Optionally removes stopwords, performs lemmatization and/or stemming.
    """
    # Convert to lowercase and remove unwanted characters
    cleaned = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    tokens = nltk.word_tokenize(cleaned)

    if remove_stop:
        stops = set(stopwords.words('english'))
        tokens = [w for w in tokens if w not in stops]
    if do_lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    if do_stem:
        stemmer = SnowballStemmer('english')
        tokens = [stemmer.stem(w) for w in tokens]
    return tokens


# --- 3. Word Frequency Calculation ---
def compute_frequency(token_list):
    """
    Computes the frequency of each token in the provided list.
    Returns a DataFrame with columns: 'word', 'count', and 'normalised_freq'.
    """
    freq_series = pd.Series(token_list).value_counts()
    freq_df = freq_series.reset_index()
    freq_df.columns = ['word', 'count']
    total = freq_df['count'].sum()
    freq_df['normalised_freq'] = freq_df['count'] / total
    return freq_df


# --- 4. Theoretical Zipf Frequencies ---
def get_zipf_distribution(freq_df):
    """
    Computes the theoretical Zipf distribution (normalized) for the vocabulary.
    Returns an array of ranks and their corresponding Zipf frequencies.
    """
    ranks = np.arange(1, len(freq_df) + 1)
    zipf_vals = 1 / ranks
    zipf_vals = zipf_vals / zipf_vals.sum()
    return zipf_vals, ranks


# --- 5. Plotting Distributions ---
def save_distribution_plot(freq_df, plot_num, remove_stop, scale="log-log"):
    """
    Generates and saves a plot comparing the empirical frequency distribution to
    the theoretical Zipf distribution. The plot is saved as a PDF with a filename
    following the pattern: Task_1_<plot_num>_fig.pdf.
    """
    zipf_vals, ranks = get_zipf_distribution(freq_df)
    plt.figure()
    if scale == "linear":
        plt.plot(ranks, zipf_vals, '-', label="Zipf's Distribution")
        plt.plot(ranks, freq_df['normalised_freq'], '--', label="Empirical Distribution")
    else:
        plt.loglog(ranks, zipf_vals, '-', label="Zipf's Distribution")
        plt.loglog(ranks, freq_df['normalised_freq'], '--', label="Empirical Distribution")
    plt.xlabel("Rank (log)")
    plt.ylabel("Frequency (log)")
    if remove_stop:
        plt.title("Empirical vs. Zipf's Law (No Stopwords)")
    else:
        plt.title("Empirical vs. Zipf's Law (With Stopwords)")
    plt.legend()
    plt.savefig(f'Task_1_{plot_num}_fig.pdf', format='pdf')
    plt.close()


def create_all_plots(with_stop_df, without_stop_df):
    """
    Generates the three required plots:
      1. Log-log plot with stopwords
      2. Log-log plot without stopwords
      3. Linear-scale plot with stopwords
    """
    save_distribution_plot(with_stop_df, plot_num=1, remove_stop=False)
    save_distribution_plot(without_stop_df, plot_num=2, remove_stop=True)
    save_distribution_plot(with_stop_df, plot_num=3, remove_stop=False, scale="linear")


# --- 7. Main Execution ---
if __name__ == "__main__":
    start_time = timer()

    # Load the passage data
    passage_file = "passage-collection.txt"
    data_df = read_passage_file(passage_file)
    all_text = " ".join(data_df['passage'])

    # Tokenize without removing stopwords
    tokens_with = preprocess_text(all_text)
    freq_with = compute_frequency(tokens_with)

    # Tokenize with stopword removal
    tokens_without = preprocess_text(all_text, remove_stop=True)
    freq_without = compute_frequency(tokens_without)

    # Print required statistics
    print(f"Total number of tokens: {freq_with['count'].sum()}")
    print(f"Total number of tokens (no stop words): {freq_without['count'].sum()}")
    print(f"Vocabulary size (with stopwords): {len(freq_with)}")
    print(f"Vocabulary size (without stopwords): {len(freq_without)}")

    # Generate and save the plots
    create_all_plots(freq_with, freq_without)

    end_time = timer()
    print(f"Process completed in {((end_time - start_time)/60):.2f} minutes.")
