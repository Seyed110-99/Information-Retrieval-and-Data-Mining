import os
import csv 
import pandas as pd
import re 
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data

data = pd.read_csv('passage-collection.txt', delimiter='\t')

print(data.head())

# Clean the text

clean_text = re.sub(r'[^\w\s]', '', data['text'][0].lower())

print(clean_text.head())