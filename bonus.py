#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import codecs

vocab = codecs.open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):    
    word_index_dict[line.strip()] = i
vocab.close()

f = codecs.open("brown_100.txt")

#initialize numpy 0s array
vocab_size = len(word_index_dict)
counts = np.zeros((vocab_size, vocab_size), dtype=float)
unigram_counts = np.zeros((vocab_size,), dtype=float)

total_pairs = 0
#iterate through file and update counts
for line in f:
    previous_word = '<s>'  # Start of sentence marker
    words = line.strip().split()  # Add end of sentence marker
    for word in words[1:]:
        word = word.lower()
        if previous_word in word_index_dict and word in word_index_dict:
            unigram_counts[word_index_dict[word]] += 1
            counts[word_index_dict[previous_word], word_index_dict[word]] += 1
            total_pairs += 1
        previous_word = word  # Update previous word

reversed_word_dict = {value: key for key, value in word_index_dict.items()}
min_freq = 10
pmi_list = []
for i in range(vocab_size):
    for j in range(vocab_size):
        freq1 = unigram_counts[i]
        freq2 = unigram_counts[j]
        if freq1 > min_freq and freq2 > min_freq and counts[i,j] > 0:
            pmi_val = np.log((counts[i,j] * total_pairs) / (freq1 * freq2))
            pmi_list.append(((reversed_word_dict[i], reversed_word_dict[j]), pmi_val))

pmi_list.sort(key=lambda x: x[1])

bottom_20 = pmi_list[:20]
top_20 = pmi_list[-20:]

with codecs.open('bonus.txt', 'w', encoding='utf-8') as out_file:
    out_file.write("Top 20 PMI Values:\n")
    for pair, value in reversed(top_20):
        out_file.write(f"{pair[0]} {pair[1]}: {value}\n")

    out_file.write("\nBottom 20 PMI Values:\n")
    for pair, value in bottom_20:
        out_file.write(f"{pair[0]} {pair[1]}: {value}\n")


            
f.close()
