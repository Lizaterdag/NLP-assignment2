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

# Write the dictionary to a file
with codecs.open('word_to_index_100.txt', 'w', encoding='utf-8') as wf:
    # Convert the dictionary to a string and write it to the file
    wf.write(str(word_index_dict))
wf.close()

f = codecs.open("brown_100.txt")

#initialize numpy 0s array
vocab_size = len(word_index_dict)
counts = np.zeros((vocab_size, vocab_size), dtype=int)

#iterate through file and update counts
for line in f:
    words = line.strip().split()
    if len(words) > 1:
        prev_word = '<s>' 
        for word in words:
            if word in word_index_dict and prev_word in word_index_dict:
                counts[word_index_dict[prev_word], word_index_dict[word]] += 1
            prev_word = word
f.close()

#normalize counts
prob_matrix = normalize(counts, norm='l1', axis=1)

#writeout bigram probabilities
output_probs = [
    ('the', 'all'),
    ('jury', 'the'),
    ('campaign', 'the'),
    ('calls', 'anonymous')
]

with codecs.open('bigram_probs.txt', 'w', encoding='utf-8') as out_file:
    for prev_word, word in output_probs:
        if prev_word in word_index_dict and word in word_index_dict:
            prob = prob_matrix[word_index_dict[prev_word], word_index_dict[word]]
            out_file.write(f"p({word} | {prev_word}) = {prob}\n")