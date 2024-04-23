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

#iterate through file and update counts
previous_word = '<s>'
for line in f:
    for word in line.split():
        word = word.lower()
        if word in word_index_dict:
            counts[word_index_dict[previous_word]][word_index_dict[word]] += 1
        previous_word = word
f.close()

counts += 0.1

#normalize counts
prob_matrix = normalize(counts, norm='l1', axis=1)

#writeout bigram probabilities
output_probs = [
    ('the', 'all'),
    ('jury', 'the'),
    ('campaign', 'the'),
    ('calls', 'anonymous')
]

with codecs.open('smooth_probs.txt', 'w', encoding='utf-8') as out_file:
    for prev_word, word in output_probs:
        if prev_word in word_index_dict and word in word_index_dict:
            prob = prob_matrix[word_index_dict[word], word_index_dict[prev_word]]
            out_file.write(f"p({prev_word} | {word}) = {prob}\n")