#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from generate import GENERATE
import codecs

vocab = open("brown_vocab_100.txt")
#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    word_index_dict[line.rstrip()] = i

f = open("brown_100.txt")

counts = np.zeros(len(word_index_dict))

for line in f:
    for word in line.split():
        word.lower()
        if word in word_index_dict:
            counts[word_index_dict[word]] += 1

f.close()
probs = counts / np.sum(counts)

np.savetxt('unigram_probs.txt', probs)




