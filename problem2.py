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
    #import part 1 code to build dictionary
    word_index_dict[line.rstrip()] = i
    
f = open("brown_100.txt")

#initialize counts to a zero vector
counts = np.zeros(len(word_index_dict))

#iterate through file and update counts
for line in f:
    for word in line.split():
        word = word.lower()
        if word in word_index_dict:
            counts[word_index_dict[word]] += 1
f.close()

#normalize and writeout counts. 
probs = counts / np.sum(counts)
    
np.savetxt('unigram_probs.txt', probs)





