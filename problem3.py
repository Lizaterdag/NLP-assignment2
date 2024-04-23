#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import codecs
import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random


vocab = codecs.open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    #TODO: import part 1 code to build dictionary 
    word_index_dict[line.rstrip()] = i

f = codecs.open("brown_100.txt")

#TODO: initialize numpy 0s array
counts = np.zeros((len(word_index_dict), len(word_index_dict)))

#TODO: iterate through file and update counts
previous_word = '<s>'
for line in f:
    for word in line.split():
        word = word.lower()
        if word in word_index_dict:
            counts[word_index_dict[previous_word]][word_index_dict[word]] += 1
            #print(counts[word_index_dict[previous_word]][word_index_dict[word]])
        previous_word = word
f.close()

#TODO: normalize counts
probs = normalize(counts, norm='l1', axis=1)

#TODO: writeout bigram probabilities
 
print("p(the | all):", probs[word_index_dict['all']][word_index_dict['the']])
print("p(jury | the):", probs[word_index_dict['the']][word_index_dict['jury']])
print("p(campaign | the):", probs[word_index_dict['the']][word_index_dict['campaign']])
print("p(calls | anonymous):", probs[word_index_dict['anonymous']][word_index_dict['calls']])
