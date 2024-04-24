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
        word.lower()
        if word in word_index_dict:
            counts[word_index_dict[word]] += 1
f.close()

#normalize and writeout counts. 
probs = counts / np.sum(counts)

start_of_sentence_char = "<s>"
end_of_sentence_char = "</s>"

start_seen = False
end_seen = True

f = open("toy_corpus.txt")
for line in f:
    total_prob = 0
    for i, word in enumerate(line.split()):
        word.lower()
        if word in word_index_dict and word != start_of_sentence_char and word != end_of_sentence_char:
            if i == 1:
                total_prob = probs[word_index_dict[word]]
            elif i > 1:
                total_prob *= probs[word_index_dict[word]]
    print(total_prob)
    #     else:
    #         break #invalid word as the word is not between start or end of sentence characters.

f.close()



