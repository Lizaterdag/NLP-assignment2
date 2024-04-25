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
counts = np.zeros((vocab_size, vocab_size), dtype=int)

#iterate through file and update counts
previous_word = '<s>'  # Start of sentence marker
for line in f:
    words = line.strip().split() + ['</s>']  # Add end of sentence marker
    for word in words:
        word = word.lower()
        if previous_word in word_index_dict and word in word_index_dict:
            counts[word_index_dict[previous_word], word_index_dict[word]] += 1
        previous_word = word  # Update previous word
    previous_word = '<s>'  # Reset to start for next sentence
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
    for word, prev_word in output_probs:
        if prev_word in word_index_dict and word in word_index_dict:
            prob = prob_matrix[word_index_dict[prev_word], word_index_dict[word]]
            out_file.write(f"p({prev_word} | {word}) = {prob}\n")


#####
#PROBLEM 6
#####
with codecs.open('bigram_eval.txt', 'w', encoding='utf-8') as out_file, \
     codecs.open("toy_corpus.txt", encoding='utf-8') as toy_corpus:
    
    for line in toy_corpus:
        words = ['<s>'] + line.strip().split()
        sentprob = 1
        previous_word = '<s>'

     # Calculate the joint probability of the sentence
    for current_word in words[1:]:  # Start from the first actual word after <s>
        current_word = current_word.lower()
        if previous_word in word_index_dict and current_word in word_index_dict:
            idx_prev = word_index_dict[previous_word]
            idx_curr = word_index_dict[c000000000urrent_word]
            wordprob = prob_matrix[idx_prev, idx_curr]
            sentprob *= wordprob
        else:
            # If the bigram is not found, assign a small probability (smoothing could be applied here)
            sentprob *= 1e-12
        previous_word = current_word

    # Calculate sentence length excluding the start symbol <s>
    sent_len = len(words) - 1
    # Calculate perplexity
    if sentprob > 0:
        perplexity = 1 / (pow(sentprob, 1.0 / sent_len))
    else:
        perplexity = float('inf')  # Handle case where sentprob is 0 (logically impossible in this setup due to smoothing)

    # Write the perplexity of the sentence to the output file
    out_file.write(f"{perplexity}\n")


