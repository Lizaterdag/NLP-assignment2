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

vocab_size = len(word_index_dict)

#writeout trigram probabilities
output_probs = [
    ('past', ('in', 'the')),
    ('time', ('in', 'the')),
    ('said', ('the', 'jury')),
    ('recommended', ('the', 'jury')),
    ('that', ('jury', 'said')),
    (',', ('agriculture', 'teacher')),
]

def get_trigram_probs(corpus_path, target_word, target_prev_prev_word, target_prev_word, vocab_size=0, smoothing=False):
    f = codecs.open(corpus_path)

    curr_prev_prev_word = '<s>'
    curr_prev_word = '<s>'
    bigram_count = 0
    target_count = 0
    for line in f:
        for curr_word in line.split():
            curr_word = curr_word.lower()
            if curr_prev_prev_word == target_prev_prev_word and curr_prev_word == target_prev_word:
                bigram_count += 1
                if curr_word == target_word:
                    target_count += 1
            curr_prev_prev_word = curr_prev_word
            curr_prev_word = curr_word

    f.close()

    if smoothing:
        target_count += 0.1
        bigram_count += 0.1 * vocab_size #Each word after the bigram is seen at least 0.1 times with smoothing, so the sum would be 0.1 * vocab_size

    if bigram_count == 0 and not smoothing:
        return 0.0
    
    return target_count/bigram_count

with codecs.open('trigram_probs.txt', 'w', encoding='utf-8') as out_file:
    for word, (prev_prev_word, prev_word) in output_probs:
        if word in word_index_dict and prev_prev_word in word_index_dict and prev_word in word_index_dict:
            prob = get_trigram_probs("brown_100.txt", word, prev_prev_word, prev_word)
            out_file.write(f"p({word} | {prev_prev_word}, {prev_word}) = {prob}\n")

with codecs.open('smoothed_trigram_probs.txt', 'w', encoding='utf-8') as out_file:
    for word, (prev_prev_word, prev_word) in output_probs:
        if word in word_index_dict and prev_prev_word in word_index_dict and prev_word in word_index_dict:
            prob = get_trigram_probs("brown_100.txt", word, prev_prev_word, prev_word, vocab_size, True)
            out_file.write(f"p({word} | {prev_prev_word}, {prev_word}) = {prob}\n")