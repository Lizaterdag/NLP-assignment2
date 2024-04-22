#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import codecs

vocab_file_path = 'brown_vocab_100.txt'
output_file_path = 'word_to_index_100.txt'

word_index_dict = {}

# read brown_vocab_100.txt into word_index_dict
with codecs.open(vocab_file_path, 'r', encoding='utf-8') as file:
    for index, line in enumerate(file):
        word_index_dict[line.rstrip()] = index

# write word_index_dict to word_to_index_100.txt
with codecs.open(output_file_path, 'w', encoding='utf-8') as wf:
    # Convert the dictionary to a string and write it to the file
    wf.write(str(word_index_dict))


print(word_index_dict['all'])
print(word_index_dict['resolution'])
print(len(word_index_dict))
