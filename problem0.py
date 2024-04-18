#!/usr/bin/env python3
from nltk.corpus import brown
from collections import Counter
import string

brown_corpus = brown.words()
word_count = Counter(brown_corpus)
word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

punc = set(string.punctuation + "``" + "''" + "..." + "''" + "``" + "!" + "?" + "." + "," + ":" + ";" + "-")
unique_words = [(word, freq) for word, freq in word_count if word not in punc]

print("Unique words:")
for word, freq in unique_words[:20]:  
    print(word, freq)
