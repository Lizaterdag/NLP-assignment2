#!/usr/bin/env python3
from nltk.corpus import brown
from collections import Counter
import string

brown_corpus = brown.words()

bad_chars = ["``" , "''" , "..." , "''" , "``" , "!" , "?" , "." , "," , ":" , ";" , "-"]

def get_word_freq(corpus):
    count = Counter(corpus)
    words = sorted(count.items(), key=lambda x: x[1], reverse=True)
    return [(word, freq) for word, freq in words if word.strip() and word not in bad_chars]


unique_words = get_word_freq(brown_corpus) 
print("Unique words:")
for word, freq in unique_words[:20]:  
    print(word, freq)

romance = get_word_freq(brown.words(categories='romance'))
humor = get_word_freq(brown.words(categories='humor'))


print("\ngenre romance :")
for word, freq in romance[:20]:  
    print(word, freq)

print("\ngenre humor :")
for word, freq in humor[:20]:  
    print(word, freq)