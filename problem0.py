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



num_tokens = len(brown.words())
num_types = len(set(word.lower() for word in brown.words()))
num_words = len(brown.sents())
avg_words = int(num_tokens / num_words)
avg_word_len = int(sum(len(word) for word in brown.words()) / num_tokens)

print("Num of tokens:", num_tokens)
print("Num of types:", num_types)
print("Num of words:", num_words)
print("Avg num of words per sentence:", avg_words)
print("Avg word len:", avg_word_len)

# POS tagger
tagged_words = brown.tagged_words()
pos = [tag for (word, tag) in tagged_words]
pos_tags = Counter(pos).most_common(10)

print("\n10 most used POS tags:")
for tag, count in pos_tags:
    print(tag, count)