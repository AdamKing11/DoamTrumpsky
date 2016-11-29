from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Embedding, Bidirectional
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys



def read_corpus(infile):
    """
    reads in a plain-text file and returns a list of the sentences within
    """
    c = []
    with open(infile, "r") as rf:
        for l in rf:
            c.append(l.rstrip().rsplit(" "))
    return c

def flatten(l):
    """
    takes a 2-d
    """
    return [x for y in l for x in y]

chomps = flatten(read_corpus("CHOMSKY/BigC.txt"))
trumps = flatten(read_corpus("TRUMP/BigT.txt"))

both = chomps + trumps

sub = 15000

split = int(.8 * sub)
#both = both[0:sub]
# get the list ready to convert to list of integers to stand in for words
words = sorted(list(set(both)))
word_indices = dict((w, i) for i, w in enumerate(words))
indices_word = dict((i, w) for i, w in enumerate(words))

print("Words in both:", len(words))
print("Tokens in both:", len(both))

# build ngrams... or whatever grams
gram_size = 8
step = 2
ngrams = []
next_words = []
# go through the list of all words and make ngrams
for i in range(0, len(both) - gram_size, step):
# what we're doing here is making ngrams AND for each ngram, we store the next word
# that way, we can try to PREDICT the next word, given an ngram
    ngrams.append(both[i: i + gram_size])
    next_words.append(both[i + gram_size])

# make a list where we've replaced the actual words with the integer value
X = np.zeros((len(ngrams), gram_size), dtype="int32")
Y = np.zeros((len(ngrams)), dtype="int32")
print("X shape --", X.shape)
print("Y shape --", Y.shape)

print(len(ngrams), "ngrams to do.")
for i, ng in enumerate(ngrams):
    for j, w in enumerate(ng):
        X[i, j] = word_indices[w]
    Y[i] = word_indices[next_words[i]]


print("X --", X.shape)
print("Y --", Y.shape)
print("X[0]", X[0])
print("Y[0]", Y[0])


print('Building model...')
model = Sequential()

model.add(Embedding(len(words), 32, input_length=gram_size, mask_zero=True))

model.add(LSTM(32))

model.add(Dense(len(words), activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, batch_size=32, nb_epoch=3)
