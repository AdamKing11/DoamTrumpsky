from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Embedding, Bidirectional
from keras.optimizers import RMSprop
import numpy as np
import random
import sys, re

def read_corpus(infile, n = 2):
    """
    opens up a plain-text file and returns a list of sentences,
    as lists of their ngrams (after a little cleaning)
    """
    c = []
    with open(infile, "r") as rf:
        for l in rf:
            l = l.rstrip()
            # instead, break in to ngrams over phones
            ngrams = ngram_chars(l, n)
            c.append(ngrams)
    return c

def ngram_chars(s, n = 2):
    """
    takes a string and breaks in down into ngrams over characters
    just like C. Shannon would've wanted.... :'(
    """
    chars_to_clean = "~@©®¢¥%„°$­£€«■·|}_{<^!Q?\>►±&»\§/\\•\*\(\):;\[\]=+—"
    chars_to_clean = "[" + chars_to_clean + "]"
    ngrams = []
    s = re.sub(chars_to_clean, "", s)
    for i in range(len(s)-n+1):
        ngrams.append(s[i:i+n])
    ngrams = [n for n in ngrams if not re.search("([a-z][A-Z]|[A-Za-z][0-9]|[0-9][A-Za-z])",n)]
    return ngrams

def flatten(l):
    """
    takes a 2-d list of lists and makes in 1-d
    concatenate all the vectors!
    """
    return [x for y in l for x in y]

def ngram_freq(ngrams):
    return dict((n,ngrams.count(n)) for n in set(ngrams))

chom = read_corpus("CHOMSKY/cleaned_all.txt", 1)
#for i in chom:
#    for j in i:
#        print(j)


fchom = flatten(chom)
fchom_ngrams = set(fchom)

print(len(chom))
print(len(fchom))
print(len(fchom_ngrams))

#fchom_freq = ngram_freq(fchom)
#for i in sorted(fchom_freq, key=fchom_freq.get, reverse=True):
#    print(i, fchom_freq[i])

# index to ngram
itn = dict((x,y) for x,y in enumerate(fchom_ngrams)) 
# ngram to index
nti = dict((y,x) for x,y in enumerate(fchom_ngrams)) 

slice_size = 20
slice_step = 10
slice_count = int((len(fchom)-slice_size)/slice_step)+1
slices = np.zeros((slice_count,slice_size+1), dtype="int16")
j = 0
for i in range(0,len(fchom)-slice_size-1,slice_step):
    slices[j] = np.array([nti[x] for x in fchom[i:i+slice_size+1]], dtype="int16")
    j+=1

print(slices.shape)
# free up some memory...
fchom = []
fchom_ngrams = []
X = np.zeros((slice_count,slice_size,len(nti)), dtype="int8")
y = np.zeros((slice_count,len(nti)), dtype="int8")

print(X.shape)
print(y.shape)

for i, slc in enumerate(slices):
    for j,ind in enumerate(slc[:-1]):
        X[i,j] = np.zeros(len(nti))
        X[i,j,ind] = 1
    y[i] = np.zeros(len(nti))
    y[i,slc[-1]] = 1

# free up some more memory.....
slices = []

doam = Sequential()

doam.add(LSTM(64,input_shape=(slice_size,len(nti))))
doam.add(Dense(len(nti), activation="softmax"))

doam.compile(loss="categorical_crossentropy", optimizer="adam")

doam.fit(X, y, batch_size=128, nb_epoch=1)