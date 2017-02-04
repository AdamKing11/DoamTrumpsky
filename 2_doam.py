#-*- coding: utf-8 -*-
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Embedding, Bidirectional
from keras.optimizers import RMSprop
from keras.models import load_model
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

def ngram_chars(s, n = 2, clean = True):
    """
    takes a string and breaks in down into ngrams over characters
    just like C. Shannon would've wanted.... :'(
    """
    chars_to_clean = u'~©®¢¥%„°$­£€«■·|}_{<^!Q?\>►±&»\§/\\•\*\(\):;\[\]=+—'.encode("utf-8")    
    chars_to_clean = "[" + chars_to_clean + "]"
    ngrams = []
    if clean:
        s = re.sub(chars_to_clean, "", s)
    
    for i in range(len(s)-n+1):
        ngrams.append(s[i:i+n])
    
    if clean:
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

n_size = 2
chom = read_corpus("CHOMSKY/cleaned_all.txt", n_size)


fchom = flatten(chom)#[:10000]
fchom_ngrams = set(fchom)

print(len(chom))
print(len(fchom))
print(len(fchom_ngrams))


# index to ngram
itn = dict((x,y) for x,y in enumerate(fchom_ngrams)) 
# ngram to index
nti = dict((y,x) for x,y in enumerate(fchom_ngrams)) 

slice_size = 30
slice_step = 20
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
file_name = "saved/" + str(n_size) +"doamdoam.mod"
print("Ivana!")

doam = Sequential()

doam.add(LSTM(128,input_shape=(slice_size,len(nti))))
doam.add(Dense(len(nti), activation="softmax"))

doam.compile(loss="categorical_crossentropy", optimizer="adam")


doam.save(file_name)

"""
print("loadinggg")
doam = load_model(file_name)
"""
doam.fit(X, y, batch_size=128, nb_epoch=10)


prediction_base = ngram_chars("endorsed @MittRomney not becau", n_size, clean=False)
#prediction_base = ngram_chars("endorsed @MittRomney not becau", 1, clean=False)

for i in range(20):
    # take our sting of ngrams and make them into 
    P = np.zeros((1,slice_size, len(nti)))
    for i, n in enumerate(prediction_base):
        P[0,i] = np.zeros(len(nti))
        P[0,i,nti[n]] = 1
    # gives back vector of length = (number of char ngrams) with probabilities
    # for each
    predictions = doam.predict(P,verbose = 0)

    predictions = predictions[0]
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions)
    predictions = np.exp(predictions) / sum(np.exp(predictions))

    next = np.argmax(np.random.multinomial(1,predictions,1))
    
    prediction_base = prediction_base[1:] + [itn[next]]

print("".join(prediction_base))

