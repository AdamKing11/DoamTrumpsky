from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Embedding, Bidirectional
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.utils.np_utils import to_categorical
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
    takes a 2-d list of lists and makes in 1-d
    concatenate all the vectors!
    """
    return [x for y in l for x in y]

def vectorize(wl, by_char = False):
    """
    takes a bunch of lists of words and replaces them with vectors of integer indeces
    """
    char_to_int = {}
    int_to_char = {}
    vl = []

    for w in wl:
        if by_char:
            v = []
            for c in w:
                if c not in char_to_int:
                    c_ind = len(char_to_int) + 1
                    char_to_int[c] = c_ind
                    int_to_char[c_ind] = c
                v.append(char_to_int[c])
            vl.append(v)
            v = []
        else:
            if w not in char_to_int:
                w_ind = len(char_to_int) + 1
                char_to_int[w] = w_ind
                int_to_char[w_ind] = w
            vl.append(char_to_int[w])
    
    return vl, int_to_char, char_to_int

def make_one_hot(v, categories):
    """
    takes a vector 'v' of word indices and returns a list of
    1-hot vectors that correspond to the indices

    e.g. [1, 2, 3] -> [[1,0,0], [0,1,0], [0,0,1]]
    """

    new_v = np.zeros((len(v), categories), dtype="int8")
    for i, j in enumerate(v):
        # j-1 because arrays are 0-index
        new_v[i,j-1] = 1
    return new_v



chomps = flatten(read_corpus("CHOMSKY/BigC.txt"))
trumps = flatten(read_corpus("TRUMP/BigT.txt"))

both = chomps + trumps

#both = both[0:10000]

# get the list ready to convert to list of integers to stand in for words
words = sorted(list(set(both)))
#word_indices = dict((w, i) for i, w in enumerate(words))
#indices_word = dict((i, w) for i, w in enumerate(words))

_, indices_word, word_indices = vectorize(both)

print("#"*20)
print("Word types in both:", len(words))
print("Tokens in both:", len(both))

# build ngrams... or whatever grams
gram_size = 8
step = 4
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
Y = np.zeros((len(ngrams), 1), dtype="int32")

print("X shape --", X.shape)
print("Y shape --", Y.shape)

print(len(ngrams), "ngrams to do.")

# build the ngrams we use to train the model
# go through the ngrams and replace each word with its integer index in that
# dictionary we made earlier
for i, ng in enumerate(ngrams):
    for j, w in enumerate(ng):
        X[i, j] = word_indices[w]
    Y[i] = word_indices[next_words[i]]

###
print("#"*20)
# make the output a one-hot vector for each possible word
Y = make_one_hot(Y, len(words))
###


# check the shape/value of the data
print("X --", X.shape)
print("Y --", Y.shape)

b_size = 32
print("Building the model. It will be fabulous, great. No one does it better.")
model = Sequential()

model.add(Embedding(len(words)+1, 64, input_shape=(b_size, gram_size)))

model.add(Bidirectional(LSTM(64)))

model.add(Dense(len(words), activation="softmax"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#print(model.summary())
print("Fitting....")
model.fit(X, Y, batch_size=b_size, nb_epoch=3)


####################################################
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


start_index = random.randint(0, len(both) - gram_size - 1)
generated = ""
sentence = both[start_index: start_index + gram_size]
print(sentence, generated)

for w in sentence:
    generated += w + " "

print("Starting with::\n\t", generated)
# let's generate 20 random words based on the model
for i in range(3):
    x = np.zeros(5)
    
    for i, w in enumerate(sentence[-6:-1]):
        x[i] = word_indices[w]
        
    preds = model.predict(x, verbose=0)
    preds = np.asarray(preds).astype('float64')


    pred_word = indices_word[np.argmax(preds[-1])]
    generated += pred_word + " "
    sentence = sentence[1:] + [pred_word]

print(generated)
