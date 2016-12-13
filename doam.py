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

def count_unigrams(v):
    """
    goes through flat list 'v' and returns dict with count for each word
    """
    rd = {}
    for w in v:
        if w in rd:
            rd[w] += 1
        else:
            rd[w] = 1
    return rd

def get_top_words(guys_words,both,how_many,threshold=.75):
    """
    goes through and gets 'how_many' words where the majority -
    'threshold' or more percentage - of the tokens in the 'both'
    list show up in the 'guy' list
    e.g. if 95% of the tokens of "syntax" come from Chomsky, we 
    count that, even if Big T said it once
    """
    top_words = []
    for w in sorted(guys_words, key=lambda v:guys_words[v], reverse=True):
        if guys_words[w] / both[w] >= threshold:
            top_words.append(w)
        if len(top_words) > how_many:
            break
    return top_words

def repl_with_index(flat_vec, lex_dict):
    """
    goes through 'flat_vec' which is 
    """
    rv = []
    space_index = lex_dict[" "]
    for w in flat_vec:
        if w in lex_dict:
            rv.append(lex_dict[w])
        else:
            for c in w:
                rv.append(lex_dict[c])
        rv.append(space_index)
    return rv

def make_one_hot(v):
    """
    takes a vector 'v' of word/character indices and returns a list of
    1-hot vectors that correspond to the indices

    e.g. [1, 2, 3] -> [[1,0,0], [0,1,0], [0,0,1]]
    """
    categories = len(set(v))
    new_v = np.zeros((len(v), categories), dtype="int8")
    for i, j in enumerate(v):
        # j-1 because arrays are 0-index
        new_v[i,j-1] = 1
    return new_v

def indices_to_string(list_ind, index_dict):
    """
    """
    rs = ""
    for i in list_ind:
        rs += index_dict[i]
    return rs

nb_top_words = 25

# read in the corpora and fold case
chomps = [w.lower() for w in flatten(read_corpus("CHOMSKY/BigC.txt"))][:10000]
trumps = [w.lower() for w in flatten(read_corpus("TRUMP/BigT.txt"))][:10000]

both = chomps + trumps

chomps_words = count_unigrams(chomps)
trumps_words = count_unigrams(trumps)
both_words = count_unigrams(both)

both_top = sorted(both_words, key=lambda v: both_words[v], reverse=True)[:nb_top_words]
chomps_top = get_top_words(chomps_words, both_words, nb_top_words)
trumps_top = get_top_words(trumps_words, both_words, nb_top_words)

all_chars = set(flatten([[c for c in w] for w in both]))
lexicon_set = all_chars.union([" "])
lexicon_set = lexicon_set.union(chomps_top)
lexicon_set = lexicon_set.union(trumps_top)

lex_index = dict((l, i) for i, l in enumerate(lexicon_set))
index_lex = dict((i, l) for i, l in enumerate(lexicon_set))

# now have list of indices
both_indexed = repl_with_index(both, lex_index)

chunk_size = 20
step_size = 5

chunks = []
chunk_next = []
for i in range(0, len(both_indexed) - chunk_size, step_size):
    chunks.append(both_indexed[i:i+chunk_size])
    chunk_next.append(both_indexed[i+chunk_size])

X = np.zeros((len(chunks), chunk_size, len(lex_index)), dtype=np.bool)
y = np.zeros((len(chunks), len(lex_index)), dtype=np.bool)
for i, chunk in enumerate(chunks):
    for j, ind in enumerate(chunk):
        X[i, j, ind] = 1
    y[i, chunk_next[i]] = 1

print(X.shape)
print(y.shape)

#####################################33

model = Sequential()
model.add(LSTM(128, input_shape=(chunk_size, len(lex_index))))
model.add(Dense(len(lex_index)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#####################################3
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 40):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(both_indexed) - chunk_size - 1)
    
    sentence = both_indexed[start_index: start_index + chunk_size]
    generated = sentence

    for i in range(50):
        x = np.zeros((1, chunk_size, len(lex_index)))
        for t, ind in enumerate(sentence):
            x[0, t, ind] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds)
        next_char = index_lex[next_index]

        sentence = sentence[1:] + [next_index]
        generated += [next_index]


    print(indices_to_string(generated,index_lex))    
    with open("test.out.txt", "a") as wf:
        wf.write(str(i) + "\t" + indices_to_string(generated,index_lex) + "\n")
    print()
