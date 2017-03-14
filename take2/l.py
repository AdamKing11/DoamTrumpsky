#from NPhoner import NPhoner
#from DoamPrePros import DoamPrePros
import numpy, re, pickle

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Embedding, Bidirectional
from keras.optimizers import RMSprop

"""
Generative LSTM model to build Haiku's, trained off tweets of DJ Trump
LSTM model is HEAVILY influenced by fchollet's Nietzche script
"""


class DHPrePros:
# class for various preprocessing functions for the data

	def __init__(self, raw_file = "BigT.txt", load = False, fc = False, w_only = False):
		# for setting if we're
		#	1) folding case = fc
		# 	2) only using "word-characters" (including "#@ " as this is twitter...)
		self.fc = fc
		self.w_only = w_only

		raw = self.load_text(raw_file)
		self.phones = self._build_phones(raw)
		self.char_index, self.index_char = self._build_index_dict(self.phones)
	
	def load_text(self, raw_file):
		# load a text file and get the raw text
		raw = ''
		with open(raw_file, 'r') as rf:
			raw = rf.read()
			raw = re.sub('\n', ' ', raw)
			# if we want to clean up the raw text, do it now
			if self.fc:
				raw = raw.lower()
			if self.w_only:
				# ^ for not....
				raw = re.sub("[^a-z@#' ]", "", raw)
		# return raw text
		return raw

	def _build_phones(self, raw_text):
		# get a list of all the characters that were in the training text file
		phones = set([c for c in raw_text])
		return phones

	def _build_index_dict(self, phones):
		# we build dictionaries to hold an index for each char and vice versa
		# e.g. 1->a, 2->b, 3->c, 4->d, etc.
		# these have the benefit that we can store them in Numpy arrays AND
		# when making one-hots, they make it much easier to just set a single element to 1

		# char to index
		ci = dict((p, i) for i, p in enumerate(phones))
		# index to char
		ic = dict((i, p) for i, p in enumerate(phones))


		return ci, ic

	def build_Xy(self, raw_file, chunk = 20, jump = 3):

		char_X = []
		char_y = []
		
		raw = self.load_text(raw_file)
		# build the lists of X's (inputs) and y's (responses)
		# because this is a language model, X's are just the 'chunk' number
		# of nphones before the single responce, y
		i = 0
		# we first do this with their actual character values before
		# converting to numpy one-hot vectors
		for i in range(0,len(raw)-chunk,jump):
			## build X
			char_X.append(raw[i:i+chunk])
			## build y
			char_y.append(raw[i+chunk])
			

		# time to make 1-hot vectors
		X = numpy.zeros((len(char_X), chunk, len(self.phones)), dtype = 'int8')
		y = numpy.zeros((len(char_y), len(self.phones)), dtype = 'int8')

		# because these are 1-hot, the vectors will be all 0's except one 1
		# we use the dictionary we made earlier where we mapped an nphone to a unique
		# integer to look up which index in the big matrix we set to 1
		for i in range(X.shape[0]):
			for j in range(chunk):
				# based on the value of the given character in our little dictionary,
				# we set a single 0 to one in the Numpy array. Because we use the same
				# dictionary for everything, we're guaranteed to have a good "translation"
				# between one-hots and actual charaters
				ind = self.char_index[char_X[i][j]]
				X[i,j,ind] = 1
			# put the 1 in the correct spot for the y vector
			ind = self.char_index[char_y[i]]
			y[i,ind] = 1

		return X, y

	def get_phones(self):
		# returns the SET of phones/characters the model knows about
		return self.phones

	def get_ind_char(self, i):
		# given an index, returns a character/phone
		return self.index_char[i]

	def vec_ind_char(self, l):
		# given a VECTOR of indexes, returns a LIST of characters
		chars = [self.index_char[i] for i in l]
		return chars

	def vec_char_ind(self, l):
		# given a LIST of characters, returns a LIST of integer indexes
		indexs = [self.char_index[c] for c in l]
		return indexs

	def vec_ind_onehot(self, l, vec_size):
		# given a LIST of indexes, returns a Numpy ARRAY of one-hots
		little_X = numpy.zeros((1, len(l), vec_size), dtype='int8')

		for i, j in enumerate(l):
			little_X[0,i, j] = 1
		return little_X

	def vec_ind_onehot_padded(self, l, pad_len, vec_size):
		# like above, but pads the beginning with 'pad_len' 0's
		# this way, we can get around the semi-strict Keras req on input length
		little_X = numpy.zeros((1, len(l) + pad_len, vec_size), dtype='int8')

		for i, j in enumerate(l):
			little_X[0,i+pad_len,j] = 1
		return little_X		


	def get_random_string(self, raw_file, chunk):
		# loads the text and spits back a random string from the text of a given
		# length
		raw = self.load_text(raw_file)
		
		i = numpy.random.randint(0, len(raw)-chunk)
		return raw[i:i+chunk]


###############################################

class haikuModel:

	def __init__(self, X_shape, y_shape, hidden_size = 128, load = False, file_name="haikuDon.model"):
		if not load:
			self.model = self._build_lstm(X_shape, y_shape, hidden_size)
		else:
			self.load(file_name)

	def _build_lstm(self, Xs, ys, h):
		model = Sequential()

		model.add(LSTM(h,input_shape=(Xs[1],Xs[2]),dropout_W=0.2, dropout_U=0.2))
		model.add(Dense(ys[1], activation="softmax"))

		model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=0.01))

		return model

	def train(self, X, y, batch = 128, epochs = 1):
		self.model.fit(X, y, batch_size = batch, nb_epoch = epochs, verbose = 1)

	def save(self, file_name = "haikuDon.model"):
		self.model.save(file_name)

	def load(self, file_name = "haikuDon.model"):
		self.model = load_model(file_name)

	def get_model(self):
		return self.model

##################################3

def softmax(x):
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / numpy.sum(e_x)

def make_prediction(model, little_X):
    predictions = model.predict(little_X, verbose=0)[0]
    predictions = numpy.log(predictions)
    predictions = softmax(predictions)
    try:
    	next_index = numpy.argmax(numpy.random.multinomial(1,predictions,1))
    except:
   		predictions = softmax(predictions)
   		next_index = numpy.argmax(numpy.random.multinomial(1,predictions,1))
    return next_index



chunk = 30

d = DHPrePros('BigT.txt', load=False, fc = True, w_only = True)

phones = d.get_phones()
(X, y) = d.build_Xy("BigT.txt", chunk = chunk)


#for p in sorted(phones):
#	print(p)
#

print()
print(len(phones))


print(X.shape)
print(y.shape)

gen = d.get_random_string("BigT.txt", chunk)
gen_ind = d.vec_char_ind(gen)
gen_onehot = d.vec_ind_onehot(gen_ind, len(phones))

print("Building model")
#haiku = haikuModel(X.shape, y.shape, load = True, file_name = "haikuDon.model")
haiku = haikuModel(X.shape, y.shape, load = False)
#haiku.load("haikuDon.model")
print("training model")



for epoch in range(30):
	print()
	print("*"*80)
	print("Epoch", epoch)
	#haiku.train(X,y)
	haiku.save("haikuDon.model")

	print("Starting with:")
	print(gen)
	print("-"*80)
	print(gen, end = '')

	for i in range(60):
		next_index = make_prediction(haiku.get_model(), gen_onehot)
		#next_index = numpy.random.randint(0,100)
		print(d.get_ind_char(next_index), end = '')
		
		gen_ind = gen_ind[1:]
		gen_ind.append(next_index)
		gen_onehot = d.vec_ind_onehot(gen_ind, len(phones))
	haiku.train(X,y)