import NPhoner
import numpy, re, pickle

class DoamPrePros:

	def __init__(self, nphones, raw_file = 'cleaned_all.txt', chunk = 50, jump = 6):
		self.phones = nphones
		self.raw_file = raw_file
		self.chunk = chunk
		
		self.char_index, self.index_char = self._build_index_dict(nphones)

		self.X, self.y = self._build_Xy(self.raw_file, chunk, jump)

	def _build_index_dict(self, nphones):
		# char to index
		ci = dict((p, i) for i, p in enumerate(nphones))
		# index to char
		ic = dict((i, p) for i, p in enumerate(nphones))

		return ci, ic

	def load_text(self, raw_file):
		raw = ''
		with open(raw_file, 'r') as rf:
			raw = rf.read()
			raw = re.sub('\n', ' ', raw)
		return raw


	def _build_Xy(self, raw_file, chunk, jump):

		raw = self.load_text(raw_file)

		max_nphone = max([len(n) for n in self.phones])
		
		char_X = []
		char_y = []
		
		# build the lists of X's (inputs) and y's (responses)
		# because this is a language model, X's are just the 'chunk' number
		# of nphones before the single responce, y
		i = 0
		# we first do this with their actual character values before
		# converting to numpy one-hot vectors		
		while i < len(raw)-chunk:
			## build X
			char_X.append([])

			for j in range(chunk):

				p = ''
				for k in range(1,max_nphone+1):
					temp_p = ''.join(raw[i+j:i+j+k]) 

					if temp_p in self.phones:
						p = temp_p
				char_X[-1].extend(p)

			## build y
			p = ''
			for j in range(1,max_nphone+1):
				temp_p = ''.join(raw[i+len(char_X[-1]):i+j+len(char_X[-1])])
				if temp_p in self.phones:
					p = temp_p
			char_y.append([p])
			# next iteration
			i += jump

		# time to make 1-hot vectors
		X = numpy.zeros((len(char_X), chunk, len(self.phones)), dtype = 'int8')
		y = numpy.zeros((len(char_y), len(self.phones)), dtype = 'int8')

		# because these are 1-hot, the vectors will be all 0's except one 1
		# we use the dictionary we made earlier where we mapped an nphone to a unique
		# integer to look up which index in the big matrix we set to 1
		for i in range(X.shape[0]):
			try:
				for j in range(chunk):
					c = char_X[i][j]
					ind = self.char_index[c]
					X[i,j,ind] = 1
			# put the 1 in the correct spot for the y vector
				ind = self.char_index[char_y[i][0]]
				y[i,ind] = 1
			except:
				pass

		return X, y

	def test_train(self, ratio = .8):
		split_index = int(self.X.shape[0])
		
		train_X = self.X[:split_index]
		train_y = self.y[:split_index]
		test_X = self.X[split_index:]
		test_y = self.y[split_index:]

		return (train_X, train_y), (test_X, test_y)

	def X_y(self):
		return self.X, self.y

	def get_char_index(self, c):
		return self.char_index[c]

	def get_index_char(self, i):
		return self.index_char[i]

	def get_random_string(self):
		raw = self.load_text(self.raw_file)
		max_nphone = max([len(n) for n in self.phones])

		i = numpy.random.randint(0, len(raw)-(self.chunk*max_nphone))

		rand_str = []
		while len(rand_str) < self.chunk:
			p = ''
			for j in range(1,max_nphone+1):
				temp_p = ''.join(raw[i:i+j])
				if temp_p in self.phones:
					p = temp_p
			rand_str.append(p)
			i+= len(p)

		return rand_str

	def list_to_onehot(self, l):
		little_X = numpy.zeros((1, self.X.shape[1], self.X.shape[2]), dtype='int8')

		for i in range(self.X.shape[1]):
			little_X[0,i, l[i]] = 1
		return little_X

	def list_to_indeces(self, l):
		return [self.char_index[i] for i in l]

	def list_to_chars(self, l):
		return [self.index_char[c] for c in l]

	def nb_phones(self):
		return len(self.phones)