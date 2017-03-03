from NPhoner import NPhoner
from DoamPrePros import DoamPrePros
import numpy, re, pickle

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Embedding, Bidirectional
from keras.optimizers import RMSprop

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def make_prediction(model, little_X):
    predictions = model.predict(little_X, verbose=0)[0]
    predictions = softmax(predictions)
    next_index = np.argmax(numpy.random.multinomial(1,predictions,1))
    return next_index


d = NPhoner('np.txt', load=True)
phones = d.get_nphones()

dpp = DoamPrePros(phones)

(X, y) = dpp.X_y()

print(X.shape)
print(y.shape)

gen = dpp.get_random_string()
gen_ind = dpp.list_to_indeces(gen)
print(''.join(gen))
print(gen_ind)
print(''.join(dpp.list_to_chars(gen_ind)))

doam = Sequential()

doam.add(LSTM(128,input_shape=(X.shape[1],X.shape[2])))
doam.add(Dense(y.shape[1], activation="softmax"))

doam.compile(loss="categorical_crossentropy", optimizer="adam")

for epoch in range(10):
	
	#doam.fit(X, y, batch_size=1024, nb_epoch=1)
	gen = dpp.get_random_string()
	print("Epoch #" + str(epoch))
	print("Starting with:")
	print(''.join(gen))
	print()
	little_X = numpy.array(dpp.list_to_indeces(gen))
	print(''.join(gen), end = '')
	for i in range(60):
		#next_index = make_prediction(doam, little_X)
		next_index = numpy.random.randint(0,100)
		print(dpp.get_index_char(next_index), end = '')
		little_X = list(little_X[1:])
		little_X.append(next_index)
		little_X = numpy.array(little_X)
		pass
	print("\n\n")
	


