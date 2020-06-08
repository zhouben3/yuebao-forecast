# multivariate multi-step encoder-decoder lstm example
from numpy import array
import copy
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Bidirectional

n_look_back=80
# convert an array of values into a data set matrix
def create_data_set(look_back, data_set):
	data_x, data_y = [], []
	for i in range(len(data_set) - look_back - 30):
		a = data_set[i:(i + look_back), 0]
		data_x.append(a)
		data_y.append([list(data_set[i + look_back: i +look_back + 30, 0])])
	# print(numpy.array(data_y).shape)
	return numpy.array(data_x), numpy.array(data_y), data_set[-look_back:, 0].reshape(1, 1, look_back)

def access_data( data_frame):
	# load the data set
	data_set = data_frame.values
	data_set = data_set.astype('float32')

	# LSTMs are sensitive to the scale of the input data, specifically when the sigmoid (default) or tanh activation functions are used. It can be a good practice to rescale the data to the range of 0-to-1, also called normalizing.
	scaler = MinMaxScaler(feature_range=(0, 1))
	data_set = scaler.fit_transform(data_set)

	# reshape into X=t and Y=t+1
	train_x, train_y, test =create_data_set(n_look_back,data_set)

	# reshape input to be [samples, time steps, features]
	train_x = numpy.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
	return train_x, train_y, test, scaler


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


n_steps_in, n_steps_out = 1, 1

# covert into input/output
purchase = pandas.read_csv('data/date_label.csv', usecols=[1], engine='python')
# X, y = split_sequences(dataset, n_steps_in, n_steps_out)
X,y,purchase_test, purchase_scaler =access_data(purchase)
# the dataset knows the number of features, e.g. 2
n_features = 30

# define model
model = Sequential()
model.add(Bidirectional(LSTM(200, activation='relu', input_shape=(n_steps_in, n_look_back))))
model.add(RepeatVector(n_steps_out))
model.add(Bidirectional(LSTM(200, activation='relu', return_sequences=True)))
model.add(TimeDistributed(Dense(30)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=300, verbose=0)
# demonstrate prediction
x_input=purchase_test
x_input = x_input.reshape((1, n_steps_in, n_look_back))
yhat = model.predict(x_input, verbose=0)
purchase=purchase_scaler.inverse_transform(yhat[0]).reshape(30, 1)
redeem=copy.deepcopy(purchase)
for i in range(len(redeem)):
	redeem[i] = 0
print(purchase)
test_user = pandas.DataFrame({'report_date': [20140900 + i for i in range(1, 31)]})
test_user['purchase'] = purchase
test_user['redeem'] = redeem
test_user.to_csv('test/test_lstm1.csv', encoding='utf-8', index=None, header=None)
