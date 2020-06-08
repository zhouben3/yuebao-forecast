# Stacked LSTM for international airline passengers problem with memory
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Bidirectional
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0];
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

def create_result_dataset(dataset):
	resultX=dataset[-60:-30,0];
	temp = []
	temp.append(resultX)
	resultX = numpy.array(temp)
	resultX = numpy.reshape(resultX, (resultX.shape[0], resultX.shape[1], 1))
	return resultX


# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('data/date_label.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
print(dataset)
my_test=dataset[-30:,0]
print(my_test)
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
# print(train_size)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


# print(resultX)
# reshape into X=t and Y=t+1
look_back = 30
trainX, trainY = create_dataset(train, look_back)
# print(len(trainX))
# print(len(trainY))
testX, testY = create_dataset(test, look_back)

# temp=[]
# temp.append(resultX)
# resultX=numpy.array(temp)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# resultX = numpy.reshape(resultX, (resultX.shape[0], resultX.shape[1], 1))
# print(resultX)
# print(testX)

# resultX=numpy.reshape(resultX, (resultX.shape[0], resultX.shape[1], 1))

# print(len(testX))
#
# print(testY)
# create and fit the LSTM network
batch_size = 1
model = Sequential()
model.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(30,1)))
model.add(Dense(1))
# model.add(LSTM(64, input_shape=(1, self.look_back), return_sequences=True))
# model.add(LSTM(32, return_sequences=False))
# model.add(Dense(32))
# model.add(Dense(30))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(trainX, trainY, epochs=1000, verbose=0)
# for i in range(100):
# 	model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
# 	model.reset_states()
# make predictions
trainPredict = model.predict(trainX, batch_size=batch_size)
model.reset_states()
testPredict = model.predict(testX, batch_size=batch_size)
model.reset_states()

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
temp=dataset[-30:]
tempx=[]
def creat_temp(temp):
	datay = []
	for i in temp:
		datay.append([i])
	return numpy.array(datay)
for i in range(0,30):
	 if i==0:
		 resultX=create_result_dataset(test)
		 # print(resultX)
		 result = model.predict(resultX, batch_size=batch_size)
		 result = scaler.inverse_transform(result)
		 tempx.append(result)
		 # print(temp)
		 print(result)
		 temp=numpy.insert(temp, -1, result[0])
		 temp=creat_temp(temp)
		 # print(temp)
		 # create_result_dataset(temp)
	 # else:
		#  # print(tempx[0][0][0])
		#  # temp.append(tempx[0][0][0])
		#  # print(temp)
		#  # print(type(temp))
		#  # print(temp)
		#  resultX = create_result_dataset(temp)
		#  result = model.predict(resultX, batch_size=batch_size)
		#  result = scaler.inverse_transform(result)
		#  tempx.append(result)
		#  temp=numpy.insert(temp, -1, result)
		#  temp = creat_temp(temp)
# 		 print(len(temp))
# print(numpy.array(tempx))

# resultX=create_result_dataset(test)
# result=model.predict(resultX, batch_size=batch_size)
# # print(result)
# result=scaler.inverse_transform(result)
# print(result)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# print(len(testPredict))
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()