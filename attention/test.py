from keras.engine import Layer
import numpy as np
import pandas as pd
import numpy
import pandas
from keras import initializers
from keras.layers import LSTM, RNN, GRU, SimpleRNN
from keras.layers import Dense, Dropout,Input
from keras import backend as K
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model


# Attention GRU network
class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')

        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

# input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(input)
# l_lstm = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
# l_att = AttLayer()(l_lstm)
# preds = Dense(2, activation='softmax')(l_att)
# model = Model(sequence_input, preds)
# model.compile(loss='categorical_crossentropy',
#              optimizer='rmsprop',
#              metrics=['acc'])
#
# print("model fitting - attention GRU network")
# model.summary()
# model.fit(x_train, y_train, validation_data=(x_val, y_val),
#          nb_epoch=10, batch_size=50)

class RNNModel(object):
    def __init__(self, look_back=1, epochs_purchase=20, epochs_redeem=40, batch_size=1, verbose=2, patience=10,
                 store_result=True):
        self.look_back = look_back
        self.epochs_purchase = epochs_purchase
        self.epochs_redeem = epochs_redeem
        self.batch_size = batch_size
        self.verbose = verbose
        self.store_result = store_result
        self.patience = patience
        self.purchase = pandas.read_csv('data/date_label.csv', usecols=[1], engine='python')
        self.redeem = pandas.read_csv('data/date_label.csv', usecols=[2], engine='python')

    def access_data(self, data_frame):
        # load the data set
        data_set = data_frame.values
        data_set = data_set.astype('float32')

        # LSTMs are sensitive to the scale of the input data, specifically when the sigmoid (default) or tanh activation functions are used. It can be a good practice to rescale the data to the range of 0-to-1, also called normalizing.
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_set = scaler.fit_transform(data_set)

        # reshape into X=t and Y=t+1
        train_x, train_y, test = self.create_data_set(data_set)

        # reshape input to be [samples, time steps, features]
        train_x = numpy.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
        return train_x, train_y, test, scaler

    # convert an array of values into a data set matrix
    def create_data_set(self, data_set):
        data_x, data_y = [], []
        for i in range(len(data_set) - self.look_back - 30):
            a = data_set[i:(i + self.look_back), 0]
            data_x.append(a)
            # data_y.append(list(data_set[i + self.look_back: i + self.look_back + 30, 0]))
            data_y.append(list(data_set[i + self.look_back: i + self.look_back+1, 0]))
        # print(numpy.array(data_y).shape)
        return numpy.array(data_x), numpy.array(data_y), data_set[-self.look_back:, 0].reshape(1, 1, self.look_back)
        # return numpy.array(data_x), numpy.array(data_y), data_set[-90:-30, 0].reshape(1, 1, self.look_back)

    def rnn_model(self, train_x, train_y, epochs):
        model = Sequential()

        model.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(1, self.look_back)))
        model.add(AttLayer)
        model.add(Dropout(0.5))
        # model.add(Dropout(0.5))
        # model.add(Flatten())
        model.add(Dense(1))
        # model.add(LSTM(64, input_shape=(1, self.look_back), return_sequences=True))
        # model.add(LSTM(32, return_sequences=False))
        # model.add(Dense(32))
        # model.add(Dense(30))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.summary()
        early_stopping = EarlyStopping('loss', patience=self.patience)
        history = model.fit(train_x, train_y, epochs=epochs, batch_size=self.batch_size, verbose=self.verbose,
                            callbacks=[early_stopping])
        return model

    def predict(self, model, data):
        prediction = model.predict(data)
        return prediction

    # def plot_show(self, predict):
    #     predict = predict[['purchase', 'redeem']]
    #     predict.plot()
    #     plt.show()

    def run(self):
        purchase_train_x, purchase_train_y, purchase_test, purchase_scaler = self.access_data(self.purchase)
        redeem_train_x, redeem_train_y, redeem_test, redeem_scaler = self.access_data(self.redeem)

        # print(purchase_train_y)

        purchase_model = self.rnn_model(purchase_train_x, purchase_train_y, self.epochs_purchase)
        redeem_model = self.rnn_model(redeem_train_x, redeem_train_y, self.epochs_redeem)

        purchase_predict = self.predict(purchase_model, purchase_test)

        datax, datay = [], []
        for i in range(len(purchase_train_x)):
            datax = []
            datax.append(purchase_train_x[i])
            test_p_pre = self.predict(purchase_model, numpy.array(datax))
            datay.append(test_p_pre[0])
        # print(len(datay))
        # print(numpy.array([datay]))
        # print(purchase_train_y)
        # purchase_scaler.inverse_transform(purchase_train_y).reshape(1, 1)
        # purchase_train_y.plot()
        # numpy.array([datay]).plot()
        trainPredictPlot = numpy.empty_like(purchase_train_y)
        trainPredictPlot[:, :] = numpy.nan
        trainPredictPlot[0:len(purchase_train_y), :] = purchase_train_y

        testPredictPlot = numpy.empty_like(purchase_train_y)
        testPredictPlot[:, :] = numpy.nan
        testPredictPlot[0:len(purchase_train_y), :] = numpy.array([datay])
        # print(testPredictPlot)
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        plt.show()

        redeem_predict = self.predict(redeem_model, redeem_test)

        test_user = pandas.DataFrame({'report_date': [20140900 + i for i in range(1, 31)]})

        purchase = purchase_scaler.inverse_transform(purchase_predict).reshape(1, 1)
        redeem = redeem_scaler.inverse_transform(redeem_predict).reshape(1, 1)
        print(purchase)

        for i in range(len(redeem)):
            redeem[i]=0
        # test_user['purchase'] = purchase
        # test_user['redeem'] = redeem
        # print(test_user)

        """Store submit file"""
        if self.store_result is True:
            test_user.to_csv('test/test_lstm3.csv', encoding='utf-8', index=None, header=None)

        """plot result picture"""
        # self.plot_show(test_user)


if __name__ == '__main__':
    initiation = RNNModel(look_back=60, epochs_purchase=330, epochs_redeem=1, batch_size=16, verbose=2, patience=50,
                          store_result=True)
    initiation.run()
