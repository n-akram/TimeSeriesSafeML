import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

class DataPreparation:

    def __init__(self, test_frac=0.2, window_size=20):

        self.test_frac = test_frac
        self.window_size = window_size

    def __prepareDataForTraining(self, seq):

        x_data = []
        y_data = []
        L = len(seq)
        for i in range(L-self.window_size):        
            window = seq[i:i+self.window_size]
            label = seq[i+self.window_size:i+self.window_size+1]
            x_data.append(window)
            y_data.append(label)
        return x_data, y_data  

    def normalize_and_prepare_data(self, ts, scaler):
        
        test_set_size = int(np.round(self.test_frac*len(ts)))
        train_set = ts[:-test_set_size]    
        test_set = ts[-test_set_size:]

        #Normalize data
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        train_norm = scaler.fit_transform(train_set.reshape(-1, 1))
        test_norm = scaler.transform(test_set.reshape(-1, 1))       

        x_train, y_train = self.__prepareDataForTraining(train_norm)
        x_test, y_test = self.__prepareDataForTraining(test_norm)

        x_train = np.asarray(x_train).reshape(-1, self.window_size, 1)
        y_train = np.asarray(y_train).reshape(-1, 1)
        x_test = np.asarray(x_test).reshape(-1, self.window_size, 1)
        y_test = np.asarray(y_test).reshape(-1, 1)

        print('x_train.shape = ',x_train.shape)
        print('y_train.shape = ',y_train.shape)
        print('x_test.shape = ',x_test.shape)
        print('y_test.shape = ',y_test.shape)

        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        x_test = torch.from_numpy(x_test).type(torch.Tensor)
        y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
        y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)

        return scaler, x_train, x_test, y_train_lstm, y_test_lstm


   

