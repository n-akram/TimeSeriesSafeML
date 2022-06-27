import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

class DataPreparation:

    """
    The function initializes the test size and window size for preparing time seires data.
    
    :param test_frac: input value for test set size. Default value is 0.2.
    :param window_size: input value for considering length of previous time steps. Default value is 20.
    """
    
    def __init__(self, test_frac=0.2, window_size=20):

        self.test_frac = test_frac
        self.window_size = window_size

    def _formulateTimeSeies(self, seq):

        x_data = []
        y_data = []
        L = len(seq)
        for i in range(L-self.window_size):        
            window = seq[i:i+self.window_size]
            label = seq[i+self.window_size:i+self.window_size+1]
            x_data.append(window)
            y_data.append(label)
        return x_data, y_data  

    def splitData(self, ts):

        """This function splits the given time series into train and test set of defined length.

        :return array: training and test set
        """
        test_set_size = int(np.round(self.test_frac*len(ts)))
        train_set = ts[:-test_set_size]    
        test_set = ts[-test_set_size:]

        return train_set, test_set

    def prepare_data(self, x_train, x_test, y_train, y_test):

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
        y_train = torch.from_numpy(y_train).type(torch.Tensor)
        y_test = torch.from_numpy(y_test).type(torch.Tensor)

        return x_train, x_test, y_train, y_test

    def prepare_data_with_normalization(self, ts, scaler):
        
        """This function prepares time series data for a supervied learning problem by applying normalization to the data.

        :param array ts: entire time series
        :param scaler: scaler object to normalize/standardize the time series

        :return : scaler object and the time series split into train and test as (x, y).
        """
        train_set, test_set = self.splitData(ts)

        #Normalize data        
        train_norm = scaler.fit_transform(train_set.reshape(-1, 1))
        test_norm = scaler.transform(test_set.reshape(-1, 1))  

        x_train, y_train = self._formulateTimeSeies(train_norm)
        x_test, y_test = self._formulateTimeSeies(test_norm)

        x_train, x_test, y_train, y_test = self.prepare_data(x_train, x_test, y_train, y_test)

        return scaler, x_train, x_test, y_train, y_test

    def prepare_data_without_normalization(self, ts):
        
        """This function prepares time series data for a supervied learning problem without normalizing the data.

        :return array: the time series split into train and test as (x, y).
        """
        train_set, test_set = self.splitData(ts)            

        x_train, y_train = self._formulateTimeSeies(train_set)
        x_test, y_test = self._formulateTimeSeies(test_set)

        x_train, x_test, y_train, y_test = self.prepare_data(x_train, x_test, y_train, y_test)

        return x_train, x_test, y_train, y_test

        

        



   

