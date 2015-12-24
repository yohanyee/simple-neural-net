from util import *
import numpy as np

DATA_DIGITS = "/add/your/path/to/digits/data/here"

class Data(object):
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None
        self.input_shape = None
        self.output_shape = None
        self.data_num = 0
        self.valid_num = 0
        self.train_num = 0
        self.test_num = 0
    
    def load_digits_data(self):
        self.X_train, self.X_valid, self.X_test, \
            self.y_train, self.y_valid, self.y_test = LoadDigits(DATA_DIGITS)
        self.train_num = self.X_train.shape[0]
        self.valid_num = self.X_valid.shape[0]
        self.test_num = self.X_test.shape[0]
        
    def coerce_data(self):
        self.X = np.vstack([self.X_train, self.X_valid, self.X_test])
        self.y = np.vstack([self.y_train, self.y_valid, self.y_test])
        self.set_sizes()
    
    def split_data(self, proportions):
        data_size = self.X.shape[0]
        num_train = int(proportions[0]*data_size)
        num_test = int(proportions[1]*data_size)
        num_valid = data_size - num_train_num_test
        indices = range(data_size)
        np.random_shuffle(indices)
        indices_train = indices[0:num_train]
        indices_valid = indices[num_train:num_train+num_valid]
        indices_test = indices[num_train+num_valid:data_size]
        self.X_train = self.X[indices_train]
        self.X_valid = self.X[indices_valid]
        self.X_test = self.X[indices_test]
        self.y_train = self.y[indices_train]
        self.y_valid = self.y[indices_valid]
        self.y_test = self.y[indices_test]
        self.set_sizes()
        
    def set_sizes(self):
        self.data_num = self.X.shape[0]
        self.train_num = self.X_train.shape[0]
        self.valid_num = self.X_valid.shape[0]
        self.test_num = self.X_test.shape[0]
        
    def reshape(self, input_shape, output_shape):
        self.X_train = self.X_train.reshape([self.train_num] + input_shape)
        self.y_train = self.y_train.reshape([self.train_num] + output_shape)
        self.X_valid = self.X_valid.reshape([self.valid_num] + input_shape)
        self.y_valid = self.y_valid.reshape([self.valid_num] + output_shape)
        self.X_test = self.X_test.reshape([self.test_num] + input_shape)
        self.y_test = self.y_test.reshape([self.test_num] + output_shape)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.coerce_data()
        
    def pair_reduce_data(self):
        true_train_indices = np.where(self.y_train==1)[0]
        true_valid_indices = np.where(self.y_valid==1)[0]
        true_test_indices = np.where(self.y_test==1)[0]
        false_train_indices = np.random.choice(np.where(self.y_train==0)[0], true_train_indices.size, replace=False)
        false_valid_indices = np.random.choice(np.where(self.y_valid==0)[0], true_valid_indices.size, replace=False)
        false_test_indices = np.random.choice(np.where(self.y_test==0)[0], true_test_indices.size, replace=False)
        self.y_train = np.vstack([self.y_train[false_train_indices], self.y_train[true_train_indices]])
        self.y_valid = np.vstack([self.y_valid[false_valid_indices], self.y_valid[true_valid_indices]])
        self.y_test = np.vstack([self.y_test[false_test_indices], self.y_test[true_test_indices]])
        self.X_train = np.vstack([self.X_train[false_train_indices], self.X_train[true_train_indices]])
        self.X_valid = np.vstack([self.X_valid[false_valid_indices], self.X_valid[true_valid_indices]])
        self.X_test = np.vstack([self.X_test[false_test_indices], self.X_test[true_test_indices]])
        self.set_sizes()
