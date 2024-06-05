from sklearn.linear_model import LinearRegression as LinReg
import numpy as np


class LinearRegression:

    def __init__(self, train_data, train_data_samples=1000, test_data_samples=1000):
        super().__init__()
        self.regression = LinReg()
        
        self.test_start_index = train_data_samples
        self.test_end_index = train_data_samples + test_data_samples
        
        inputs, labels = self._prepare_data(train_data)
        self.inputs = inputs
        self.labels = labels

    def _prepare_data(self, train_data):
        inputs = np.empty((len(train_data), 480*848))
        labels = np.empty((len(train_data), 126))
        for sample, i in zip(train_data, range(self.test_end_index)):
            inputs[i] = sample['realsense'].numpy().reshape((480*848))
            labels[i] = sample['optitrack'].numpy()

            if i % 100 == 0:
                print(f"\rPreparing data... {i/self.test_end_index*100:.1f}%  ", end='')
        print("\rPreparing data...              ")
        return inputs, labels

    def fit(self):
        print("Fitting model... ", end='')
        self.regression.fit(self.inputs[:self.test_start_index], self.labels[:self.test_start_index])
        print("Done!")

    def predict(self):
        print("Predicting with model... ", end='')
        prediction = self.regression.predict(self.inputs[self.test_start_index:self.test_end_index])
        print("Done!")
        return prediction
    
    def score(self):
        return self.regression.score(self.inputs[self.test_start_index:self.test_end_index],
                                     self.labels[self.test_start_index:self.test_end_index])
    
    def get_left_graph_data(self):
        return self.labels[self.test_start_index:self.test_end_index] 
