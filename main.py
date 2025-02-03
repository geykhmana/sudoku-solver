import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.utils import Sequence # type: ignore
from tensorflow.keras.layers import * # type: ignore

data = pd.read_csv("/content/sudoku.csv")

try:
    data = pd.DataFrame({"quizzes": data["puzzle"], "solutions": data["solution"]})
except:
    pass

class DataGenerator(Sequence):
    def __init__(self, df, batch_size = 16, subset = "train", shuffle = False, info={}):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subset = subset
        self.info = info

        self.on_epoch_end()

    # Returns the # of batches in dataset
    def __len__(self):
        return int(np.floor(len(self.df)/self.batch_size))
    
    # Shuffles indexes at end of each epoch if shuffle == True
    def __on_epoch_end__(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    #Generates batches of data, prepares target solutions
    def __getitem__(self, index):
        X = np.empty((self.batch_size, 9, 9, 1))
        y = np.empty((self.batch_size, 81, 1))
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        for i, f in enumerate(self.df['quizzes'].iloc[indexes]):
            self.info[index * self.batch_size + 1] = f
            X[i,] = (np.array(list(map(int, list(f)))).reshape((9, 9, 1))/9) - 0.5
        
        if self.subset == 'train':
            for i, f in enumerate(self.df['solutions'].iloc[indexes]):
                self.info[index * self.batch_size + i] = f
                y[i,] = np.array(list(map(int, list(f)))).reshape((81, 1)) - 1
        
        if self.subset == 'train':
            return X, y
        else:
            return X