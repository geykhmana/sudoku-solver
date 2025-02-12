import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.utils import Sequence # type: ignore
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Reshape, Activation # type: ignore
import matplotlib.pyplot as plt

path = "/content/"
data = pd.read_csv(path+"sudoku.csv")

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

model = Sequential()

model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9, 9, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(1,1), activation='relu', padding='same'))

model.add(Flatten())
model.add(Dense(81*9))
model.add(Reshape((-1, 9)))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer = Adam(learning_rate = 0.001), metrics = ['accurary'])
model.summary()

train_idx = int(len(data) * 0.95)
data = data.sample(frac=1).reset_index(drop=True)
training_generator = DataGenerator(data.iloc[:train_idx], subset = "train", batch_size = 640)
validation_generator = DataGenerator(data.iloc[train_idx:], subset = "train", batch_size = 640)

from tf.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau # type: ignore
filepath1 = "weights-improvement-{epoch:02d}-{val_accurary:.2f}.hdf5"
filepath2 = "best_weights.hdf5"
checkpoint1 = ModelCheckpoint(filepath1, monitor = 'val_accuracy', verbose = 1, save_best_only = True, mode = 'max')
checkpoint2 = ModelCheckpoint(filepath2, monitor = 'val_accuracy', verbose = 1, save_best_only = True, mode = 'max')

reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    patience = 3,
    verbose = 1,
    min_lr = 1e-6
)
callbacks_list = [checkpoint1, checkpoint2, reduce_lr]

history = model.fit_generator(training_generator, validation_data = validation_generator, epochs = 5, verbose = 1, callbacks = callbacks_list)

model.load_weights(path+"best_weights.hdf5")

def solve_sudoku_with_nn(model, puzzle):
    #Preprocess the inputted sudoku puzzle
    puzzle = puzzle.replace('\n', '').replace(' ', '')
    initial_board = np.array([int(j) for j in puzzle]).reshape((9, 9, 1)) #Convert string to 3D Numpy array
    initial_board = initial_board / 9 - 0.5

    while True:
        #Use NN to predict values for empty cells
        predictions = model.predict(initial_board.reshape((1, 9, 9, 1))).squeeze() #Predict alues for empty cells
        pred = np.argmax(predictions, axis=1).reshape((9, 9)) + 1
        prob = np.around(np.max(predictions, axis=1).reshape((9, 9)), 2)

        initial_board = ((initial_board + 0.5) * 9).reshape((9, 9))
        mask = (initial_board == 0)

        if mask.sum() == 0:
            #Puzzle is solved
            break

        prob_new = prob * mask

        ind = np.argmax(prob_new)
        x, y = (ind // 9), (ind % 9)

        val = pred[x][y]
        initial_board[x][y] = val
        initial_board = (initial_board / 9) - 0.5
    
    #Convert solved puzzle to string
    solved_puzzle = ''.join(map(str, initial_board.flatten().astype(int)))

    return solved_puzzle