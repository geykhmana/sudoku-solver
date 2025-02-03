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