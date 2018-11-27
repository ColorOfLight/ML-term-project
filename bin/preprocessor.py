# Load Packages
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import preprocessing
from plots import draw_corr_heatmap
import seaborn as sns
import pickle
from logger import Logger

# Varaibles
train_rate = .85
model_name = 'xgboost-test2'

np.random.seed(0)

names = ['contract date', 'latitude', 'longtitude', 'altitude', '1st region id', '2nd region id', 'road id',
         'apartment id', 'floor', 'angle', 'area', 'parking lot limit', 'parking lot area', 'parking lot external',
         'management fee', 'households', 'age of residents', 'builder id', 'completion date', 'built year',
         'schools', 'bus stations', 'subway stations', 'price']

data = pd.read_csv('../data/data_train.csv',
                   names=names)

def get_X_y(data):
  data['angle'] = np.sin(data['angle'])

  data['contract date'] = pd.to_datetime(data['contract date'])
  data['completion date'] = pd.to_numeric(data['contract date'] - pd.to_datetime(data['completion date']))
  data['contract date'] = pd.to_numeric(data['contract date'] - data['contract date'].min())

  drop_columns = ['1st region id', '2nd region id',
                  'road id', 'apartment id', 'builder id', 'built year']
  data = data.drop(columns=drop_columns)

  data = data.dropna()

  def normalize(d):
    min_max_scaler = preprocessing.MinMaxScaler()
    d_scaled = min_max_scaler.fit_transform(d)
    return pd.DataFrame(d_scaled, columns=[item for item in names if item not in drop_columns])

  y = data['price']
  data = normalize(data)
  X = data.drop(columns=['price'])

  return X, y

def get_accuracy(y_pred, y_test):
  length = len(y_pred)
  _sum = 0
  for idx in range(length):
    _sum += abs((y_test[idx] - y_pred[idx]) / y_pred[idx])
  return 1 - (_sum / length)
  
# Main
# logger = Logger('xgboost')

X, y = get_X_y(data)
X_names = list(X)

# Create Train & test data
train_indexes = np.random.rand(len(X)) <= 0.85

X_train = X[train_indexes]
X_test = X[~train_indexes]
y_train = y[train_indexes]
y_test = y[~train_indexes]