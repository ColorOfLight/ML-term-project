# Load Packages
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report
from plots import draw_corr_heatmap
import seaborn as sns
import xgboost as xgb
import pickle
from logger import Logger
import os
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostRegressor
from ensemble import Ensemble
from sklearn.impute import SimpleImputer
from ilbeom_lg_v2 import Ilbeom_Linear
from sklearn.model_selection import StratifiedKFold

os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

# Varaibles
train_rate = .8
# The model will saved in ../models/{model_name}.dat
model_name = 'ensemble-test1'

np.random.seed(0)

names = ['contract date', 'latitude', 'longtitude', 'altitude', '1st region id', '2nd region id', 'road id',
         'apartment_id', 'floor', 'angle', 'area', 'parking lot limit', 'parking lot area', 'parking lot external',
         'management fee', 'households', 'age of residents', 'builder id', 'completion date', 'built year',
         'schools', 'bus stations', 'subway stations', 'price']
non_numeric_names = ['contract date', 'completion date']

tuned_parameters = {
    'n_estimators': [100, 200, 400],
    'learning_rate': [0.02, 0.04, 0.08, 0.1, 0.4],
    'gamma': [0, 1, 2],
    'subsample': [0.5, 0.66, 0.75],
    'colsample_bytree': [0.6, 0.8, 1],
    'max_depth': [6, 7, 8]
    # 'learning_rate': [0.02],
    # 'gamma': [0],
    # 'subsample': [0.5],
    # 'colsample_bytree': [0.6],
    # 'max_depth': [6]
}

def acc_scorer(model, X, y):
    y_pred = model.predict(X)
    return get_accuracy(y_pred, y.iloc)

def preprocess(data):
  data['angle'] = np.sin(data['angle'])

  data['contract date'] = pd.to_datetime(data['contract date'])
  data['completion date'] = pd.to_numeric(
      data['contract date'] - pd.to_datetime(data['completion date']))
  data['contract date'] = pd.to_numeric(
      data['contract date'] - data['contract date'].min())

  drop_columns = ['1st region id', '2nd region id',
                  'road id', 'apartment_id', 'builder id', 'built year']
  data = data.drop(columns=drop_columns)
  drop_columns.append('price')

  def normalize(d):
    min_max_scaler = preprocessing.MinMaxScaler()
    d_scaled = min_max_scaler.fit_transform(d)
    return pd.DataFrame(d_scaled, columns=[item for item in names if item not in drop_columns])
  
  return normalize(data)

def get_accuracy(y_pred, y_test):
  length = len(y_pred)
  _sum = 0
  for idx in range(length):
    _sum += abs((y_test[idx] - y_pred[idx]) / y_pred[idx])
  return 1 - (_sum / length)

# Main
logger = Logger('final')

data = pd.read_csv('../data/data_train.csv',
                   names=names)

# Fill NaN
def fill_missing_values(data, is_test=False):
  new_data = data.drop(columns=non_numeric_names)
  imputer = SimpleImputer(missing_values=np.nan, strategy='median')
  imputer = imputer.fit(new_data)
  new_data = imputer.transform(new_data)
  if is_test:
    columns = [n for n in names if n not in non_numeric_names]
    columns.remove('price')
    new_data = pd.DataFrame(
        new_data, columns=columns)
  else:
    new_data = pd.DataFrame(new_data, columns=[n for n in names if n not in non_numeric_names])
  for n in non_numeric_names:
    new_data[n] = data[n]
  return new_data

data = fill_missing_values(data)

y = data['price']
X = data.drop(columns=['price'])
# X_names = list(X)

def get_unique_model():
  xg = xgb.XGBRegressor(n_estimators=200, learning_rate=0.02, gamma=0, subsample=0.75,
                            colsample_bytree=1, max_depth=6)
  en = ElasticNet(l1_ratio=0.95, alpha=0.15, max_iter=50000)
  ada = AdaBoostRegressor(
      learning_rate=0.01, loss='square', n_estimators=100)
  lr = Ilbeom_Linear()

  lst = [xg, en, ada, lr]

  return Ensemble(lst)


# model_n = xgb.XGBRegressor(n_estimators=200, learning_rate=0.02, gamma=0, subsample=0.75,
#                            colsample_bytree=1, max_depth=6)
model_n = ElasticNet(l1_ratio=0.95, alpha=0.15, max_iter=50000)
model_u = get_unique_model()

def test_cv(model, X, y, n_splits=5):
  # print(np.mean(cross_val_score(model, X, y, scoring=acc_scorer, cv=5, n_jobs=-1)))
  skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

  results = []
  for i, (train, test) in enumerate(skf.split(X, y)):
    print("Running Fold", i+1, "/", 5)
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    results.append(get_accuracy(y_pred, y_test.iloc))
  
  print(f"result: {sum(results) / n_splits}")

# Test each model

# test_cv(model_n, preprocess(X), y)
# test_cv(model_u, X, y)

# Write Answer Sheet
def write_answers(model_n, model_u):
  data = pd.read_csv('../data/data_test.csv',
                     names=[n for n in names if n is not 'price'])
  data = fill_missing_values(data, is_test=True)
  np.savetxt('../data/result_.csv', model_n.predict(preprocess(data)).reshape(-1,1))
  np.savetxt('../data/result_unique.csv', model_u.predict(data).reshape(-1, 1))


# write answers
model_n.fit(preprocess(X), y)
model_u.fit(X, y)

write_answers(model_n, model_u)
