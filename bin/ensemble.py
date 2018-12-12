import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from keras import Sequential
from keras.layers import Dense
from preprocessor import preprocess

class Ensemble(BaseEstimator, RegressorMixin):
  def __init__(self, estimators):
    self.estimators = estimators
    self.model = Sequential()
    self.model.add(Dense(64, activation="relu"))
    self.model.add(Dense(32, activation="relu"))
    self.model.add(Dense(1))
    self.model.compile(loss='mean_squared_error', optimizer='adam')

  def fit(self, X, y):
    pro_X = preprocess(X)

    for i, est in enumerate(self.estimators):
      if i >= len(self.estimators) - 1:
        est.fit(X, y)
      else:
        est.fit(pro_X, y)

    new_features = self._get_new_features(X)

    self.model.fit(new_features, y.values, epochs=10)

  def predict(self, X):
    new_features = self._get_new_features(X)
    return self.model.predict(new_features)

  def _get_new_features(self, X):
    pro_X = preprocess(X)

    result = []
    for i, est in enumerate(self.estimators):
      if i >= len(self.estimators) - 1:
        result.append(est.predict(X).reshape(X.shape[0], 1))
      else:
        result.append(est.predict(pro_X).reshape(X.shape[0], 1))
    return np.concatenate(result, axis=1)
