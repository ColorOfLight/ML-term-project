import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge

class Ensemble(BaseEstimator, RegressorMixin):
  def __init__(self, estimators):
    self.estimators = estimators
    self.combine = Ridge(alpha=10, max_iter=50000)

  def fit(self, X, y):
    for est in self.estimators:
      est.fit(X, y)

    new_features = self._get_new_features(X)

    self.combine.fit(new_features, y)

  def predict(self, X):
    new_features = self._get_new_features(X)
    return self.combine.predict(new_features)

  def _get_new_features(self, X):
    result = []
    for est in self.estimators:
      result.append(est.predict(X).reshape(X.shape[0], 1))
    return np.concatenate(result, axis=1)