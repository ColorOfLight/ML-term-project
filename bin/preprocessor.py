# Load Packages
import pandas as pd
import numpy as np
from sklearn import preprocessing

def preprocess(data):
  names = ['contract date', 'latitude', 'longtitude', 'altitude', '1st region id', '2nd region id', 'road id',
         'apartment_id', 'floor', 'angle', 'area', 'parking lot limit', 'parking lot area', 'parking lot external',
         'management fee', 'households', 'age of residents', 'builder id', 'completion date', 'built year',
         'schools', 'bus stations', 'subway stations', 'price']
  non_numeric_names = ['contract date', 'completion date']

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
