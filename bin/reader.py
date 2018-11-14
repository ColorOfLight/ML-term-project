import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from plots import draw_corr_heatmap
import seaborn as sns

# Varaibles
train_rate = .85

np.random.seed(0)

names = ['contract_date', 'latitude', 'longtitude', 'altitude', '1st region id', '2nd region id', 'road id',
         'apartment id', 'floor', 'angle', 'area', 'parking lot limit', 'parking lot area', 'parking lot external',
         'management fee', 'households', 'age of residents', 'builder id', 'completion date', 'built year',
         'schools', 'bus stations', 'subway stations', 'price']

# (240593, 24)
data = pd.read_csv('../data/data_train.csv',
          names=names)

draw_corr_heatmap(data)

# x = data['latitude']
# y = data['longtitude']
# z = np.log(data['price'])

# colormap = plt.cm.get_cmap('Blues')  # or any other colormap
# normalize = mpl.colors.Normalize(vmin=-1, vmax=1)

# plt.title('management fee-price')
# plt.scatter(x, y, c=z, cmap=colormap, s=4)
# plt.show()

# Create Train & test data
# train_indexes = np.random.rand(len(data)) <= 0.85

# X_train = data[names[:-1]][train_indexes]
# X_test = data[names[:-1]][~train_indexes]
# y_train = data[names[-1]][train_indexes]
# y_test = data[names[-1]][~train_indexes]

