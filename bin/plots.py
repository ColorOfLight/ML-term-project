import matplotlib.pyplot as plt
import seaborn as sns

def draw_corr_heatmap(data):
  corrmat = data.corr()
  f, ax = plt.subplots(figsize=(12, 9))
  sns.heatmap(corrmat, cmap="PiYG", center=0)
  plt.show();
