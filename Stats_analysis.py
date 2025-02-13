import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import style
# import seaborn as sns

style.use('ggplot')

data = pd.read_csv('our_data.csv')
names_metric = ['Sphere', 'Euclidean', 'Manhattan', 'Octile', 'Chebyshev']
cols = ['Sphere Distance', 'Time(s)', 'Number of Steps', 'Algorithm', 'Metric Space', 'Number of Vertex',
        'Total Path Weight', 'Network Type']

# data = data.loc[data['Metric Space'] != 'H(x, y)=0', :]

points = np.unique(data['Sphere Distance'])
xx = []
for p in points:
   for way in ['drive', 'bike']:
       to_calc = data.loc[(data['Sphere Distance'] == p) & (data['Network Type'] == way), :]
       w = to_calc.loc[to_calc['Algorithm'] != 'A*', 'Total Path Weight'].to_numpy(dtype=float)[0]
       to_calc = np.abs(to_calc.loc[:, 'Total Path Weight'].to_numpy(dtype=float) - w)
       xx.extend(to_calc.tolist())

data['Precision'] = xx

plt.ylabel('Error')
plt.xlabel('$|V|$')
plt.title('Mean of $A^*$s Total Error')
data = data.loc[data['Algorithm'] != 'Dijkstra']

sns.lineplot(x='Number of Vertex', y='Precision', data=data, hue='Algorithm', style='Network Type')
plt.show()
# sns.catplot(x='Metric Space', y='Number of Steps', data=data, col='Network Type', hue='Sphere Distance', kind='bar')
