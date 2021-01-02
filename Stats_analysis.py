import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import style

style.use('ggplot')

data = pd.read_csv('our_data.csv')
names_metric = ['Sphere', 'Euclidean', 'Manhattan', 'Octile', 'Chebyshev']
cols = ['Sphere Distance', 'Time(s)', 'Number of Steps', 'Algorithm', 'Metric Space']

plt.ylabel('Time(s)')
plt.xlabel('$d(p_1, p_2)$')
points = np.unique(data['Sphere Distance'])
x = []
y = []

for point in points:
    df = data.loc[data['Sphere Distance'] == point, :]
    x.append(np.mean(df.loc[df['Algorithm'] == 'A*', 'Time(s)']))
    y.append(np.mean(df.loc[df['Algorithm'] == 'Dijkstra', 'Time(s)']))

plt.title('Time(s): $A^*$ vs. Dijkstra')
plt.plot(points, x, label='$A^*$')
plt.plot(points, y, label='Dijkstra')

plt.legend()
plt.show()
