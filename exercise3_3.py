# Exercise 3.3
# Modify Example 3.4 so that it plots the first two features (sepal length
# and sepal width) of all data points as a scatter plot.
import pandas as pd
import matplotlib.pyplot as plt

url = ('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/'
       '0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
df = pd.read_csv(url)
df = df.dropna()

species_codes = {name: i for i, name in enumerate(sorted(df['species'].unique()))}
colors = df['species'].map(species_codes)

plt.scatter(df['sepal_length'], df['sepal_width'], c=colors)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Iris: sepal length vs sepal width')
plt.show()
