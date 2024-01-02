# prendi i file da clusters e mettili insieme su una sola rappresentazione

# scatter plot con distribuzioni sugli assi

#TODO
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

comp_dir = Path('clusters\\')
data = []
for i in range(2):
    data.append(pd.read_csv(
        comp_dir / f'cluster_{i}.csv',
        usecols=['SquareFeet','Bedrooms','Bathrooms','YearBuilt','Price','state_Rural','state_Suburb','state_Urban'],
        dtype={
            'SquareFeet': 'int',
            'Price': 'float32',
            'YearBuilt': 'int',
            'Bedrooms': 'int',
            'Bathrooms': 'int',
            'state_Rural': 'bool',
            'state_Suburb': 'bool',
            'state_Urban': 'bool',
        },
    ))
    data[i]['Cluster'] = i

df = pd.concat([d for d in data], axis=0)

"""plt.figure(figsize=(26, 24))
sns.jointplot(data=df, x="SquareFeet", y="Price", hue="Cluster")
plt.show()

plt.figure(figsize=(26, 24))
sns.jointplot(data=df, x="YearBuilt", y="Price", hue="Cluster")
plt.show()

plt.figure(figsize=(26, 24))
sns.jointplot(data=df, x="SquareFeet", y="YearBuilt", hue="Cluster")
plt.show()

plt.figure(figsize=(26, 24))
sns.jointplot(data=df, x="Bedrooms", y="Bathrooms", hue="Cluster")
plt.show()"""

# Creating dataset

# Creating figure
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")

i=0
for m, zlow, zhigh in [('o', -50, -25), ('o', -30, -5)]:
    xs = df[df['Cluster']==i]['Price']*-1
    ys = df[df['Cluster']==i]['SquareFeet']*-1
    zs = df[df['Cluster']==i]['YearBuilt']
    ax.scatter(xs, ys, zs, marker=m)
    i += 1

ax.set_xlabel('Price')
ax.set_ylabel('Square feet')
ax.set_zlabel('Year')

plt.show()
