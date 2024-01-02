from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

comp_dir = Path('dataset\\')
data = pd.read_csv(
    comp_dir / 'housing_price.csv',
    usecols=['SquareFeet','Bedrooms','Bathrooms','Neighborhood','YearBuilt','Price'],
    dtype={
        'SquareFeet': 'int',
        'Price': 'float32',
        'YearBuilt': 'int',
        'Bedrooms': 'int',
        'Bathrooms': 'int',
        'Neighborhood': 'category',
    },
)

# data['YearBuilt'] = (data['YearBuilt']-data['YearBuilt']%10)

data['PricePerSquareFoot'] = data['Price']/data['SquareFeet']



"""
plt.figure(figsize=(26,24))
fig, ax = plt.subplots()
ax.scatter(data['Price'], data['SquareFeet'])
ax.set_xlabel("Price")
ax.set_ylabel("Square Feet")
plt.show()

plt.figure(figsize=(26,24))
fig, ax = plt.subplots()
ax.scatter(data['Price'], data['Bedrooms'])
ax.set_xlabel("Price")
ax.set_ylabel("Bedrooms")
plt.show()

plt.figure(figsize=(26,24))
fig, ax = plt.subplots()
ax.scatter(data['Price'], data['Bathrooms'])
ax.set_xlabel("Price")
ax.set_ylabel("Bathrooms")
plt.show()

plt.figure(figsize=(26,24))
fig, ax = plt.subplots()
ax.scatter(data['Price'], data['YearBuilt'])
ax.set_xlabel("Price")
ax.set_ylabel("YearBuilt")
plt.show()
"""

def plot_data(data, title):
    sns.set(style="darkgrid")

    plt.figure(figsize=(26,24))
    fig, ax = plt.subplots()
    grid = sns.FacetGrid(data, col = "Bathrooms", hue = "Bathrooms", col_wrap=5)
    grid.map(sns.scatterplot, "Price", "SquareFeet")
    grid.add_legend()
    ax.set_title(title)
    plt.show()

    plt.figure(figsize=(26,24))
    fig, ax = plt.subplots()
    grid = sns.FacetGrid(data, col = "Bedrooms", hue = "Bedrooms", col_wrap=5)
    grid.map(sns.scatterplot, "Price", "SquareFeet")
    plt.show()

    plt.figure(figsize=(26,24))
    fig, ax = plt.subplots()
    grid = sns.FacetGrid(data, col = "Bathrooms", hue = "Bathrooms", col_wrap=5)
    grid.map(sns.scatterplot, "Price", "YearBuilt")
    grid.add_legend()
    plt.show()

    plt.figure(figsize=(26,24))
    fig, ax = plt.subplots()
    grid = sns.FacetGrid(data, col = "Bedrooms", hue = "Bedrooms", col_wrap=5)
    grid.map(sns.scatterplot, "Price", "YearBuilt")
    grid.add_legend()
    plt.show()


#plot_data(data[data['Neighborhood']=='Rural'], 'Rural')
#plot_data(data[data['Neighborhood']=='Suburb'], 'Suburb')
#plot_data(data[data['Neighborhood']=='Urban'], 'Urban')


"""
plt.figure(figsize=(26,24))
fig, ax = plt.subplots()
grid = sns.FacetGrid(data, col = "YearBuilt", hue = "YearBuilt", col_wrap=5)
grid.map(sns.scatterplot, "Price", "SquareFeet")
grid.add_legend()
plt.show()

plt.figure(figsize=(26,24))
fig, ax = plt.subplots()
sns.scatterplot(data=data, x="PricePerSquareFoot", y="YearBuilt")
plt.show()
"""
plt.figure(figsize=(26,24))
fig, ax = plt.subplots()
sns.pairplot(data)
plt.show()

