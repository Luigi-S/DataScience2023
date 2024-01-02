import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pylab as pylab


def joined_df():
    df = pd.read_csv("dataset\\train.csv", parse_dates=['date'])
    stores_df = pd.read_csv("dataset\\stores.csv").set_index('store_nbr')
    return df.join(stores_df, on='store_nbr')

df = joined_df()

#sns.pairplot(df) #, kind='reg')
# to show
#plt.show()


heatmap1 = sns.heatmap(df.corr(method = 'spearman'), vmin=-1, vmax=1, annot=True)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap1.set_title('Spearman Correlation Heatmap', fontdict={'fontsize':12}, pad=12)

plt.show()
"""
plt.figure(figsize=(16, 12))
sns.set(font_scale=2)
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title('Pearson Correlation Heatmap', fontdict={'fontsize':24}, pad=12)

plt.show()
"""