from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


sns.set(style="darkgrid")

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

# = = = dummy features generation = = =
dummies = pd.get_dummies(data['Neighborhood'], prefix='state')
# Concatenate the dummy features with the original DataFrame
data = pd.concat([data, dummies], axis=1)
data = data.drop('Neighborhood', axis=1)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

pca = PCA()
pca.fit(data_scaled)
print(pca.explained_variance_ratio_)

plt.figure(figsize=(10,8))
plt.plot(range(1, data_scaled.shape[1] + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.title('Varianza tra le componenti')
plt.xlabel('Numero di componenti')
plt.ylabel('Varianza comulativa')
plt.show()


pca = PCA(n_components = 5, svd_solver='full')
pca.fit(data_scaled)
scores_pca = pca.transform(data_scaled)
print(f"PCA: {scores_pca.shape}")

# - - - Elbow Method  - - -
wcss = []
for i in range(1,50):
    kmeans_pca = KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans_pca.fit(scores_pca)
    wcss.append(kmeans_pca.inertia_)
plt.figure(figsize=(10,8))
plt.plot(range(1,50),wcss,marker='o',linestyle='--')
plt.title('K-Means utilizzando PCA')
plt.xlabel('Numero di cluster')
plt.ylabel('WCSS')
plt.show()

#numero di cluster
kl = KneeLocator(range(1,50),wcss,curve="convex",direction="decreasing")
print(f"Elbow: {kl.elbow}")
kmeans_pca = KMeans(n_clusters=kl.elbow,init='k-means++',random_state=42)
kmeans_pca.fit(scores_pca)

#visualizzazione risultati
df_segm_pca_kmeans = pd.concat([data.reset_index(drop=True),pd.DataFrame(scores_pca)],axis=1)
df_segm_pca_kmeans.columns.values[-4:] = ['C-1','C-2','C-3', 'C-4']
df_segm_pca_kmeans['Segm K-means PCA'] = kmeans_pca.labels_
df_segm_pca_kmeans['Segment'] = df_segm_pca_kmeans['Segm K-means PCA']
#.map({0:'primo',1:'secondo',2:'terzo',3:'quarto',4:'quinto',5:'sesto'})
#todo scommentare per aggiungere legenda
print(df_segm_pca_kmeans.head())

x_axis=df_segm_pca_kmeans['C-1']
y_axis=df_segm_pca_kmeans['C-2']
plt.figure(figsize=(10,8))
sns.scatterplot(x=x_axis,y=y_axis,hue=df_segm_pca_kmeans['Segment'],palette='colorblind')
plt.title('Clusters')
plt.show()

kmeans_silhuouette = silhouette_score(scores_pca,kmeans_pca.labels_).round(2)
print(f"Silhouette: {kmeans_silhuouette}")
from yellowbrick.cluster import SilhouetteVisualizer


# Instantiate the clustering model and visualizer
model = KMeans(6, random_state=42)
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')

plt.figure(figsize=(15,8))
visualizer.fit(scores_pca)        # Fit the data to the visualizer
visualizer.show()