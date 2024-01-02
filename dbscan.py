from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.cluster import DBSCAN


sns.set(style="darkgrid")

comp_dir = Path('dataset\\')
data = pd.read_csv(
    comp_dir / 'housing_price.csv',
    usecols=['SquareFeet', 'Bedrooms', 'Bathrooms', 'Neighborhood', 'YearBuilt', 'Price'],
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

plt.figure(figsize=(10, 8))
plt.plot(range(1, data_scaled.shape[1] + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.title('Varianza tra le componenti')
plt.xlabel('Numero di componenti')
plt.ylabel('Varianza comulativa')
plt.show()

pca = PCA(n_components=5, svd_solver='full')
pca.fit(data_scaled)
scores_pca = pca.transform(data_scaled)
print(f"PCA: {scores_pca.shape}")

#DB-SCAN

neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(scores_pca)
distances, indices = nbrs.kneighbors(scores_pca)

distances = np.sort(distances, axis=0)
plt.figure(figsize=(12,8))
plt.plot(distances[:,1])

db = DBSCAN(eps=2.3, min_samples=5).fit(scores_pca)
ymeans = db.labels_
print(f"ymeas: {set(ymeans)}")
df_segm_pca_dbScan = pd.concat([data.reset_index(drop=True),pd.DataFrame(ymeans)],axis=1)
df2 = pd.DataFrame()
for i in range(-1,len(set(ymeans))-1):
    df = df_segm_pca_dbScan[df_segm_pca_dbScan[0]==i]
    df.to_csv("clusters/cluster_"+str(i)+".csv",index=False)

dbscan_silhuouette = silhouette_score(scores_pca,db.labels_).round(2)
print(dbscan_silhuouette)


# Metodo per trovare eps
def get_metrics(eps, min_samples, dataset, iter_):
    # Fitting ======================================================================

    dbscan_model_ = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_model_.fit(dataset)

    # Mean Noise Point Distance metric =============================================
    noise_indices = dbscan_model_.labels_ == -1

    if True in noise_indices:
        neighboors = NearestNeighbors(n_neighbors=6).fit(dataset)
        distances, indices = neighboors.kneighbors(dataset)
        noise_distances = distances[noise_indices, 1:]
        noise_mean_distance = round(noise_distances.mean(), 3)
    else:
        noise_mean_distance = None

    # Number of found Clusters metric ==============================================

    number_of_clusters = len(set(dbscan_model_.labels_[dbscan_model_.labels_ >= 0]))

    # Log ==========================================================================

    print("%3d | Tested with eps = %3s and min_samples = %3s | %5s %4s" % (
    iter_, eps, min_samples, str(noise_mean_distance), number_of_clusters))

    return (noise_mean_distance, number_of_clusters)


# todo per qualche ragione nessun dato rimane unclustered
# quindi questa sezione finale non serve....


eps_to_test = [round(eps, 1) for eps in np.arange(2, 3, 0.1)]
min_samples_to_test = range(5, 50, 5)

print("EPS:", eps_to_test)
print("MIN_SAMPLES:", list(min_samples_to_test))
# Dataframe per la metrica sulla distanza media dei noise points dai K punti più vicini
results_noise = pd.DataFrame(
    data=np.zeros((len(eps_to_test), len(min_samples_to_test))),  # Empty dataframe
    columns=min_samples_to_test,
    index=eps_to_test
)

# Dataframe per la metrica sul numero di cluster
results_clusters = pd.DataFrame(
    data=np.zeros((len(eps_to_test), len(min_samples_to_test))),  # Empty dataframe
    columns=min_samples_to_test,
    index=eps_to_test
)
iter_ = 0

print("ITER| INFO%s |  DIST    CLUS" % (" " * 39))
print("-" * 65)

for eps in eps_to_test:
    for min_samples in min_samples_to_test:
        iter_ += 1

        # Calcolo le metriche
        noise_metric, cluster_metric = get_metrics(eps, min_samples, scores_pca, iter_)

        # Inserisco i risultati nei relativi dataframe
        results_noise.loc[eps, min_samples] = noise_metric
        results_clusters.loc[eps, min_samples] = cluster_metric

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8) )

sns.heatmap(results_noise, annot = True, ax = ax1, cbar = False).set_title("METRIC: Mean Noise Points Distance")
sns.heatmap(results_clusters, annot = True, ax = ax2, cbar = False).set_title("METRIC: Number of clusters")

ax1.set_xlabel("N")
ax1.set_ylabel("EPSILON")
ax2.set_xlabel("N")
ax2.set_ylabel("EPSILON")
plt.tight_layout()
plt.show()