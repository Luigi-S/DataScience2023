import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer


oil_df = pd.read_csv("dataset\\oil.csv", parse_dates=["date"])
oil_df.rename(columns={"dcoilwtico": "price"}, inplace=True)
oil_df['time'] = np.arange(len(oil_df.index))


"""
# - - - - null values in oil.csv - - - -
n_neighbors = [1, 2, 3, 5, 7, 9, 20, 30]

fig, ax = plt.subplots(figsize=(16, 8))
# Plot the original distribution
sns.kdeplot(oil_df['price'], label="Original Distribution", color="0.25")
# try k values
for k in n_neighbors:
    knn_imp = KNNImputer(n_neighbors=k)
    imputed = knn_imp.fit_transform(X=oil_df['price'].values.reshape(-1, 1))
    sns.kdeplot(imputed, label=f"Imputed Dist with k={k}")
# plot density of values of price, depending on k
plt.legend()
plt.show()
"""
# k value can be chosen randomly, the selected one is possibly low to ease the computation
k = 5
knn_imp = KNNImputer(n_neighbors=k)
oil_df[['price', 'time']] = knn_imp.fit_transform(X=oil_df[['price', 'time']])
print(oil_df.info)


# - - - - - - GeoData - - - - - -

COORDS = {
    'Ambato': (-1.254340, -78.622792),
    'Babahoyo': (-1.801924, -79.534645),
    'Cayambe': (0.044692, -78.145262),
    'Cuenca': (-2.900128, -79.005896),
    'Daule': (-1.862794, -79.977493),
    'El Carmen': (-0.266667, -79.433333),
    'Esmeraldas': (0.968179, -79.651720),
    'Guaranda': (-1.596667, -79.002778),
    'Guayaquil': (-2.203816, -79.897453),
    'Ibarra': (0.339167, -78.122778),
    'Latacunga': (-0.930556, -78.616667),
    'Libertad': (-2.234167, -79.911389),
    'Loja': (-3.993127, -79.204216),
    'Machala': (-3.258612, -79.955387),
    'Manta': (-0.967653, -80.708910),
    'Playas': (-2.637889, -80.384464),
    'Puyo': (-1.492392, -78.002837),
    'Quevedo': (-1.022392, -79.460888),
    'Quito': (-0.229498, -78.524277),
    'Riobamba': (-1.663550, -78.654646),
    'Salinas': (-2.217500, -80.958611),
    'Santo Domingo': (-0.238890, -79.177432),
}

LATS = {'Ambato': -1.25434, 'Babahoyo': -1.801924, 'Cayambe': 0.044692, 'Cuenca': -2.900128, 'Daule': -1.862794, 'El Carmen': -0.266667, 'Esmeraldas': 0.968179, 'Guaranda': -1.596667, 'Guayaquil': -2.203816, 'Ibarra': 0.339167, 'Latacunga': -0.930556, 'Libertad': -2.234167, 'Loja': -3.993127, 'Machala': -3.258612, 'Manta': -0.967653, 'Playas': -2.637889, 'Puyo': -1.492392, 'Quevedo': -1.022392, 'Quito': -0.229498, 'Riobamba': -1.66355, 'Salinas': -2.2175, 'Santo Domingo': -0.23889}
LONS = {'Ambato': -78.622792, 'Babahoyo': -79.534645, 'Cayambe': -78.145262, 'Cuenca': -79.005896, 'Daule': -79.977493, 'El Carmen': -79.433333, 'Esmeraldas': -79.65172, 'Guaranda': -79.002778, 'Guayaquil': -79.897453, 'Ibarra': -78.122778, 'Latacunga': -78.616667, 'Libertad': -79.911389, 'Loja': -79.204216, 'Machala': -79.955387, 'Manta': -80.70891, 'Playas': -80.384464, 'Puyo': -78.002837, 'Quevedo': -79.460888, 'Quito': -78.524277, 'Riobamba': -78.654646, 'Salinas': -80.958611, 'Santo Domingo': -79.177432}


df = pd.read_csv("dataset\\stores.csv").set_index('store_nbr')
def joined_df():
    df = pd.read_csv("dataset\\train.csv")
    stores_df = pd.read_csv("dataset\\stores.csv").set_index('store_nbr')
    return df.join(stores_df, on='store_nbr')

"""
df = joined_df()
print(df.keys())
print(df.head(5))
"""

# -------------------------
# stores geoplot
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

#geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
#gdf = GeoDataFrame(df, geometry=geometry)

# Load a GeoDataFrame for Ecuador's map
# TODO move shp file with other related files
#ecuador_map = gpd.read_file(    'C:\\Users\\User\\Desktop\\data_science\\ecu_adm_inec_20190724_shp\\ecu_admbnda_adm1_inec_20190724.shp')

# Plot the map
fig, ax = plt.subplots(figsize=(10, 10))
"""
ecuador_map.plot(ax=ax, color='yellow', edgecolor='black')
gdf.plot(ax=ax, marker='o', color='red', markersize=50)

# Add city names as labels
for city, (lon, lat) in COORDS.items():
    ax.text(lon, lat, city, fontsize=8)

plt.title('Map of Ecuador with Cities - 1')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
"""

import plotly.express as px

df1 = df.copy()[['city']]
stores_per_city = df1.value_counts('city')
stores_per_city = stores_per_city.to_frame()
stores_per_city.rename( columns={ 0 :'count'}, inplace=True )
stores_per_city['lat'] = stores_per_city.index.map(LATS)
stores_per_city['lon'] = stores_per_city.index.map(LONS)


fig = px.scatter_mapbox(stores_per_city, lat='lat', lon='lon', size='count',
                        color_discrete_sequence=["fuchsia"],
                        opacity=1, zoom=8, mapbox_style='open-street-map')
#  size='name column', zoom=numero

fig.show()

# -------------------------


"""


"""