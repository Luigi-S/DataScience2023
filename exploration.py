import pandas as pd

"""
# ---OIL.CSV---
oil_df = pd.read_csv("dataset\\oil.csv")

print("oil.csv shape: " + str(oil_df.shape))
print(oil_df.info)

nan_count = oil_df.isnull().sum().sum()
print('Number of NaN values:', nan_count)

nans = oil_df["dcoilwtico"].isna().groupby(oil_df["dcoilwtico"].notna().cumsum()).sum()
nans = nans[nans != 0]
cons_nan = nans.groupby(nans).size().reset_index(name='Cases')
print("Consecutive NaN values, grouped by length")
print(cons_nan)

"""
# ---HOLIDAYS_EVENT.csv---

holidays = pd.read_csv("dataset\\holidays_events.csv")

print("Holiday types: ")
hol_types = holidays["type"].unique()
hol_types.sort() 
print(hol_types)
print("Holiday localities: ")
hol_locals = holidays["locale"].unique()
hol_locals.sort() 
print(hol_locals)

tranferred = holidays.query(' type == "Transfer" or transferred == True')
print(tranferred.shape)
print(tranferred.info)

print("Holiday transferred, in the observed period: " + str(tranferred.query('transferred == True').shape[0]))
print("Days on which a holiday has been transferred, in the observed period: "
      + str(tranferred.query('type == "Transfer"').shape[0]))

"""
# --- STORES.csv ---
stores = pd.read_csv("dataset\\stores.csv")

print(stores.shape)

print("Cities: ")
cities = stores["city"].unique()
cities.sort()
print(cities)


print("Stores types: ")
store_types = stores["type"].unique()
store_types.sort()
print(store_types)
print("Stores clusters: ")
store_clusters = stores["cluster"].unique()
store_clusters.sort()
print(store_clusters)

grouped_stores = stores.sort_values(by=['type', 'state']).groupby(['state', 'type']).size().reset_index(name='total')
print(grouped_stores)
print('------------------------------------------------------------------')
store_types_count = grouped_stores.drop(columns=['state']).groupby(['type']).sum()
print(store_types_count)

store_clusters_count = stores.sort_values(by=['cluster']).groupby(['cluster']).size().reset_index(name='total').T
print(store_clusters_count)

print("City with the most stores: ")
count_by_city = stores.groupby(['city']).size().reset_index(name='total')
count_by_city_maxes = count_by_city.loc[count_by_city['total'].idxmax()]
print(count_by_city_maxes.T)
"""
"""
# -- Train, test .csv ---
df = pd.read_csv("dataset\\train.csv")

print(df.shape)
print("Sample rows:")
print(df.head(5))
print()

prod_families = df['family'].unique()
print("Product families: " + str(len(prod_families)))
print(prod_families)

days = df['date'].unique()
# totale righe = days * stores * families
print("Observed days: " + str(len(days)) + " From: " + days[0] + " To: " + days[-1])
"""
