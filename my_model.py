import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import mean_squared_error


# plot params
#todo
plt.style.use("seaborn-whitegrid")
# set ticks and tick  labels
fig, ax = plt.subplots()
"""ax.set_xticklabels(
        ['2013', '2014', '2015', '2016', '2017'],
        rotation=30,
)"""

# read csv
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

comp_dir = Path('dataset\\')
holidays_events = pd.read_csv(
    comp_dir / "holidays_events.csv",
    dtype={
        'type': 'category',
        'locale': 'category',
        'locale_name': 'category',
        'description': 'category',
        'transferred': 'bool',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
holidays_events = holidays_events.set_index('date').to_period('D')

oil = pd.read_csv(
    comp_dir / "oil.csv",
    dtype={
        'dcoilwtico': 'float64',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
oil = oil.set_index('date').to_period('D')
oil = oil.rename(columns={'dcoilwtico':'price'})

store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
store_sales['date'] = store_sales.date.dt.to_period('D')

stores = pd.read_csv(
    comp_dir / 'stores.csv',
    dtype={
        'store_nbr': 'category',
        'city': 'category',
        'state': 'category',
        'type': 'category',
        'cluster': 'category',
    }
)

store_sales = store_sales.merge(stores, on='store_nbr', how='left')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()
average_sales = (
    store_sales
    #.query("~date.isin(['2013-12-31','2014-01-01','2014-12-31','2015-01-01','2015-01-02','2015-12-31','2016-01-01',"
    #       "'2017-01-01','2017-01-02'])")
    .loc[~((store_sales.index.get_level_values('date').month.isin([4, 5]))
           & (store_sales.index.get_level_values('date').year == 2016)
           )]
    .groupby('date').mean()
    .squeeze()
    #.loc['2014']
)


# - - - feature mensili e settimanali - - -
y = average_sales.copy()

dp = DeterministicProcess(
    index=y.index,
    constant=True,
    order=4, # todo? avevo provato prima non ho più cambiato ordine con le nuove soluzioni...
    seasonal=True,
    additional_terms=[CalendarFourier(freq='M', order=4)], #todo? ho provato con 2,3,6 mesi ed è paggio, settima anche
    drop=True,
)
X = dp.in_sample()

model = LinearRegression().fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index)

ax = y[:'2016-04-01'].plot(alpha=0.5, title="Average Sales", ylabel="items sold", color='C0')
ax = y['2016-06-01':].plot(alpha=0.5, title="Average Sales", ylabel="items sold", color='C0')
ax = y_pred[:'2016-04-01'].plot(ax=ax, label="Seasonal", color='C1')
ax = y_pred['2016-06-01':].plot(ax=ax, label="Seasonal", color='C1')
ax.legend()
plt.show()

print("Average Sales")
print(mean_squared_error(y, y_pred))
print(mean_squared_error(y, y_pred, squared=False))
print()

# todo? aggiungo i dati del nuovo anno meno la stagionalità
# deseas = y-y_pred



# query delle holiday   #TODO migliorare la query == migliorare il modello
holidays = (
    holidays_events
    .query("transferred == False")
    #.query("locale in ['National', 'Regional']")
    .loc['2013':'2017-08-15', ['description']]
    .assign(description=lambda x: x.description.cat.remove_unused_categories())
)
# creazione di feature one-hot relative le festività
holidays = holidays[~holidays.index.duplicated(keep='first')]

X_holidays = pd.get_dummies(holidays)


X2 = X.join(X_holidays, on='date').fillna(0.0)

model = LinearRegression().fit(X2, y)
y_pred = pd.Series(
    model.predict(X2),
    index=X2.index,
    name='Fitted',
)

y_pred = pd.Series(model.predict(X2), index=X2.index)

ax = y[:'2016-04-01'].plot(alpha=0.5, title="Average Sales+Holiday", ylabel="items sold", color='C0')
ax = y['2016-06-01':].plot(alpha=0.5, title="Average Sales+Holiday", ylabel="items sold", color='C0')
ax = y_pred[:'2016-04-01'].plot(ax=ax, label="Seasonal", color='C1')
ax = y_pred['2016-06-01':].plot(ax=ax, label="Seasonal", color='C1')
holiday_dates = holidays.loc[holidays.index.isin(X2.index)]
plt.plot_date(holiday_dates.index, y[holiday_dates.index], color='C3', label='Holidays')
ax.legend()
plt.show()

print("Average Sales+Holiday")
print(mean_squared_error(y, y_pred))
print(mean_squared_error(y, y_pred, squared=False))
print()

# Aggiunta del costo del petrolio

#fill nans todo: capire come migliorare...
X_oil = oil.copy().fillna(method='bfill')
# add lags, compute difference
# todo numero lags...
for i in range(1, 4):
    X_oil[f'price_lag_{i}'] = X_oil['price'].shift(i)

for i in range(1, 4):
    X_oil[f'price_lag_{i}'] = X_oil['price'] - X_oil[f'price_lag_{i}']


X_oil = oil.copy().fillna(method='bfill')
# create the new dataset adding the oil price lags, and deleting the first 15 rows
X3 = X2.join(X_oil, on='date', how='inner').fillna(0.0) #, how='inner')
y1 = y.loc[ y.index.isin(X3.index)]

model = LinearRegression().fit(X3, y1)
y_pred = pd.Series(model.predict(X3), index=X3.index)


ax = y1[:'2016-04-01'].plot(alpha=0.5, title="Average Sales+Holiday+Oil(3)", ylabel="items sold", color='C0')
ax = y1['2016-06-01':].plot(alpha=0.5, title="Average Sales+Holiday+Oil(3)", ylabel="items sold", color='C0')
ax = y_pred[:'2016-04-01'].plot(ax=ax, label="Forecast", color='C1')
ax = y_pred['2016-06-01':].plot(ax=ax, label="Forecast", color='C1')
ax = X_oil['price'][:'2016-04-01'].plot(ax=ax, label="Oil price", color='C3')
ax = X_oil['price']['2016-06-01':].plot(ax=ax, label="Oil price", color='C3')
ax.legend()
plt.show()

print("Average Sales+Holiday+Oil(3)")
print(mean_squared_error(y1, y_pred))
print(mean_squared_error(y1, y_pred, squared=False))
print()



#TODO migliorare. migliora pochissimo il modello
# Create 'payday' column with True for the 15th and the last day of each month
X4 = X3.copy()
X4 = X4.reset_index()
X4['payday'] = (X4['date'].dt.day == 16) | (X4['date'].dt.day == 1)
X4 = X4.set_index('date')

model = LinearRegression().fit(X4, y1)

y_pred = pd.Series(model.predict(X4), index=X4.index)

ax = y1[:'2016-04-01'].plot(alpha=0.5, title="Average Sales+Holiday+Oil(1)+Paydays", ylabel="items sold", color='C0')
ax = y1['2016-06-01':].plot(alpha=0.5, title="Average Sales+Holiday+Oil(1)+Paydays", ylabel="items sold", color='C0')
ax = y_pred[:'2016-04-01'].plot(ax=ax, label="Forecast", color='C1')
ax = y_pred['2016-06-01':].plot(ax=ax, label="Forecast", color='C1')
ax.legend()
plt.show()

print("Average Sales+Holiday+Oil(3)+Paydays")
print(mean_squared_error(y1, y_pred))
print(mean_squared_error(y1, y_pred, squared=False))
print()


"""
def store_locs():
    stores = pd.read_csv(
        comp_dir / "stores.csv",
        dtype={
            'store_nbr': 'int',
            'city': 'category',
            'state': 'category',
            'type': 'category',
            'cluster': 'category',
        },
    )
    stores.set_index('store_nbr')

    stores_per_city =  stores.groupby(['city']).sum()
    stores_per_state = stores.groupby(['state']).sum()
    tot_stores = stores.sum()
    print(stores_per_city.head(5))
    print(stores_per_state.head(5))
    print(tot_stores)

    return (stores_per_city, stores_per_state, tot_stores)


for index, row in X_holidays.iterrows():
    for i,h in holidays.iterrows():
        if row[h['description']] == 1.0:
            row[h['description']] = tot_stores if h['locale_name']=='Ecuador'\
            else  stores_per_state[h['locale_name']] if h['locale']=='Regional'\
                else stores_per_city[h['locale_name']]

"""
#fallimento^