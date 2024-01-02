import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from xgboost import XGBRegressor

from load_datasets import load_oil, load_holidays

comp_dir = Path("dataset")

store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()

family_sales = (
    store_sales
    .loc[~((store_sales.index.get_level_values('date').month.isin([4, 5]))
           & (store_sales.index.get_level_values('date').year == 2016)
           )]
    .groupby(['family', 'date'])
    .mean()
    .unstack('family')
    #.loc['2017']
)


# You'll add fit and predict methods to this minimal class
class HybridModel:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None  # store column names from fit method

    def fit(self, X_1, X_2, y):
        self.model_1 = self.model_1.fit(X_1, y)

        y_fit = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index, columns=y.columns,
        )

        y_resid = y-y_fit
        y_resid = y_resid.stack().squeeze() # wide to long

        self.model_2.fit(X_2, y_resid)

        # Save column names for predict method
        self.y_columns = y.columns
        # Save data for question checking
        self.y_fit = y_fit
        self.y_resid = y_resid

    def predict(self, X_1, X_2):
        y_pred = pd.DataFrame(
            # predict with self.model_1
            self.model_1.predict(X_1),
            index=X_1.index, columns=self.y_columns,
        )
        y_pred = y_pred.stack().squeeze()  # wide to long

        # add self.model_2 predictions to y_pred
        y_pred += self.model_2.predict(X_2)

        return y_pred.unstack()  # long to wide


# Target series
y = family_sales.loc[:, 'sales']


# X_1: Features for Linear Regression
dp = DeterministicProcess(
    index=y.index,
    constant=True,
    order=4,
    seasonal=True,
    additional_terms=[CalendarFourier(freq='M', order=4)],
    drop=True,
)
X = dp.in_sample()

# load other datasets
holidays_events = load_holidays(comp_dir)
oil = load_oil(comp_dir)

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
X_oil = oil.copy().fillna(method='bfill')
# add lags, compute difference
# todo numero lags...
for i in range(1, 4):
    X_oil[f'price_lag_{i}'] = X_oil['price'].shift(i)

for i in range(1, 4):
    X_oil[f'price_lag_{i}'] = X_oil['price'] - X_oil[f'price_lag_{i}']

X_oil = oil.copy().fillna(method='bfill')
X_1 = X2.join(X_oil, on='date', how='inner').fillna(0.0) #, how='inner')

y = y.loc[ y.index.isin(X_1.index)]



# X_2: Features for XGBoost
X_2 = family_sales.loc[ family_sales.index.isin(X_1.index)].drop('sales', axis=1).stack()  # onpromotion feature

# Label encoding for 'family'
le = LabelEncoder()  # from sklearn.preprocessing
X_2 = X_2.reset_index('family')
X_2['family'] = le.fit_transform(X_2['family'])

# Label encoding for seasonality
X_2["day"] = X_2.index.day  # values are day of the month

"""
# Create LinearRegression + XGBRegressor hybrid with BoostedHybrid
model = BoostedHybrid(LinearRegression(), XGBRegressor())

model.fit(X_1, X_2, y)
y_pred = model.predict(X_1, X_2)

y_pred = y_pred.clip(0.0)
"""

# Model 1 (trend)
from sklearn.linear_model import ElasticNet, Lasso, Ridge

# Model 2
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Boosted Hybrid
model = HybridModel(
    model_1=Ridge(alpha=1.0),
    model_2= KNeighborsRegressor(n_neighbors=20)
)

y_train, y_valid = y[:"2017-07-01"], y["2017-07-02":]
X1_train, X1_valid = X_1[: "2017-07-01"], X_1["2017-07-02" :]
X2_train, X2_valid = X_2.loc[:"2017-07-01"], X_2.loc["2017-07-02":]

scaler = MinMaxScaler()
X1_train[:] = scaler.fit_transform(X1_train)
X1_valid[:] = scaler.transform(X1_valid)
X2_train[:] = scaler.fit_transform(X2_train)
X2_valid[:] = scaler.transform(X2_valid)
# Some of the algorithms above do best with certain kinds of
# preprocessing on the features
model.fit(X1_train, X2_train, y_train)
y_fit = model.predict(X1_train, X2_train).clip(0.0)
y_pred = model.predict(X1_valid, X2_valid).clip(0.0)

print(mean_squared_error(y_valid, y_pred))
print(mean_squared_error(y_valid, y_pred, squared=False))
print()


fig, axs = plt.subplots(6,1)
families = y.columns[0:6]

# axs = y.loc(axis=1)[families].plot(subplots=True, sharex=True, figsize=(11, 9), alpha=0.5,)
_ = y['2016':'2016-04-01'].loc(axis=1)[families].plot(subplots=True, sharex=True, figsize=(11, 9), alpha=0.5, ax=axs)
_ = y['2016-06-01':].loc(axis=1)[families].plot(subplots=True, sharex=True, figsize=(11, 9), alpha=0.5, ax=axs)
holiday_dates = holidays.loc[holidays.index.isin(y.index)]
_ = y_fit['2016':'2016-04-01'].loc(axis=1)[families].plot(subplots=True, sharex=True, color='C0', ax=axs)
_ = y_fit['2016-06-01':].loc(axis=1)[families].plot(subplots=True, sharex=True, color='C0', ax=axs)
_ = y_pred.loc(axis=1)[families].plot(subplots=True, sharex=True, color='C3', ax=axs)

for ax, family in zip(axs, families):
    ax.legend([])
    ax.set_ylabel(family, rotation=70, fontsize=7)

plt.title = mean_squared_error(y_valid, y_pred, squared=False)
plt.show()
