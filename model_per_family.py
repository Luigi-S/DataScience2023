import math

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

YEARS = ['2013', '2014', '2015', '2016', '2017']

# plot params
# todo

plt.style.use("seaborn-whitegrid")

# read csv
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
families = store_sales['family'].unique()
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()
real_sales = (
    store_sales
    .groupby('date').sum()
    .squeeze()
    #.loc['2015']
)

# - - - feature mensili e settimanali - - -
dp = DeterministicProcess(
    index=real_sales.index,
    constant=True,
    order=4,
    seasonal=True,
    additional_terms=[CalendarFourier(freq='M', order=4)],
    drop=True,
)
X = dp.in_sample()

# Create a model for each family and predict mean values
fig, ax = plt.subplots(figsize=(10, 6))

pred_sales = pd.Series(index=real_sales.index, dtype=float)

for family, group in store_sales.groupby('family'):
    y = group.groupby('date').sum().squeeze()

    model = LinearRegression().fit(X, y)
    y_pred = pd.Series(model.predict(X), index=X.index, name=f'Model_{family}')

    pred_sales = pred_sales.add(y_pred, fill_value=0)

ax = real_sales.plot(label='Sales', ylabel="items sold")
ax = pred_sales.plot(label='Forecast')
ax.legend()
ax.set_title("Total Sales Prediction")
plt.show()

print("Average Sales")
print(mean_squared_error(real_sales, pred_sales))
print(math.sqrt(mean_squared_error(real_sales, pred_sales)))
print()

#assolutamente inutile :) a quanto pare già considerava il tutto separato