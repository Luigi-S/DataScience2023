import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from statsmodels.tsa.stattools import adfuller

plt.style.use("seaborn-whitegrid")

comp_dir = Path('dataset\\')
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

store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()
average_sales = (
    store_sales
    .loc[~((store_sales.index.get_level_values('date').month.isin([4, 5])) & (store_sales.index.get_level_values('date').year == 2016))]
    .groupby('date').mean()
    .squeeze()
)

# - - - feature mensili e settimanali - - -
y = average_sales.copy()

dp = DeterministicProcess(
    index=y.index,
    constant=True,
    order=4,
    seasonal=True,
    additional_terms=[CalendarFourier(freq='M', order=4)],
    drop=True,
)
X = dp.in_sample()

model = LinearRegression().fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index)

fig, ax = plt.subplots()
ax = y.plot(alpha=0.5, title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, label="Seasonal")
ax.legend()

plt.show()

print("Average Sales")
print(mean_squared_error(y, y_pred))
print(math.sqrt(mean_squared_error(y, y_pred)))
print()

fig, ax = plt.subplots()
y_deseas = y-y_pred
ax = y_deseas.plot(ax=ax)
plt.show()

time_series = y_deseas.copy()

def perform_adf_test(series):
    # Return val
    not_stationary = True
    # Perform ADF test
    result = adfuller(series)
    adf_statistic, p_value, _, _, critical_values, _ = result

    print()
    # Print ADF test results
    print(f'ADF Statistic: {adf_statistic}')
    print(f'p-value: {p_value}')
    print('Critical Values:')
    for key, value in critical_values.items():
        print(f'   {key}: {value}')

    # Check the p-value
    if p_value <= 0.05:
        print("Reject the null hypothesis. The data is stationary.")
        not_stationary = False
    else:
        print("Fail to reject the null hypothesis. The data is non-stationary.")
    return not_stationary


d = 0
not_stationary = perform_adf_test(time_series)
print('\n')
while not_stationary:
    d += 1
    time_series = time_series.diff().dropna()
    not_stationary = perform_adf_test(time_series)
    print('\n')


from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

plot_acf(time_series)
plt.title('Autocorrelation Plot')
plt.show()

# %%


LAG = 30
plot_pacf(time_series, lags=LAG)
plt.title(f'Partial Autocorrelation Plot - LAG: {LAG}')
plt.show()



# - - - AUTO_ARIMA - - -
from pmdarima import auto_arima

auto_fit = auto_arima(y_deseas, start_p=7, start_q=7,
                      max_p=9, max_q=15,
                      m=1,              # m is for seasonality, removed in y_deseas
                      seasonal=False,  # We do not want seasonality here
                      d=d,  # The order of first-differencing. If None (by default), automatically be selected
                      trace=True,
                      # error_action='ignore',   # we don't want to know if an order does not work
                      suppress_warnings=True,  # we don't want convergence warnings
                      stepwise=True)  # set to stepwise

auto_fit.summary()
