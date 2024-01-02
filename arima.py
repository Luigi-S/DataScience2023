# %%

from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


def joined_df():
    df = pd.read_csv("dataset\\train.csv", parse_dates=['date'])
    stores_df = pd.read_csv("dataset\\stores.csv").set_index('store_nbr')
    return df.join(stores_df, on='store_nbr')


# %%

# - - - Time series setup - - -

df = pd.read_csv("dataset\\train.csv", parse_dates=['date'])
df['time'] = (df['date'] - df['date'].min()).dt.days
# data = df.copy()

# data = data.drop(['store_nbr'], axis=1).groupby(['family', 'time']).sum(['sales'])

sales_per_day = df.copy()[['time', 'sales']]
sales_per_day.dropna(inplace=True)
sales_per_day = sales_per_day.groupby(['time']).mean(['sales'])
print(sales_per_day.info)

plt.style.use("seaborn-whitegrid")
plt.figure(figsize=(30, 20))



# %%

# - - - Plot series and linear regression - - -

fig, ax = plt.subplots()
ax.set_xticks([0, 365, 730, 1095, 1461])

ax.set_xticklabels(
        ['2013', '2014', '2015', '2016', '2017'],
        rotation=30,
)

families = df['family'].unique()
"""
data['time'] = data.index.get_level_values('time')
for family in families[0:4]:
    fam_df = data.iloc[data.index.get_level_values('family') == family]
    fam_df['date_'] = fam_df.index.get_level_values('time')
    sns.lineplot(data=fam_df, x="date_", y="sales", ax=ax, marker='o', linestyle='-', label=family)
    sns.regplot(data=fam_df, x='date_', y='sales', ax=ax, fit_reg=True, ci=None, label=f'lr: {family}')
"""

# sns.lineplot(data=sales_per_day, x="time", y="sales", ax=ax, linestyle='-', label='Sales')
sns.regplot(data=sales_per_day, x=sales_per_day.index, y='sales', ax=ax, fit_reg=True, label='Linear Reg.',
            scatter_kws={"color": "black"}, line_kws={"color": "red"})

ax.legend()
plt.title('Time Series')
plt.xlabel('Date')
plt.ylabel('Sales')

plt.show()
"""
"""

# %%


# - - - TREND (ma penso si a fatto male) - - -

# Moving Average per mese, per settimana
moving_average_monthly = sales_per_day.copy()
moving_average_monthly['sales'] = moving_average_monthly['sales'].rolling(
    window=28,  # 4 weeks window
    center=True,  # puts the average at the center of the window
    min_periods=14,  # TIP: choose about half the window size
).mean()  # compute the mean (could also do median, std, min, max, ...)

moving_average_trimester = sales_per_day.copy()
moving_average_trimester['sales'] = moving_average_trimester['sales'].rolling(
    window=84,  # 3 (fiscal) months window
    center=True,  # puts the average at the center of the window
    min_periods=42,  # TIP: choose about half the window size
).mean()  # compute the mean (could also do median, std, min, max, ...)


fig, ax = plt.subplots()
ax.set_xticks([0, 365, 730, 1095, 1461])

ax.set_xticklabels(
        ['2013', '2014', '2015', '2016', '2017'],
        rotation=30,
)

sns.lineplot(x='time', y='sales', data=sales_per_day, ax=ax, label='Sales')
sns.lineplot(x='time', y='sales', data=moving_average_trimester, ax=ax, label='MA-trimestre')
sns.lineplot(x='time', y='sales', data=moving_average_monthly, ax=ax,  label='MA-mese')

ax.legend()
plt.title('Time Series')
plt.xlabel('Date')
plt.ylabel('Sales')

plt.show()


# %%

# - - - Linear Regression - - -

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier

fig, ax = plt.subplots()
ax.set_xticks([0, 365, 730, 1095, 1461])

ax.set_xticklabels(
        ['2013', '2014', '2015', '2016', '2017'],
        rotation=30,
)
y = sales_per_day['sales'].groupby('time').mean().squeeze()  # TODO grouping with 'family'?
"""
sns.lineplot(x='time', y='sales', data=sales_per_day, ax=ax)

for i in range(5):
    dp = DeterministicProcess(
        index=y.index,  # dates from the training data
        constant=True,  # dummy feature for the bias (y_intercept)
        order=i,  # the time dummy (trend)
        seasonal=True,
        additional_terms=[CalendarFourier(freq='M', order=4)],
        drop=True,  # drop terms if necessary to avoid collinearity
    )
    # `in_sample` creates features for the dates given in the `index` argument
    X = dp.in_sample()

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    sales_pred = sales_per_day.copy()
    sales_pred['sales'] = model.predict(X)

    label = f'Trend: {i}' if i > 0 else 'Mean'
    sns.lineplot(x='time', y='sales', data=sales_pred, ax=ax, label=label)
ax.legend()
plt.show()
"""

# %%

# +*+*+ SEASONALITY +*+*+

from statsmodels.tsa.seasonal import seasonal_decompose
"""
# may prefer interpolation instead of dropping nulls
# df = df.interpolate(method='bfill')
for i in [10, 20, 30, 40, 52, 60, 80, 100]:
    decompose_result_mult = seasonal_decompose(sales_per_day,  # eventually add ['sales']
                                               model="multiplicative", period=i)  # o additive, TODO modificare period?

    trend = decompose_result_mult.trend
    seasonal = decompose_result_mult.seasonal
    residual = decompose_result_mult.resid
    decompose_result_mult.plot()
    plt.show()

for i in [10, 20, 30, 40, 52, 60, 80, 100]:
    decompose_result_mult = seasonal_decompose(sales_per_day,  # eventually add ['sales']
                                               model="additive", period=i)  # o additive, TODO modificare period?

    trend = decompose_result_mult.trend
    seasonal = decompose_result_mult.seasonal
    residual = decompose_result_mult.resid
    decompose_result_mult.plot()
    plt.show()
"""
# %%

# ======================================================================
# =#=#= A*R*I*M*A =#=#=
# ======================================================================


# **> ADF test <**
from statsmodels.tsa.stattools import adfuller

time_series = sales_per_day.copy()['sales']


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

print(f"Value of parameter d: {d}")

# %%
# - - - LAG PLOTS - - -

df1 = sales_per_day['sales'].copy()

"""
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 6))
plt.title('Lag Plot')
pd.plotting.lag_plot(time_series, lag=10, ax=ax1)
plt.show()


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 6))
plt.title('Lag Plot NON DIFF')
pd.plotting.lag_plot(df1, lag=10, ax=ax1)
plt.show()
"""

# **> Autocorrelazione (k,q) <**
# (serie differenziata)
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf


df2 = df1.diff(periods=7).dropna()
plot_acf(df2)
plt.title('Autocorrelation Plot')
plt.show()

# %%

# Plot partial autocorrelation #todo set lag corretto

#for LAG in [1,2,3,4, 5, 10, 20]:
LAG= 30
plot_pacf(df2, lags=LAG)
plt.title(f'Partial Autocorrelation Plot - LAG: {LAG}')
plt.show()


# TODO
decompose_result_mult = seasonal_decompose(df1,  # eventually add ['sales']
                                               model="additive", period=7
                                           )

trend = decompose_result_mult.trend
seasonal = decompose_result_mult.seasonal
residual = decompose_result_mult.resid.dropna()
decompose_result_mult.plot()
plt.show()

""""""
plot_acf(residual)
plt.title('Autocorrelation Plot')
plt.show()
plot_pacf(residual, lags=7)
plt.title(f'Partial Autocorrelation Plot - LAG: {7}')
plt.show()


# %%
"""
# - - - Test statistici per vari set di parametri di ARIMA - - -
from statsmodels.tsa.arima.model import ARIMA as arima

def arima_fit(time_series, p=1, d=1, q=1):
    # Fit ARIMA model
    arima_model = arima(time_series, order=(p, d, q))  # Example order, adjust as needed
    arima_results = arima_model.fit()

    print(arima_results.summary())



for d in range(2):
    for p in range(3):
        for q in range(3):
            arima_fit(df1, p, d, q)

# %%
"""
# - - - AUTO_ARIMA - - -
from pmdarima import auto_arima

auto_fit = auto_arima(df1, start_p=0, start_q=0,
                      max_p=7, max_q=7,
                      m=7,  # m is used for seasonality
                      seasonal=True,  # We do not want seasonality here
                      d=None,  # The order of first-differencing. If None (by default), automatically be selected
                      trace=True,
                      # error_action='ignore',   # we don't want to know if an order does not work
                      suppress_warnings=True,  # we don't want convergence warnings
                      stepwise=True)  # set to stepwise

auto_fit.summary()


