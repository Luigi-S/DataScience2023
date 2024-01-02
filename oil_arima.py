import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA as arima

from pmdarima import auto_arima

df = pd.read_csv("dataset\\oil.csv", parse_dates=["date"])

seas_df = df.copy()

oil_df = df.copy()
oil_df = oil_df.set_index("date").to_period()
oil_df['time_dummy'] = np.arange(len(oil_df.index))

plt.style.use("seaborn-whitegrid")
plt.figure(figsize=(10, 6))
fig, ax = plt.subplots()
ax = sns.lineplot(x='time_dummy', y='dcoilwtico', data=oil_df, marker='o', linestyle='-')
# linear regression
ax = sns.regplot(x='time_dummy', y='dcoilwtico', data=oil_df, ci=None)

# sns.lineplot(x='date', y='dcoilwtico', data=oil_df, linestyle='-')


plt.title('Time Series of oil prices')
plt.xlabel('Date')
plt.ylabel('Oil price')
plt.show()


# +--- TREND (ma penso si a fatto male) ---+
moving_average = df.rolling(
    window=365,       # 365-day window
    center=True,      # puts the average at the center of the window
    min_periods=183,  # choose about half the window size
).mean()              # compute the mean (could also do median, std, min, max, ...)

ax = df.plot(style=".", color="0.5")
moving_average.plot(
    ax=ax, linewidth=3, title="Tunnel Traffic - 365-Day Moving Average", legend=False,
)
plt.show()

from sklearn.linear_model import LinearRegression

df.dropna(inplace=True)
"""
for i in range(5):
    dp = DeterministicProcess(
        index=df.index,  # dates from the training data
        constant=True,       # dummy feature for the bias (y_intercept)
        order=i,             # the time dummy (trend)
        drop=True,           # drop terms if necessary to avoid collinearity
    )
    # `in_sample` creates features for the dates given in the `index` argument
    X = dp.in_sample()
    y= df['dcoilwtico']
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    y_pred = pd.Series(model.predict(X), index=X.index)
    ax = df.plot(style=".", color="0.5", title="Trend"+str(i))
    _ = y_pred.plot(ax=ax, linewidth=3, label="Trend")
    plt.show()
"""

"""
# +*+*+ SEASONALITY +*+*+
from statsmodels.tsa.seasonal import seasonal_decompose
plt.style.use("seaborn-whitegrid")
plt.figure(figsize=(10, 6))

seas_df.dropna(inplace=True)
# seas_df.interpolate(inplace=True, method='bfill')
seas_df.set_index('date', inplace=True)
decompose_result_mult = seasonal_decompose(seas_df['dcoilwtico'], model="multiplicative", period=84)

trend = decompose_result_mult.trend
seasonal = decompose_result_mult.seasonal
residual = decompose_result_mult.resid

decompose_result_mult.plot()
plt.show()
"""

# **> ADF test <**
time_series = oil_df["dcoilwtico"]
# provvisoriamente drop dei nan
time_series.dropna(inplace=True)


def perform_adf_test(series):
    # Return val
    not_stationary = True
    # Perform ADF test
    result = adfuller(series)
    adf_statistic, p_value, _, _, critical_values, _ = result

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

# **> Autocorrelazione (k,q) <**
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 6))
pd.plotting.lag_plot(time_series, lag=1, ax=ax1)
pd.plotting.lag_plot(time_series, lag=2, ax=ax2)
pd.plotting.lag_plot(time_series, lag=3, ax=ax3)
pd.plotting.lag_plot(time_series, lag=4, ax=ax4)
plt.title('Lag Plot')
plt.show()

df1 = oil_df['dcoilwtico'].dropna()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 6))
pd.plotting.lag_plot(df1, lag=10, ax=ax1)
pd.plotting.lag_plot(df1, lag=20, ax=ax2)
pd.plotting.lag_plot(df1, lag=30, ax=ax3)
pd.plotting.lag_plot(df1, lag=40, ax=ax4)
plt.title('Lag Plot NON DIFF')
plt.show()

plot_acf(time_series)
plt.title('Autocorrelation Plot')
plt.show()

# Plot partial autocorrelation
plot_pacf(time_series, lags=20)
plt.title('Partial Autocorrelation Plot')
plt.show()

"""
def arima_fit(time_series, p=1, d=1, q=1):
    # Fit ARIMA model
    arima_model = arima(time_series, order=(p, d, q))  # Example order, adjust as needed
    arima_results = arima_model.fit()

    print(arima_results.summary())


df1 = oil_df['dcoilwtico'].dropna()
for p in range(2):
    for q in range(2):
        arima_fit(df1,p,d,q)


auto_fit = auto_arima( oil_df["dcoilwtico"].dropna(), start_p=0, start_q=0,
                          max_p=2, max_q=2,
                          m=1,                     #TODO m is used for seasonality, m=1 means no seasonality (cover this later)
                          seasonal=False,          # We do not want seasonality here
                          d=None,                  # The order of first-differencing. If None (by default), automatically be selected
                          trace=True,
                          #error_action='ignore',   # we don't want to know if an order does not work
                          suppress_warnings=True,  # we don't want convergence warnings
                          stepwise=True)           # set to stepwise

auto_fit.summary()


# ---------------------------------------------------


# KPSS test
result_kpss = kpss(time_series_data)
kpss_statistic, p_value, lags, critical_values = result_kpss

# Print results
print(f'KPSS Statistic: {kpss_statistic}')
print(f'p-value: {p_value}')
print(f'Lags Used: {lags}')
print('Critical Values:')
for key, value in critical_values.items():
    print(f'   {key}: {value}')


# Fit ARIMA model
# arima_model = sm.tsa.ARIMA(time_series_data, order=(1, 1, 1))  # Example order, adjust as needed
# arima_results = arima_model.fit()

# Print ARIMA model summary
# print(arima_results.summary())

"""
