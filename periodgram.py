from warnings import simplefilter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

simplefilter("ignore")

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

# %config InlineBackend.figure_format = 'retina'


# annotations: https://stackoverflow.com/a/49238256/5769929
def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax


def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = 1457 # pd.Timedelta("365D") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("symlog")
    ax.set_xticks([4, 8, 16, 24, 48, 104, 208, 416, 1457])

    ax.set_xticklabels(
        [
            "Annual (4)",
            "Semiannual (8)",
            "Quarterly (16)",
            "Bimonthly (24)",
            "Monthly (48)",
            "Biweekly (104)",
            "Weekly (208)",
            "Semiweekly (416)",
            "Daily (1457)"
        ],
        rotation=30,
    )

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax


df = pd.read_csv("dataset\\train.csv", parse_dates=["date"])
df = df.interpolate(method='bfill')

X = df.copy()[['date', 'sales']]
X = X.set_index("date").to_period("D")
X = X.groupby(['date']).sum(['sales'])

# days within a week
X["day"] = X.index.dayofweek
X["week"] = X.index.week
X["month"] = X.index.month
# days within a year
X["dayofyear"] = X.index.dayofyear
X["year"] = X.index.year

print(X.shape)
X = X[X['year'] < 2017]
print(X.shape)

fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(20, 20))
seasonal_plot(X, y="sales", period="week", freq="day", ax=ax0)
seasonal_plot(X, y="sales", period="year", freq="month", ax=ax1)
seasonal_plot(X, y="sales", period="year", freq="dayofyear", ax=ax2)

plot_periodogram(X.sales)
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

decompose_result_mult = seasonal_decompose(X, model="additive", period=30)

trend = decompose_result_mult.trend
seasonal = decompose_result_mult.seasonal
residual = decompose_result_mult.resid
decompose_result_mult.plot()
plt.show()

