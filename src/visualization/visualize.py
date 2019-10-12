import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter

from scipy.stats import norm
from statsmodels.graphics.gofplots import qqplot
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

register_matplotlib_converters()


def plot_prediction(full_pred_values, goback_years=None):
    """
    Plots the models's output as blue lines, and actual values as black dots
    Drwas a red vertical line at the point where the out-of-sample predictions start
    returns matplotlib, figure and axis objects
    """
    df = full_pred_values.copy(deep=True)
    final_year = df.index.year.unique()[-1]
    oos_start = df.loc[str(final_year)].index[0]

    if goback_years:
        start_year = final_year - goback_years + 1
        df = df.loc[str(start_year) : str(final_year)]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(df["y"], "ko", markersize=3, label="Actual")
    ax.plot(df["yhat"], color="steelblue", lw=0.5, label="Predicted")
    ax.axvline(oos_start, color="r", alpha=0.8)
    if 'yhat_lower' in full_pred_values.columns:
        ax.fill_between(df.index,
                        df["yhat_lower"],
                        df["yhat_upper"],
                        color="blue",
                        alpha=0.05,
                        label="Confidence Interval")
    ax.grid(ls=":", lw=0.1, color="k")
    plt.legend()

    return fig, ax

def resids_vs_preds_plot(pred_vals):
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(pred_vals[pred_vals['is_forecast'] == 1]['y'].values,
              pred_vals[pred_vals['is_forecast'] == 1]['resid'].values)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Predictions')
    return fig, ax


def plot_joint_plot(full_pred_values, goback_years=1):
    """
    
    """
    axes_titles = {"yhat": "Predicted", "y": "Actual"}

    df = full_pred_values.copy(deep=True)
    final_year = df.index.year.unique()[-1]
    #oos_start = df.loc[str(final_year)].index[0]

    #if goback_years:
    start_year = final_year - goback_years + 1
    df = df.loc[str(start_year) : str(final_year)]

    g = sns.jointplot(x="yhat", y="y", data=df, kind="reg", color="0.4")

    g.fig.set_figwidth(10)
    g.fig.set_figheight(10)
    g.ax_joint.set_xlabel(axes_titles["yhat"])
    g.ax_joint.set_ylabel(axes_titles["y"])

    return g, g.ax_joint


def residual_plots(full_pred_values, figsize=(14, 14), bins=10, goback_years=1):
    """
    Produce a set of residual diagnosos plots similar to statsmodels tome series analysis
    """
    df = full_pred_values.copy(deep=True)
    final_year = df.index.year.unique()[-1]
    start_year = final_year - goback_years + 1
    df = df.loc[str(start_year) : str(final_year)]
    
    resids = df['resid']
        
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 14))
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]

    # ax1
    ax1.plot(resids)
    ax1.set_title("Line Plot")

    # ax2
    ax2.hist(resids, bins=bins, alpha=0.5, density=True)
    pd.Series(resids).plot(kind="density", ax=ax2, label="data")
    # Fit statsmodels distribution here and plot density
    mu, std = norm.fit(resids)
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p1 = norm.pdf(x, 0, std)
    ax2.plot(x, p1, "k", label="fitted normal: (0, data std)")
    ax2.legend()
    ax2.set_title("Distribution")

    # ax3
    qqplot(resids, ax=ax3, line="s")
    ax3.set_title("QQ Plot")

    # ax4
    plot_acf(resids, ax=ax4, zero=False)

    return fig, axes

def print_residual_stats(predicted_vals, goback_years=1):
                               
    df = predicted_vals.copy(deep=True)
    final_year = df.index.year.unique()[-1]
    
    start_year = final_year - goback_years + 1
    df = df.loc[str(start_year) : str(final_year)]
    
    resids = df['resid']
                               
    print(
        f"LJung Box Corr p value:\t\t{round(acorr_ljungbox(resids)[1][:10].max(),5)}"
    )
    print(
        f"Jarque Bera Normal p value:\t{round(jarque_bera(resids, axis=0)[1], 5)}"
    )
    
def seasonal_plot(df: pd.DataFrame, labels: list, date_format: str,
                  mdates_locator, figsize=(12,8)):
    """
    If labels is an empty list then no labels attributed to series 
    
    """
    myFmt = DateFormatter(date_format) 
    x = df.index
    seriess = [df[col] for col in df.columns]
    fig, ax1 = plt.subplots(frameon=False, figsize=figsize)
    #ax1.set_prop_cycle('color', palettable.colorbrewer.qualitative.Set1_9.mpl_colors)
    if len(labels) == 0:
        labels = ['' for l in range(len(df.columns))]
    for series, label in zip(seriess, labels):
        ax1.plot(x, series)
    ax1.xaxis.set_major_formatter(myFmt)
    ax1.xaxis.set_major_locator(mdates_locator)

    return fig, ax1

