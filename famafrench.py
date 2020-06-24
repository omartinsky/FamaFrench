# Copyright © 2020 Ondrej Martinsky, All rights reserved
# http://github.com/omartinsky/FamaFrench

import functools
from typing import Dict, Any, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import time
import math
import os
import seaborn as sns
import matplotlib.ticker as mtick
from sklearn.linear_model import LinearRegression

lookup_ma = {'1D': 1, '1M': 20, '1Y': 250, '3Y': 250 * 3, '5Y': 250 * 5, '10Y': 250 * 10}
lookup_factors = {'Market Factor (MER)': 'MER',
                  'Size Factor (SMB)': 'SMB',
                  'Value Factor (HML)': 'HML',
                  'Profitability Factor (RMW)': 'RMW',
                  'Investment Factor (CMA)': 'CMA'}
maxFigWidth = 22


def load_dataframes(file: str) -> Dict[str, pd.DataFrame]:
    from collections import defaultdict
    from io import StringIO
    with open(file) as reader:
        name = None
        data = defaultdict(list)
        for line in reader.readlines():
            if line.startswith("#data-begin#"):
                name = line[line.rfind("#") + 1:].rstrip('\n')
            elif line.startswith("#data-end#"):
                name = None
            elif name is not None:
                data[name].append(line.rstrip('\n'))
        data2 = dict()
        for k, v in data.items():
            p = pd.read_csv(StringIO('\n'.join(data[k])), index_col=0, parse_dates=[0], low_memory=False)
            p = p.replace(-99.99, math.nan).replace(999) / 100
            # if p.index.dtype=='int64': p.index = pd.DatetimeIndex([pd.Timestamp(str(i)+'01').date() for i in p.index])
            data2[k] = p
    return data2


def load_all_portfolios():
    portfolios = dict()
    i = 1
    for file in os.listdir('data-portfolios'):
        fileport = load_dataframes(os.path.join('data-portfolios', file))
        for k2, v2 in fileport.items():
            for k3, v3 in v2.items():
                if 'ignore' not in k2.lower() and 'value' in k2.lower():
                    portfolios['%i. ' % i + ' ➤ '.join([file[:-4], k2, k3])] = v3
                    i += 1
    for file in os.listdir('data-stocks'):
        x = pd.read_csv(os.path.join('data-stocks', file), index_col=0, parse_dates=[0], low_memory=False)['Close']
        x = (x - x.shift(1)) / x.shift(1)
        portfolios['%i. ' % i + file[:-4]] = x

    return portfolios


def coloriter(array, cmap_name, alpha=None) -> List:
    cm = plt.cm.get_cmap(cmap_name)
    o = list()
    for item, position in zip(array, np.linspace(0, 1, len(array))):
        c = cm(position)
        c = tuple([*c[0:3], alpha or 1.0])
        o.append((item, c))
    return o


def remove_outliers(series: pd.Series, zscore: float) -> pd.Series:
    return series[abs(sp.stats.zscore(series)) < zscore]


def rollma(data: pd.DataFrame, ma: str) -> pd.DataFrame:
    return data.rolling(window=lookup_ma[ma]).mean().dropna()


def factor_inverse(factor):
    return [k for k, v in lookup_factors.items() if v == factor][0]


def plot_factor_hists(ff: str, ma: str) -> None:
    f: str = lookup_factors[ff]
    periods = [(1960, 1980), (1980, 2007), (2007, 2020)]
    plt.figure(figsize=(9, 5))
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f bp'))
    for s, e in periods:
        df_factor: pd.DataFrame = df_factors[f]
        df = df_factor[str(s):str(e)]
        df = df.rolling(window=lookup_ma[ma]).mean().dropna()
        x = remove_outliers(df, 5) * 1e4
        sns.distplot(x, hist=True, rug=False, label="%s-%s" % (s, e), bins=30)
        # plt.hist(x=x, label="%s-%s"%(s,e), bins=30, alpha=0.5)
        plt.axvline(0.0, color='black', linestyle='dotted')
        plt.title(f"{ff}\n{ma} rolling average")
    plt.legend()


def plot_factor_timeseries(ff: str, ma: str) -> None:
    f: str = lookup_factors[ff]
    df = rollma(df_factors[f], ma)
    plt.figure(figsize=(9, 5))
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f bp'))
    color = coloriter(factors_list, 'tab10')[factors_list.index(f)][1]
    plt.plot(df * 1e4, color=color, linewidth=1)
    plt.axhline(0.0, color='black', linestyle='dotted')
    plt.title(f"{ff}\n{ma} rolling average")


def fit_model(factor_names: List[str], ret_actual: pd.DataFrame) -> Tuple[pd.Series, float]:
    X = df_factors[factor_names].reindex(ret_actual.index).dropna()
    Y = (ret_actual - df_factors.RF).dropna()
    reg = LinearRegression()
    reg.fit(X, Y)
    ret_predict = pd.Series(reg.predict(X), index=X.index) + df_factors.RF
    score = reg.score(X, Y)
    return ret_predict, score


def get_actual_returns(pname: str, daterange: Tuple[Any, Any]) -> pd.DataFrame:
    date0, date1 = str(daterange[0]), str(daterange[1])
    return portfolios[pname].loc[date0:date1].reindex(df_factors.index).dropna()


def fit_portfolio_returns(pname: str,
                          f_mer: bool,
                          f_smb: bool,
                          f_hml: bool,
                          f_rmw: bool,
                          f_cma: bool,
                          ma: str,
                          daterange: Tuple[Any, Any]):
    sel_factors = list()
    if f_mer: sel_factors.append('MER')
    if f_smb: sel_factors.append('SMB')
    if f_hml: sel_factors.append('HML')
    if f_rmw: sel_factors.append('RMW')
    if f_cma: sel_factors.append('CMA')
    if len(sel_factors) == 0:
        print("No factors selected")
        return
    ret_actual = get_actual_returns(pname, daterange)
    ret_predict, score = fit_model(sel_factors, ret_actual)
    ix = ret_actual.index.intersection(df_factors.index)
    plt.figure(figsize=(maxFigWidth, 4))
    plt.title("Actual vs. Predicted Returns ($R^2$ = %3.2f%%)" % float(100 * score))
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f bp'))
    plt.plot(rollma(ret_actual, ma) * 1e4, label='Actual return', color='black', linewidth=0.5)
    plt.plot(rollma(ret_predict, ma) * 1e4, label='Predicted return', color='red', linewidth=0.5)
    plt.axhline(0.0, color='black', linestyle='dotted')
    plt.legend(), plt.show()
    plt.figure(figsize=(maxFigWidth, 3))
    for iFactor, (factor, color) in enumerate(coloriter(factors_list, 'tab10')):
        if factor in sel_factors:
            plt.subplot(1, len(factors_list), iFactor + 1)
            X = df_factors[factor].loc[ix]
            Y = ret_actual.loc[ix]
            plt.axhline(0.0, color='black', linestyle='dotted')
            plt.axvline(0.0, color='black', linestyle='dotted')
            plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f bp'))
            plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f bp'))
            plt.scatter(X * 1e4, Y * 1e4, color=color, s=1), plt.title(
                f"Actual ret. vs. %s" % factor_inverse(factor))
    plt.subplots_adjust(wspace=0.3)
    plt.show()


@functools.lru_cache(maxsize=128, typed=False)
def estimate_R2_hist(pname, daterange):
    ret_actual = get_actual_returns(pname, daterange)
    _factors, _scores = [], []
    factor_scenarios = ['MER', 'SMB', 'HML', 'RMW', 'CMA', 'MER+SMB', 'MER+HML', 'MER+SMB+HML', 'MER+SMB+HML+RMW',
                        'MER+SMB+HML+CMA', 'MER+SMB+HML+RMW+CMA']
    for factor_names in factor_scenarios:
        factor_names = factor_names.split('+')
        X = df_factors[factor_names].reindex(ret_actual.index).dropna()
        Y = (ret_actual - df_factors.RF).dropna()
        reg = LinearRegression()
        reg.fit(X, Y)
        score = reg.score(X, Y)
        factor_names = "\n".join(factor_names)
        _factors.append(factor_names)
        _scores.append(score)
    return _factors, _scores


def draw_R2_hist(pname, daterange):
    _factors, _scores = estimate_R2_hist(pname, daterange)
    plt.figure(figsize=(8, 3))
    plt.title(f'Explanatory power ($R^2$) of factor combinations ({daterange[0]}-{daterange[1]})')
    plt.ylim(0, 1), plt.ylabel("$R^2$")
    plt.bar(_factors, _scores,
            color=5 * ['red'] + 2 * ['lightgreen'] + 1 * ['green'] + 2 * ['lightblue'] + 1 * ['blue'])
    plt.show()


@functools.lru_cache(maxsize=128, typed=False)
def estimate_R2_series(pname):
    years = list(range(1960, 2021))
    factor_scenarios = ['MER', 'SMB', 'HML', 'RMW', 'CMA', 'MER+SMB+HML', 'MER+SMB+HML+RMW+CMA']
    from collections import defaultdict
    XXX = list()
    YYY = defaultdict(list)
    for iYear, year in enumerate(years):
        daterange = str(year - 10), str(year)
        ret_actual = get_actual_returns(pname, daterange)
        if len(ret_actual) < 5:
            continue
        XXX.append(pd.Timestamp(daterange[1]))
        for factor_names in factor_scenarios:
            factor_names = factor_names.split('+')
            X = df_factors[factor_names].reindex(ret_actual.index).dropna()
            Y = (ret_actual - df_factors.RF).dropna()
            reg = LinearRegression()
            reg.fit(X, Y)
            score = reg.score(X, Y)
            factor_names = "+".join(factor_names)
            YYY[factor_names].append(score)
    return XXX, YYY


def draw_R2_series(pname, f_mer, f_smb, f_hml, f_rmw, f_cma, daterange):
    XXX, YYY = estimate_R2_series(pname)
    plt.figure(figsize=(8, 3))

    plt.gca().xaxis.grid(linestyle='dotted', color='black')
    plt.title(f'Explanatory power ($R^2$) timeline')
    sel_factors = list()
    if f_mer: sel_factors.append('MER')
    if f_smb: sel_factors.append('SMB')
    if f_hml: sel_factors.append('HML')
    if f_rmw: sel_factors.append('RMW')
    if f_cma: sel_factors.append('CMA')
    for f in sel_factors:
        plt.plot(XXX, YYY[f], label=factor_inverse(f))
    for f in 'MER+SMB+HML'.split(','):
        plt.plot(XXX, YYY[f], label='3 Factor Model', color='black')
    for f in 'MER+SMB+HML+RMW+CMA'.split(','):
        plt.plot(XXX, YYY[f], label='5 Factor Model', linestyle='--', color='black')
    # plt.xlim(str(daterange[0]), str(daterange[1]))
    plt.ylabel('$R^2$'), plt.legend(loc=(1.02, 0))
    plt.show()


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"timing '{self.name}' {int(self.interval * 1000)} ms")


###

portfolios = load_all_portfolios()
df_factors = load_dataframes('data-factors/F-F_Research_Data_5_Factors_2x3_daily.csv')['']
factors_list = [c for c in df_factors.columns if c != 'RF']
