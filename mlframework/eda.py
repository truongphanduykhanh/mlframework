"""Provides common functions when doing EDA.
"""
__author__ = 'khanhtruong'
__date__ = '2022-06-30'


import math
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


def count_missing_values(data: pd.DataFrame, figsize: tuple[int] = (12, 5)):
    """Count missing values according to columns and rows.

    Args:
        data (pd.DataFrame): Input data frame.
        figsize (tuple[int], optional): Size of figure. Defaults to (12, 4).
    """
    bins = np.linspace(0, 1, 11)
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    miss_col = data.isna().sum(axis=0) / data.shape[0]  # % missing values at every column
    miss_all_col = miss_col[miss_col == 1]  # note all-missing columns
    # cut missing percentage into intervals
    miss_col = pd.cut(miss_col, bins=bins, include_lowest=False, right=True)
    miss_col = miss_col.cat.add_categories([0, 1])  # add categories 0 and 1
    miss_col = miss_col.fillna(0)  # none missing column
    miss_col[miss_all_col.index] = 1  # all-missing column
    miss_col = (
        miss_col.replace(
            pd.Interval(0.9, 1.0, closed='right'),
            pd.Interval(0.9, 1.0, closed='neither')))  # replace the closed 1 by open 1.
    miss_col = (
        miss_col
        .value_counts()
        .sort_index())
    miss_col_idx = [0] + [x for x in miss_col.index if x != 0]  # move index 0 to first
    miss_col = miss_col[miss_col_idx]
    ax = miss_col.plot(kind='barh')
    total = miss_col.sum()
    for p in ax.patches:
        percentage = 100 * p.get_width() / total
        percentage = f'{percentage:.1f}%'
        x = p.get_x() + p.get_width()
        y = p.get_y() + 0.05
        ax.annotate(percentage, (x, y))
    plt.xlabel('Count')
    plt.ylabel('Percentage of missing')
    plt.title('Missing values - Columns', size=15)

    plt.subplots_adjust(wspace=0.5)
    plt.subplot(1, 2, 2)
    miss_row = data.isna().sum(axis=1) / data.shape[1]  # % missing values at every row
    miss_all_col = miss_row[miss_row == 1]  # note all-missing rows
    # cut missing percentage into intervals
    miss_row = pd.cut(miss_row, bins=bins, include_lowest=False, right=True)
    miss_row = miss_row.cat.add_categories([0, 1])  # add categories 0 and 1
    miss_row = miss_row.fillna(0)  # none missing row
    miss_row[miss_all_col.index] = 1  # all-missing row
    miss_row = (
        miss_row.replace(
            pd.Interval(0.9, 1.0, closed='right'),
            pd.Interval(0.9, 1.0, closed='neither')))  # replace the closed 1 by open 1.
    miss_row = (
        miss_row
        .value_counts()
        .sort_index())
    miss_row_idx = [0] + [x for x in miss_row.index if x != 0]  # move index 0 to first
    miss_row = miss_row[miss_row_idx]
    ax = miss_row.plot(kind='barh')
    total = miss_row.sum()
    for p in ax.patches:
        percentage = 100 * p.get_width() / total
        percentage = f'{percentage:.1f}%'
        x = p.get_x() + p.get_width()
        y = p.get_y() + 0.05
        ax.annotate(percentage, (x, y))
    plt.xlabel('Count')
    plt.ylabel('Percentage of missing')
    plt.title('Missing values - Rows', size=15)


def top_missing_columns(
    data: pd.DataFrame,
    ntop: Union[int, str] = 10,
    figsize: Union[tuple[int], str] = 'auto'
):
    """Count missing values in every columns.

    Args:
        data (pd.DataFrame): Input data frame.
        ntop (int, optional): Number of top missing columns displayed. 'all' means all columns.
            Defaults to 10.
        figsize (Union[tuple[int], str], optional): Size of figure. If 'auto', figsize =
            (8, ntop / 2). Defaults to 'auto'.
    """
    if ntop == 'all':
        ntop = data.shape[1]
    if figsize == 'auto':
        figsize = (8, ntop / 2)
    plt.figure(figsize=figsize)
    miss_count = data.isna().sum()
    miss_count = miss_count.sort_values(ascending=False)[0: ntop]
    miss_count = miss_count.sort_values(ascending=True)
    ax = miss_count.plot(kind='barh')
    for p in ax.patches:
        percentage = p.get_width() / len(data) * 100
        percentage = f'{percentage:.1f}%'
        x = p.get_x() + p.get_width()
        y = p.get_y() + 0.15
        ax.annotate(percentage, (x, y))
    plt.title('Top missing values columns', size=15)


def oversample(
    data: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Oversampling an imbalanced data to get perfect balanced data.

    Args:
        data (pd.DataFrame): Imbalanced input data.
        label (str): Label name in the data.

    Returns:
        (pd.DataFrame): Balanced data.
    """
    label_count = (  # count every label in the data set
        data
        .loc[:, label]
        .value_counts()
    )
    label_max = label_count.idxmax()  # most common label
    label_remain = [x for x in label_count.index if x != label_max]
    for label_i in label_remain:  # if there're more than one 'minor' label
        # oversample the 'minor' labels so that size of 'minor' labels equal to the most common one
        factor = int(label_count[label_max] / label_count[label_i])
        remainder = label_count[label_max] - factor * label_count[label_i]
        label_data = data.loc[lambda df: df[label] == label_i]
        over_sample_factor = pd.concat([label_data] * factor)
        over_sample_remainder = label_data.sample(remainder, replace=False, random_state=0)
        over_sample = pd.concat([
            data.loc[lambda df: df[label] == label_max],
            over_sample_factor, over_sample_remainder
        ], axis=0)
    return over_sample


def displot(
    data: pd.DataFrame,
    label: str = None,
    kind: str = 'box',
    nrows: Union[int, str] = 'auto',
    ncols: int = 2,
    figsize: Union[tuple[int], str] = 'auto',
    hspace: float = 0.7,
    wspace: float = 0.5
):
    """Distribution plot of numerical variables.

    Args:
        data (pd.DataFrame): Input data frame.
        label (str, optional): Label column in the data. Defaults to None.
        kind (str, optional): Kind of plot. 'box', 'kde' or 'hist'. Defaults to 'box'.
        nrows (Union[int, str], optional): Number of rows in the plot.
            If 'auto', will be automatically calulated based on ncols. Defaults to 'auto'.
        ncols (int, optional): Number of columns in the plot. Defaults to 2.
        figsize (Union[tuple[int], str], optional): Size of the whole plot. If 'auto',
            figsize = (12, 2 * nrows). Defaults to 'auto'.
        hspace (float, optional): Height space between sup plots. Defaults to 0.7.
        wspace (float, optional): Width space between sup plots. Defaults to 0.5.
    """
    # select columns which are numeric and not the label column
    cols = data.select_dtypes(include=np.number).columns
    cols = [col for col in cols if col != label]
    # nrows, ncols and size of figure
    if len(cols) == 1:
        ncols = 1
    if nrows == 'auto':
        nrows = math.ceil(len(cols) / ncols)
    if figsize == 'auto':
        figsize = (12, 2 * nrows)
    plt.figure(figsize=figsize)
    for i, col in enumerate(cols):
        if label is None:
            data_i = data[[col]]
        else:  # oversample the data according to the label
            data_i = data[[col, label]].dropna()
            data_i = oversample(data_i, label).reset_index()
        plt.subplot(nrows, ncols, i + 1)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        if kind == 'box':
            sns.boxplot(x=col, y=label, data=data_i, orient='h')
        if kind == 'kde':
            sns.kdeplot(x=col, hue=label, data=data_i)
        if kind == 'hist':
            sns.histplot(x=col, hue=label, data=data_i, stat='density')
        plt.xlabel('')
        plt.ylabel('')
        plt.title(col)


def countplot(
    data: pd.DataFrame,
    label: str = None,
    nclass: Union[int, str] = 5,
    nrows: Union[int, str] = 'auto',
    ncols: int = 2,
    figsize: Union[tuple[int], str] = 'auto',
    hspace: float = 0.7,
    wspace: float = 0.5
):
    """Count plot categorical variables.

    Args:
        data (pd.DataFrame): Input data frame.
        label (str, optional): Label column in the data. Defaults to None.
        nclass (Union[int, str], optional): Number of class displayed in the plot.
            If 'all', display all classes. Defaults to 5.
        nrows (Union[int, str], optional): Number of rows in the plot.
            If 'auto', will be automatically calulated based on ncols. Defaults to 'auto'.
        ncols (int, optional): Number of columns in the plot. Defaults to 2.
        figsize (Union[tuple[int], str], optional): Size of the whole plot. If 'auto',
            figsize = (12, 2 * nrows). Defaults to 'auto'.
        hspace (float, optional): Height space between sup plots. Defaults to 0.7.
        wspace (float, optional): Width space between sup plots. Defaults to 0.5.
    """
    cols = data.select_dtypes(exclude=np.number).columns
    if nclass == 'all':
        nclass = data[cols].nunique().max()
    if len(cols) == 1:
        ncols = 1
    if nrows == 'auto':
        nrows = math.ceil(len(cols) / ncols)
    if figsize == 'auto':
        figsize = (12, 3 * nrows)
    plt.figure(figsize=figsize)
    for i, col in enumerate(cols):
        plt.subplot(nrows, ncols, i + 1)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        # order of the barplot
        nclass_col = min(nclass, data[col].nunique())
        order = data[col].value_counts().iloc[0: nclass_col].index
        # config plot
        ax = sns.countplot(
            y=col,
            data=data,
            order=order,
            hue=label
        )
        # add percentage to the plot
        total = data.shape[0]
        count = data[col].value_counts()
        for i, p in enumerate(ax.patches):
            # if no label, percentage over all values
            if label is None:
                percentage = 100 * p.get_width() / total
                percentage = f'{percentage:.1f}%'
            # if label, percentage of label over each class
            else:
                count_index = i % nclass_col
                percentage = 100 * p.get_width() / count.iloc[count_index]
                percentage = f'{percentage:.1f}%'
            x = p.get_x() + p.get_width()
            # modify the position of percentage
            if label is None:
                y_adjust = 0.55
            else:
                y_adjust = 0.35
            y = p.get_y() + y_adjust
            ax.annotate(percentage, (x, y))
        # label and title
        plt.title(f'{col} ({data[col].nunique()})', size=12)
        plt.xlabel('')
        plt.ylabel('')


def correlation_matrix(data: pd.DataFrame, figsize: tuple = (7, 7)):
    """Plot correlation matrix.

    Args:
        data (pd.DataFrame): Input data frame.
        figsize (tuple, optional): Size of the figure. Defaults to (7, 7).
    """
    # Compute the correlation matrix
    corr = data.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation matrix', size=15)
