'''
This script module contains end-to-end functions to use LightGBM model.
'''
__author__ = 'Khanh Truong'
__date__ = '2021-10-01'


import datetime
import itertools
from typing import Union

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import metrics


def train_val_test_split(
    data: pd.DataFrame,
    label_col: str,
    feat_cols: list,
    val_size: Union[int, float] = 0.2,
    test_size: Union[int, float] = 0.2,
    label_time_col: str = None,
    *args, **kargs
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    '''
    Split data to train, validation and test sets

    Parameters
    ----------
    data : pandas.DataFrame
        Input data frame.
    val_size : int or float
        If int, number of ending periods of label_time_col to be in validation set.
        train -> validation -> test
        If float, proportion of validation set.
    test_size : int or float
        If int, number of ending periods of label_time_col to be in test set.
        train -> validation -> test
        If float, proportion of test set.
    label_time_col : str
        Label time column name. Need to specify this in case val_size and test_size are integers.
    label_col : str
        Label column name.
    feat_cols : list
        Features columns names.

    Returns
    -------
    three pandas.DataFrame and three pandas.Series
        Feature and label for train, validation and test sets.
    '''
    if (val_size < 1) & (test_size < 1):
        test = data.sample(frac=test_size, random_state=0)
        val = data[~data.index.isin(test.index)].sample(frac=val_size / (1 - test_size), random_state=0)
        train = data[~data.index.isin(list(test.index) + list(val.index))]

    else:
        if label_time_col is None:
            raise ValueError('label_time_col must be specified.')
        label_time = data[label_time_col].unique()
        label_time = np.sort(label_time)

        label_time_test = label_time[-test_size:]
        label_time_val = label_time[-(test_size + val_size):-test_size]

        test = data[data[label_time_col].isin(list(label_time_test))]
        val = data[data[label_time_col].isin(list(label_time_val))]
        train = data[~data[label_time_col].isin(list(label_time_test) + list(label_time_val))]

    X_train = train[feat_cols]
    X_val = val[feat_cols]
    X_test = test[feat_cols]

    y_train = train[label_col]
    y_val = val[label_col]
    y_test = test[label_col]

    return X_train, X_val, X_test, y_train, y_val, y_test


def convert_to_lgb_data(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_val: pd.Series, y_test: pd.Series,
    categorical_col: tuple[list[str], list[int], str] = 'auto'
) -> tuple[lgb.Dataset, lgb.Dataset, lgb.Dataset]:
    '''
    Convert pandas data frame and series to LightGBM type

    Parameters
    ----------
    X_train, X_val, X_test, y_train, y_val, y_test : pandas.DataFrame and pandas.Series
        Train, validation and test data sets.

    Returns
    -------
    three data LightGBM type
        train, val, test
    '''
    train = lgb.Dataset(data=X_train, label=y_train, categorical_feature=categorical_col)
    val = lgb.Dataset(data=X_val, label=y_val, reference=train)
    test = lgb.Dataset(data=X_test, label=y_test, categorical_feature=categorical_col)
    return train, val, test


def __get_hours_minutes_seconds(timedelta: datetime.timedelta) -> tuple[int, int, int]:
    '''
    Convert time delta to hours, minutes, seconds

    Parameters
    ----------
    timedelta : datetime.timedelta
        Time delta between two time points. Ex: datetime.timedelta(0, 9, 494935).

    Returns
    -------
    three integers
        Corresponding to number of hours, minutes and seconds.
    '''
    total_seconds = timedelta.seconds
    hours = total_seconds // 3600
    minutes = (total_seconds - (hours * 3600)) // 60
    seconds = total_seconds - (hours * 3600) - (minutes * 60)
    return hours, minutes, seconds


def __convert_params_to_list_dict(params_dict: dict) -> list[dict]:
    '''
    Convert dictionary to list of dictionaries

    Parameters
    ----------
    params_dict : dict
        Dictionary of parameters.

    Returns
    -------
    list of dict
        List of dictionaries, ready to grid search.
    '''
    keys, values = zip(*params_dict.items())
    params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return params_list


def grid_search_lgb(
    train: lgb.Dataset,
    val: lgb.Dataset,
    params_dict: dict,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 100
) -> pd.DataFrame:
    '''
    Perform grid search through many sets of hyperparameters

    Parameters
    ----------
    train : LightGBM data
        Training data.
    val : LightGBM data
        Validation data.
    params_dict : dict
        Dictionary of hyperparameters.
    num_boost_round : int
        Number of boosting iterations.
    early_stopping_rounds : int
        Activates early stopping. Validation metric needs to improve at least once in
        every **early_stopping_rounds** round(s) to continue training.

    Returns
    -------
    pandas.DataFrame
        results of grid search
    '''
    params_list = __convert_params_to_list_dict(params_dict)
    print(f'There are {len(params_list)} hyperparameter sets.')
    global grid_search
    grid_search = []
    start = datetime.datetime.now()

    for i, params in enumerate(params_list):
        evals_result = {}
        lgb.train(
            params=params,
            train_set=train,
            num_boost_round=num_boost_round,
            valid_sets=[train, val],
            valid_names=['train', 'val'],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
            evals_result=evals_result
        )

        metric = list(evals_result['train'].keys())[-1]
        metric_train = evals_result['train'][metric]
        metric_val = evals_result['val'][metric]
        metric_gap = [x - y for x, y in zip(metric_train, metric_val)]

        params['metric_val_last'] = metric_val[-1]
        params['metric_val_max'] = max(metric_val)
        params['metric_val_max_index'] = metric_val.index(params['metric_val_max'])
        params['gap_at_val_max'] = metric_gap[params['metric_val_max_index']]

        # append loop result
        grid_search.append(params)

        # print log
        end_i = datetime.datetime.now()
        until_i = end_i - start
        est_total = (end_i - start) * len(params_list) / (i + 1)
        est_remain = est_total - until_i
        hours, minutes, seconds = __get_hours_minutes_seconds(est_remain)
        print(f'Finishing {i+1:4}/{len(params_list)} \
            ---> Remaining {hours:02}:{minutes:02}:{seconds:02}')

    grid_search = pd.DataFrame(grid_search)
    print('Done')
    return grid_search


def train(train: lgb.Dataset, params: dict, num_boost_round: int) -> lgb.Booster:
    '''
    Train LightGBM model

    Parameters
    ----------
    train : LightGBM data
        Training data.
    params : dict
        Dictionary of hyperparameters.
    num_boost_round : int
        Number of bossting iterations.

    Returns
    -------
    Booster
        LightGBM model.
    '''
    booster = lgb.train(
        params=params,
        train_set=train,
        num_boost_round=num_boost_round,
        verbose_eval=False,
    )
    return booster


def get_best_model(
    train: lgb.Dataset,
    params_dict: dict,
    grid_search: pd.DataFrame,
    criteria: str = 'metric_val_max',
    higher_better: bool = True
) -> lgb.Booster:
    '''
    Select the best model from grid search

    Parameters
    ----------
    train : LightGBM data
        Training data.
    params_dict : dict
        Dictionary of hyperparameters.
    grid_search : pandas.DataFrame
        Results of grid search.
    criteria : str
        Metric name in result grid search to select best model.
    higher_better : bool
        If the metric is the higher the better.

    Returns
    -------
    Booster
        Best LightGBM model.
    '''
    best_model_index = grid_search.sort_values(criteria, ascending=1 - higher_better).head(1).index[0]
    best_model_params = __convert_params_to_list_dict(params_dict)[best_model_index]

    booster = lgb.train(
        params=best_model_params,
        train_set=train,
        num_boost_round=grid_search.loc[best_model_index]['metric_val_max_index'] + 1,
        verbose_eval=False
    )
    return booster


def get_auc(y_true, y_pred):
    auc = metrics.roc_auc_score(y_true, y_pred)
    return auc


def get_f1_score(y_true, y_pred):
    f1 = metrics.f1_score(y_true, y_pred)
    return f1


def save_model(booster, path):
    booster.save_model(path)
