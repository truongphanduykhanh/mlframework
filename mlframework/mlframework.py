'''
This script module contains end-to-end functions to use LightGBM model.
'''
__author__ = 'Khanh Truong'
__date__ = '2021-10-01'


import datetime
import itertools
from typing import Union, Tuple
import random

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def train_valid_test_split(
    data: pd.DataFrame,
    label_col: str,
    feat_cols: list,
    valid_size: Union[int, float] = 0.2,
    test_size: Union[int, float] = 0.2,
    label_time_col: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    '''
    Split data to train, validation and test sets

    Parameters
    ----------
    data : pandas.DataFrame
        Input data frame.
    valid_size : int or float
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
    if (valid_size < 1) & (test_size < 1):
        test = data.sample(frac=test_size, random_state=0)
        valid = data[~data.index.isin(test.index)].sample(frac=valid_size / (1 - test_size), random_state=0)
        train = data[~data.index.isin(list(test.index) + list(valid.index))]

    else:
        if label_time_col is None:
            raise ValueError('label_time_col must be specified.')
        label_time = data[label_time_col].unique()
        label_time = np.sort(label_time)

        label_time_test = label_time[-test_size:]
        label_time_valid = label_time[-(test_size + valid_size):-test_size]

        test = data[data[label_time_col].isin(list(label_time_test))]
        valid = data[data[label_time_col].isin(list(label_time_valid))]
        train = data[~data[label_time_col].isin(list(label_time_test) + list(label_time_valid))]

    X_train = train[feat_cols]
    X_valid = valid[feat_cols]
    X_test = test[feat_cols]

    y_train = train[label_col]
    y_valid = valid[label_col]
    y_test = test[label_col]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def convert_to_lgb_data(
    X_train: pd.DataFrame, X_valid: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_valid: pd.Series, y_test: pd.Series,
    categorical_col: Tuple[list, str] = 'auto'
) -> Tuple[lgb.Dataset, lgb.Dataset, lgb.Dataset]:
    '''
    Convert pandas data frame and series to LightGBM type

    Parameters
    ----------
    X_train, X_valid, X_test, y_train, y_valid, y_test : pandas.DataFrame and pandas.Series
        Train, validation and test data sets.

    Returns
    -------
    three data LightGBM type
        train, valid, test
    '''
    train = lgb.Dataset(data=X_train, label=y_train, categorical_feature=categorical_col)
    valid = lgb.Dataset(data=X_valid, label=y_valid, reference=train)
    test = lgb.Dataset(data=X_test, label=y_test, categorical_feature=categorical_col)
    return train, valid, test


def get_hours_minutes_seconds(timedelta: datetime.timedelta) -> Tuple[int, int, int]:
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


def convert_params_to_list_dict(params_dict: dict) -> list:
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


class LightGBM:

    def __init__(self):
        self.booster: lgb.Booster = None
        self.grid_search: pd.DataFrame = None

    def train(self, train: lgb.Dataset, params: dict, num_boost_round: int):
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
        '''
        booster = lgb.train(
            params=params,
            train=train,
            num_boost_round=num_boost_round,
            verbose_eval=False,
        )
        self.booster = booster

    def grid_search_lgb(
        self,
        train: lgb.Dataset,
        valid: lgb.Dataset,
        params_dict: dict,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 100
    ):
        '''
        Perform grid search through many sets of hyperparameters

        Parameters
        ----------
        train : LightGBM data
            Training data.
        valid : LightGBM data
            Validation data.
        params_dict : dict
            Dictionary of hyperparameters.
        num_boost_round : int
            Number of boosting iterations.
        early_stopping_rounds : int
            Activates early stopping. Validation metric needs to improve at least once in
            every **early_stopping_rounds** round(s) to continue training.
        '''
        params_list = convert_params_to_list_dict(params_dict)
        random.shuffle(params_list)  # shuffle to get better estimate of completing time
        print(f'There are {len(params_list)} hyperparameter sets.')
        global grid_search_results
        grid_search_results = []
        start = datetime.datetime.now()

        for i, params in enumerate(params_list):
            evals_result = {}
            lgb.train(
                params=params,
                train_set=train,
                num_boost_round=num_boost_round,
                valid_sets=[train, valid],
                valid_names=['train', 'valid'],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
                evals_result=evals_result
            )

            metric = list(evals_result['train'].keys())[-1]
            metric_train = evals_result['train'][metric]
            metric_valid = evals_result['valid'][metric]
            metric_gap = [x - y for x, y in zip(metric_train, metric_valid)]

            params['metric_valid_last'] = metric_valid[-1]
            params['metric_valid_max'] = max(metric_valid)
            params['metric_valid_max_index'] = metric_valid.index(params['metric_valid_max'])
            params['gap_at_valid_max'] = metric_gap[params['metric_valid_max_index']]

            # append loop result
            grid_search_results.append(params)

            # print log
            end_i = datetime.datetime.now()
            until_i = end_i - start
            est_total = (end_i - start) * len(params_list) / (i + 1)
            est_remain = est_total - until_i
            hours, minutes, seconds = get_hours_minutes_seconds(est_remain)
            print(f'Finishing {i+1:4}/{len(params_list)} \
                ---> Remaining {hours:02}:{minutes:02}:{seconds:02}')

        grid_search_results = pd.DataFrame(grid_search_results)

        best_params = (
            grid_search_results
            .sort_values('metric_valid_max', ascending=False)
            .drop([
                'metric_valid_last',
                'metric_valid_max',
                'metric_valid_max_index',
                'gap_at_valid_max'
            ], axis=1)
            .iloc[0]
            .to_dict()
        )
        booster = lgb.train(
            params=best_params,
            train_set=train,
            num_boost_round=num_boost_round,
            valid_sets=[train, valid],
            valid_names=['train', 'valid'],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
            evals_result=evals_result
        )
        self.grid_search_results = grid_search_results
        self.booster = booster
        print('Done')

    def get_feature_importance(
        self,
        n: int = 10,
        importance_type: str = 'gain'
    ) -> pd.Series:
        '''
        Get feature importance from trained booster

        Parameters
        ----------
        n : int
            Number of top features.
        importance_type : str
            'gain' or 'split'.

        Returns
        -------
        pandas series
            Feature importance.
        '''
        importance = pd.Series(
            data=self.booster.feature_importance(importance_type=importance_type),
            index=self.booster.feature_name(),
            name='importance'
        )
        importance = importance.sort_values(ascending=False).head(n)
        return importance

    def plot_feature_importance(
        self,
        n: int = 10,
        importance_type: str = 'gain',
        fmt: str = '%.0f',
        padding: int = 5
    ):
        '''
        Plot feature importance from trained booster

        Parameters
        ----------
        n : int
            Number of top features.
        importance_type : str
            'gain' or 'split'.
        fmt : str
            Format of printing values.
        padding : int
            Space between values and bars.
        '''
        importance = self.get_feature_importance(n, importance_type)
        ax = sns.barplot(x=importance, y=importance.index)
        plt.xlabel('')
        plt.title('Feature Importance', size=14)
        ax.bar_label(ax.containers[0], fmt=fmt, padding=padding)

    def predict(self, X: pd.DataFrame) -> np.array:
        '''
        Predict propensity score using lgb booster

        Parameters
        ----------
        X : pandas data frame
            Feaure data set.

        Returns
        -------
        numpy array
            propensity score.
        '''
        y = self.booster.predict(X)
        return y

    def save_model(self, file_path: str):
        '''
        Save lgb booster

        Parameters
        ----------
        file_path : str
            Location and file file name of the booster.
        '''
        self.booster.save_model(file_path)

    def load_model(self, file_path: str):
        '''
        Load lgb booster

        Parameters
        ----------
        file_path : str
            Location and file file name of the booster.
        '''
        self.booster = lgb.Booster(model_file=file_path)


def get_auc(y_true: list, y_pred: list) -> float:
    '''
    Get AUC score from true and prediction values

    Parameters
    ----------
    y_true : list
        List of true values.
    y_pred : list
        List of prediction values.

    Returns
    -------
    float
        AUC score.
    '''
    auc = metrics.roc_auc_score(y_true, y_pred)
    return auc


def get_f1_score(y_true: list, y_pred: list) -> float:
    '''
    Get F1 score from true and prediction values

    Parameters
    ----------
    y_true : list
        List of true values.
    y_pred : list
        List of prediction values.

    Returns
    -------
    float
        F1 score.
    '''
    f1 = metrics.f1_score(y_true, y_pred)
    return f1
