"""Provides end-to-end flow to use LightGBM model.
"""
__author__ = 'khanhtpd'
__date__ = '2021-10-01'


import datetime
import itertools
from typing import Union, Tuple
import random

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score
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
    """Split data to train, validation and test sets.

    Args:
        data (pd.DataFrame):  Input data frame. Should include both label and features.
        label_col (str): Label column name.
        feat_cols (list): Features columns names.
        valid_size (Union[int, float], optional): If int, number of ending periods
            of label_time_col to be in validation set. Arrangement: train -> validation -> test.
            If float, proportion of validation set. Defaults to 0.2.
        test_size (Union[int, float], optional): If int, number of ending periods
            of label_time_col to be in test set. Arrangement: train -> validation -> test.
            If float, proportion of test set. Defaults to 0.2.
        label_time_col (str, optional): Label time column name. Need to specify this in case
            val_size and test_size are integers. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
            Features and label for train, validation and test sets.
    """
    if (valid_size < 1) & (test_size < 1):
        test = data.sample(frac=test_size, random_state=0)
        valid = (
            data[~data.index.isin(test.index)]
            .sample(frac=valid_size / (1 - test_size), random_state=0)
        )
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

    x_train = train[feat_cols]
    x_valid = valid[feat_cols]
    x_test = test[feat_cols]

    y_train = train[label_col]
    y_valid = valid[label_col]
    y_test = test[label_col]

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def convert_to_lgb_data(
    x_train: pd.DataFrame, x_valid: pd.DataFrame, x_test: pd.DataFrame,
    y_train: pd.Series, y_valid: pd.Series, y_test: pd.Series,
    categorical_col: Union[list, str] = 'auto'
) -> Tuple[lgb.Dataset, lgb.Dataset, lgb.Dataset]:
    """Convert pandas data frame and series to LightGBM type.

    Args:
        x_train (pd.DataFrame): Training features.
        x_valid (pd.DataFrame): Validation features.
        x_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training label.
        y_valid (pd.Series): Validation label.
        y_test (pd.Series): Test label.
        categorical_col (Union[list, str], optional): List of categorical features.
            Defaults to 'auto'.

    Returns:
        Tuple[lgb.Dataset, lgb.Dataset, lgb.Dataset]: train, valid and test lgb dataset
    """
    train = lgb.Dataset(data=x_train, label=y_train, categorical_feature=categorical_col)
    valid = lgb.Dataset(data=x_valid, label=y_valid, reference=train)
    test = lgb.Dataset(data=x_test, label=y_test, categorical_feature=categorical_col)
    return train, valid, test


def get_hours_minutes_seconds(timedelta: datetime.timedelta) -> Tuple[int, int, int]:
    """Convert time delta to hours, minutes, seconds.

    Args:
        timedelta (datetime.timedelta): Time delta between two time points.
            Ex: datetime.timedelta(0, 9, 494935).

    Returns:
        Tuple[int, int, int]: Corresponding to number of hours, minutes and seconds.
    """
    total_seconds = timedelta.seconds
    hours = total_seconds // 3600
    minutes = (total_seconds - (hours * 3600)) // 60
    seconds = total_seconds - (hours * 3600) - (minutes * 60)
    return hours, minutes, seconds


def convert_params_to_list_dict(params_dict: dict) -> list:
    """Convert dictionary to list of dictionaries.

    Args:
        params_dict (dict): Dictionary having items as lists.

    Returns:
        list: List of dictionary.
    """
    keys, values = zip(*params_dict.items())
    # convert single element to a list of single lement
    values = ([value] if type(value) is not list else value for value in values)
    params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return params_list


class LightGBM:
    """Train, evaluate and predict in LightGBM"""

    DEFAULT_PARAMS = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.1,
        'max_depth': 6,
        'max_leaves': 31,
        'subsample': 0.7,
        'sub_feature': 1.0
    }

    DEFAULT_PARAMS_DICT = {
        'objective': 'binary',
        'metric': 'auc',
        'verbose': -1,
        'learning_rate': 0.1,
        'max_depth': [4, 6, 10],
        'max_leaves': [15, 31, 63],
        'subsample': [0.5, 0.7, 0.9],
        'sub_feature': [0.5, 0.7, 0.9]
    }

    def __init__(self):
        self.grid_search_results: pd.DataFrame = None
        self.booster: lgb.Booster = None
        self.params: dict = None
        self.params_dict: dict = None

    def train(
        self,
        train: lgb.Dataset,
        valid: lgb.Dataset,
        params: dict = 'default',
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50
    ):
        """Train LightGBM model.

        Args:
            train (lgb.Dataset): Training data.
            valid (lgb.Dataset): Validation data.
            params (dict): Dictionary of hyperparameters. Defaults to 'default'.
            num_boost_round (int, optional): Number of boosting iterations. Defaults to 500.
            early_stopping_rounds (int, optional): Activates early stopping. Validation metric
                needs to improve at least once in every **early_stopping_rounds** round(s)
                to continue training. Defaults to 50.
        """
        if params == 'default':
            params = LightGBM.DEFAULT_PARAMS
            print(f'Training LightGBM with default params settings. \
                See more at attribute "params".')

        booster = lgb.train(
            params=params,
            train_set=train,
            num_boost_round=num_boost_round,
            valid_sets=[train, valid],
            valid_names=['train', 'valid'],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=50,
        )
        self.booster = booster
        self.params = params

    def grid_search(
        self,
        train: lgb.Dataset,
        valid: lgb.Dataset,
        params_dict: dict = 'default',
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50
    ):
        """Perform grid search through many combinations of hyperparameters.

        Args:
            train (lgb.Dataset): Training data.
            valid (lgb.Dataset): Validation data.
            params_dict (dict): Dictionary of hyperparameters. Defaults to 'default'.
            num_boost_round (int, optional): Number of boosting iterations. Defaults to 500.
            early_stopping_rounds (int, optional): Activates early stopping. Validation metric
                needs to improve at least once in every **early_stopping_rounds** round(s)
                to continue training. Defaults to 50.
        """
        if params_dict == 'default':
            params_dict = LightGBM.DEFAULT_PARAMS_DICT
            print('Training LightGBM with default params settings.')
            print('See more at attribute "DEFAULT_PARAMS_DICT".')

        params_list = convert_params_to_list_dict(params_dict)
        random.seed(0)
        random.shuffle(params_list)  # shuffle to get better estimate of completing time
        start = datetime.datetime.now()
        start_time = start.strftime('%H:%M:%S')
        print(f'There are {len(params_list)} hyperparameter combinations.')
        print(f'Starting turning at {start_time}...')

        self.grid_search_results = []
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
            self.grid_search_results.append(params)

            # print log
            end_i = datetime.datetime.now()
            until_i = end_i - start
            est_total = (end_i - start) * len(params_list) / (i + 1)
            est_remain = est_total - until_i
            hours, minutes, seconds = get_hours_minutes_seconds(est_remain)
            print(f'Finishing {i+1:4}/{len(params_list)} \
                ---> Remaining {hours:02}:{minutes:02}:{seconds:02}')

        self.grid_search_results = pd.DataFrame(self.grid_search_results)
        best_params = (
            self.grid_search_results
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
        self.booster = booster
        self.params = best_params
        self.params_dict = params_dict
        end_i = end_i.strftime('%H:%M:%S')
        hours, minutes, seconds = get_hours_minutes_seconds(until_i)
        print(f'Done at {end_i}')
        print(f'Duration {hours:02}:{minutes:02}:{seconds:02}')

    def plot_metric_sensitivity(self, nrows: Union[int, str] = 'auto', ncols: int = 2):
        """Plot metric sensitivity.

        Args:
            nrows (Union[int, str], optional): Number of rows in the plot.
                If 'auto', will be automatically calulated based on ncols.
            ncols (int, optional): Number of columns in the plot.
        """
        grid_short = self.grid_search_results.copy()

        for col in grid_short.columns:
            if grid_short[col].nunique() == 1:
                grid_short.drop(col, inplace=True, axis=1)

        if nrows == 'auto':
            nparams = len(grid_short.columns)
            nrows = round(nparams / ncols + 0.5)

        params = grid_short.columns
        params = [
            param for param in params if param not in
            ['metric_valid_last', 'metric_valid_max', 'metric_valid_max_index', 'gap_at_valid_max']
        ]

        plt.figure(figsize=(12, 2 * nrows))
        for i, param in enumerate(params):
            plt.subplot(nrows, ncols, i + 1)
            sns.boxenplot(x='metric_valid_max', y=param, data=grid_short, orient='h')
            plt.xlabel('')
            plt.subplots_adjust(hspace=0.5, wspace=0.5)
        metric = self.params['metric'].upper()
        plt.suptitle(f'{metric} Sensitivity', size=15, y=0.92)

    def save_model(self, file_path: str):
        """Save lgb booster.

        Args:
            file_path (str): Location and file name of the booster.
        """
        self.booster.save_model(file_path)

    def load_model(self, file_path: str):
        """Load lgb booster.

        Args:
            file_path (str): Location and file name of the booster.
        """
        self.booster = lgb.Booster(model_file=file_path)

    def get_feature_importance(
        self,
        ntop: int = 10,
        importance_type: str = 'gain'
    ) -> pd.Series:
        """Get top feature importance from trained booster.

        Args:
            n (int, optional): Number of top features. Defaults to 10.
            importance_type (str, optional): 'gain' or 'split'. Defaults to 'gain'.

        Returns:
            pd.Series: Top feature importance.
        """
        importance = pd.Series(
            data=self.booster.feature_importance(importance_type=importance_type),
            index=self.booster.feature_name(),
            name='importance'
        )
        importance = importance.sort_values(ascending=False).head(ntop)
        return importance

    def plot_feature_importance(
        self,
        ntop: int = 10,
        importance_type: str = 'gain',
        fmt: str = '%.0f',
        padding: int = 5
    ):
        """Plot feature importance from trained booster.

        Args:
            ntop (int, optional): Number of top features. Defaults to 10.
            importance_type (str, optional): 'gain' or 'split'. Defaults to 'gain'.
            fmt (str, optional): Format of printing values. Defaults to '%.0f'.
            padding (int, optional): Space between values and bars. Defaults to 5.
        """
        importance = self.get_feature_importance(ntop, importance_type)
        ax = sns.barplot(x=importance, y=importance.index)
        plt.xlabel('')
        plt.title('Feature Importance', size=14)
        ax.bar_label(ax.containers[0], fmt=fmt, padding=padding)

    def predict(self, x_data: pd.DataFrame) -> np.array:
        """Predict propensity score using lgb booster.

        Args:
            x_data (pd.DataFrame): Feature data set.

        Returns:
            np.array: Propensity score.
        """
        y_pred = self.booster.predict(x_data)
        return y_pred


def get_gain_table(y_true: list, y_pred: list, n_level: int = 10) -> pd.DataFrame:
    """Get gain table.

    Args:
        y_true (list): True label.
        y_pred (list): Prediction score.
        n_level (int, optional): Number of levels. Defaults to 10.

    Returns:
        pd.DataFrame: Gain table.
    """
    label_levels = reversed(range(1, 1 + n_level))
    label_levels = ['Level' + str(x) for x in label_levels]
    level = pd.qcut(y_pred, n_level, labels=label_levels, duplicates='raise')
    gain_df = pd.DataFrame({'true': y_true, 'predict': y_pred, 'level': level})
    gain_df = (
        gain_df
        .groupby('level')
        .agg(
            predict_min=('predict', 'min'),
            predict_mean=('predict', 'mean'),
            predict_max=('predict', 'max'),
            true_count=('true', 'count'),
            true_sum=('true', 'sum'),)
        .assign(true_mean=lambda df: df['true_sum'] / df['true_count'])
        .sort_index(ascending=False)
        .reset_index()
    )
    return gain_df


def plot_calibration_curve(y_true: list, y_pred: list, n_level: int = 10):
    """Plot calibration curve

    Args:
        y_true (list): True label.
        y_pred (list): Prediction score.
        n_level (int, optional): Number of levels. Defaults to 10.
    """
    gain_df = get_gain_table(y_true, y_pred, n_level)
    sns.lineplot(x='predict_mean', y='true_mean', data=gain_df, marker='o')
    max_predict_mean = max(gain_df['predict_mean'])
    max_true_mean = max(gain_df['true_mean'])
    lim_axis = max(max_predict_mean, max_true_mean)
    plt.plot([0.0, lim_axis], [0.0, lim_axis], linestyle='-.', color='grey')
    plt.title('Calibration Curve', size=14)


def get_auc(y_true: list, y_pred: list) -> float:
    """Get AUC score from true and prediction values.

    Args:
        y_true (list): True label.
        y_pred (list): Prediction score.

    Returns:
        float: AUC score.
    """
    auc = roc_auc_score(y_true, y_pred)
    return auc


def get_f1_score(y_true: list, y_pred: list, threshod: float = 0.5) -> float:
    """Get F1 score from true and prediction values.

    Args:
        y_true (list): True label.
        y_pred (list): Prediction score.
        threshod (float, optional): Convert values in y_pred to 0 if the values <= threshold,
            convert to 1 otherwise. Defaults to 0.5.

    Returns:
        float: F1 score.
    """
    y_pred_round = [0 if x < threshod else 1 for x in y_pred]
    f_one = f1_score(y_true, y_pred_round)
    return f_one
