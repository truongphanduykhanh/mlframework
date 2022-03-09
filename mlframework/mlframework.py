"""Provides end-to-end flow of a machine learning model.
"""
__author__ = 'khanhtruong'
__date__ = '2022-03-01'

import random
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score


def time_folds_split(
    time_col: Union[str, pd.Series],
    data: pd.DataFrame = None,
    nfold: int = 5,
    valid_size: int = 1,
    gap: int = 2,
    verbose: bool = True
) -> list:
    """Split k-folds according to time series.

    Args:
        time_col (str, optional): Time column. Either column name in a data or a separated series.
        data (pd.DataFrame): Input data if time column is a string. Defaults to None.
        nfold (int, optional): Number of folds. Defaults to 5.
        valid_size (int, optional): Periods in valid set. Defaults to 1.
        gap (int, optional): Gap in periods between train and valid set. Defaults to 2.
        verbose (bool, optional): Show time folds. Defaults to True.

    Returns:
        list: [(train_index, valid_index), (train_index, valid_index), ...] where train_index and
            valid_index are list of indices.
    """
    if isinstance(time_col, str):
        time_col = data[time_col]
    # lightgbm (xgboost) folds apply on reset index (continuously starting from 0)
    time_col = time_col.reset_index(drop=True)
    time = list(np.sort(time_col.unique()))
    folds = []
    if verbose:
        print('---------> Folds ([train], [valid]) are as following:')
    for _ in range(nfold):
        valid_time = time[-valid_size:]
        train_time = time[: -valid_size - gap + 1]
        valid_index = time_col.loc[lambda x: x.isin(valid_time)].index.to_list()
        train_index = time_col.loc[lambda x: x.isin(train_time)].index.to_list()
        folds.append((train_index, valid_index))
        time = time[: -valid_size]  # remove last valid_size periods
        if verbose:
            print((train_time, valid_time))
    return folds


def get_data_folds(data: pd.DataFrame, folds: list) -> list:
    """Return data for according folds.

    Args:
        data (pd.DataFrame): Input data that is used to created folds.
        folds (list): Folds indices.

    Returns:
        list: [(train, valid), (train, valid), ...] where train and valid are data frame.
    """
    data = data.reset_index(drop=True)
    data_folds = []
    for index, _ in enumerate(folds):
        train = data.loc[folds[index][0]]
        valid = data.loc[folds[index][1]]
        data_folds.extend([(train, valid)])
    return data_folds


class DataPrep:

    def __init__(self, data: Union[pd.DataFrame, str] = None):
        if data is not None:
            # read from path
            if isinstance(data, str):
                data = pd.read_csv(data)
            # convert non-numeric columns to category
            string_cols = data.select_dtypes(exclude=np.number).columns
            data[string_cols] = data[string_cols].astype('category')
        # list of attributes
        self.data = data
        self.train_label: pd.Series = None
        self.train_feat: pd.DataFrame = None
        self.valid_label: pd.Series = None
        self.valid_feat: pd.DataFrame = None
        self.test_label: pd.Series = None
        self.test_feat: pd.DataFrame = None

    def register_train_valid_test(
        self,
        label_col: str,
        feat_cols: Union[list, str],
        train_data: pd.DataFrame = None,
        valid_data: pd.DataFrame = None,
        test_data: pd.DataFrame = None,
    ):
        """Register train, valid and test data.

        Args:
            label_col (str): Label column.
            feat_cols (Union[list, str]): Feature columns.
            train_data (pd.DataFrame, optional): Train data. Defaults to None.
            valid_data (pd.DataFrame, optional): Validation  data. Defaults to None.
            test_data (pd.DataFrame, optional): Test data. Defaults to None.
        """
        # concat to have full data
        self.data = pd.concat([train_data, valid_data, test_data], axis=0).reset_index(drop=True)
        # convert non-numeric columns to category
        string_cols = self.data.select_dtypes(exclude=np.number).columns
        self.data[string_cols] = self.data[string_cols].astype('category')

        self.train_label = train_data[label_col]
        self.train_feat = train_data[feat_cols]
        self.train_feat[string_cols] = self.train_feat[string_cols].astype('category')
        if valid_data is not None:
            self.valid_label = valid_data[label_col]
            self.valid_feat = valid_data[feat_cols]
            self.valid_feat[string_cols] = self.valid_feat[string_cols].astype('category')
        self.test_label = test_data[label_col]
        self.test_feat = test_data[feat_cols]
        self.test_feat[string_cols] = self.test_feat[string_cols].astype('category')
        print('New attributes: data, train_label, train_feat, valid_label, valid_feat,', end=' ')
        print('test_label, test_feat.')

    def train_valid_test_split(
        self,
        label_col: str,
        feat_cols: list,
        time_col: Union[str, pd.Series] = None,
        data: pd.DataFrame = None,
        valid_size: Union[int, float] = 0.2,
        test_size: Union[int, float] = 0.2,
    ):
        """Split data to train, validation and test sets.

        Args:
            label_col (str): Label column.
            feat_cols (list): Feature columns.
            time_col (Union[str, pd.Series], optional): Time column. Either column name in data or
                a separated series. Need to specify this if valid_size and test_size are integers.
                Defaults to None.
            data (pd.DataFrame, optional): Input data frame. Should include both label and features.
                If specified, this new data will overwrite the data assigned when creating instance.
                Defaults to None.
            valid_size (Union[int, float], optional): If int, number of ending periods of
                time_col to be in validation set. Order: train -> valid -> test.
                If float, proportion of validation set. Defaults to 0.2.
            test_size (Union[int, float], optional): If int, number of ending periods of
                time_col to be in test set. Order: train -> valid -> test.
                If float, proportion of test set. Defaults to 0.2.
        """
        # if data is provided, overwrite the data
        if data is not None:
            # read from path
            if isinstance(data, str):
                data = pd.read_csv(data)
            # convert non-numeric columns to category
            string_cols = data.select_dtypes(exclude=np.number).columns
            data[string_cols] = data[string_cols].astype('category')
            self.data = data

        # if time_col is provided
        if time_col is not None:
            if isinstance(time_col, str):  # if time_col is string (part of the data)
                time_col = self.data[time_col]
            if not all(time_col.index == self.data.index):
                raise ValueError('time_col indices are not the same as data indices.')
            self.time_col = time_col

        # if sizes are proportion
        if (valid_size < 1) & (test_size < 1):
            test = self.data.sample(frac=test_size, random_state=0)
            valid = (
                self.data[~self.data.index.isin(test.index)]
                .sample(frac=valid_size / (1 - test_size), random_state=0)
            )
            train = self.data[~self.data.index.isin(list(test.index) + list(valid.index))]
            if valid_size == 0:
                print('test is randomly sampled.')
            else:
                print('valid and test are randomly sampled.')

        # if test size is number of ending periods and valid size is proportion
        elif (valid_size < 1) & (test_size >= 1):
            if time_col is None:
                raise ValueError('time_col must be specified.')
            time = list(np.sort(time_col.unique()))
            test_time = time[-test_size:]
            test_index = time_col.loc[lambda x: x.isin(test_time)].index.to_list()
            test = self.data.loc[test_index]
            valid = (
                self.data
                .drop(test_index, axis=0)
                .sample(frac=valid_size, random_state=0)
            )
            train = self.data[~self.data.index.isin(list(test.index) + list(valid.index))]
            if valid_size == 0:
                print('test is out-of-time sampled.')
            else:
                print('valid is randomly sampled, test is out-of-time sampled.')

        # if sizes are number of ending periods
        elif (valid_size >= 1) & (test_size >= 1):
            if time_col is None:
                raise ValueError('time_col must be specified.')
            time = list(np.sort(time_col.unique()))
            test_time = time[-test_size:]
            valid_time = time[-(test_size + valid_size): -test_size]
            test_index = time_col.loc[lambda x: x.isin(test_time)].index.to_list()
            valid_index = time_col.loc[lambda x: x.isin(valid_time)].index.to_list()
            test = self.data.loc[test_index]
            valid = self.data.loc[valid_index]
            train = self.data.drop(valid_index + test_index, axis=0)
            if valid_size == 0:
                print('test is out-of-time sampled.')
            else:
                print('valid and test are out-of-time sampled.')
        else:
            raise ValueError('Invalid valid_size and test_size.')

        self.train_label = train[label_col]
        self.train_feat = train[feat_cols]
        self.valid_label = valid[label_col]
        self.valid_feat = valid[feat_cols]
        self.test_label = test[label_col]
        self.test_feat = test[feat_cols]
        print('New attributes: data, train_label, train_feat, valid_label, valid_feat,', end=' ')
        print('test_label, test_feat.')

    @staticmethod
    def _update_params(trial: optuna.Trial, DEFAULTED_PARAMS: dict, params: dict):
        """Update defaulted parameters.
        """
        # update params
        if params:
            DEFAULTED_PARAMS.update(params)
        for key in DEFAULTED_PARAMS:
            # if not tune the param
            if not isinstance(DEFAULTED_PARAMS[key], list):  # input is not a list
                DEFAULTED_PARAMS[key] = DEFAULTED_PARAMS[key]
            elif len(DEFAULTED_PARAMS[key]) == 1:  # input is a list with only one element
                DEFAULTED_PARAMS[key] = DEFAULTED_PARAMS[key]
            # if tune the param
            elif all([isinstance(x, int) for x in DEFAULTED_PARAMS[key]]):  # integers
                DEFAULTED_PARAMS[key] = trial.suggest_int(
                    key,
                    DEFAULTED_PARAMS[key][0],
                    DEFAULTED_PARAMS[key][1]
                )
            elif 'log' in DEFAULTED_PARAMS[key]:  # float log
                DEFAULTED_PARAMS[key] = trial.suggest_float(
                    key,
                    DEFAULTED_PARAMS[key][0],
                    DEFAULTED_PARAMS[key][1],
                    log=True
                )
            elif any([isinstance(x, float) for x in DEFAULTED_PARAMS[key]]):  # float
                DEFAULTED_PARAMS[key] = trial.suggest_float(
                    key,
                    DEFAULTED_PARAMS[key][0],
                    DEFAULTED_PARAMS[key][1]
                )
            elif all([isinstance(x, str) for x in DEFAULTED_PARAMS[key]]):  # categorical
                DEFAULTED_PARAMS[key] = trial.suggest_categorical(
                    key,
                    DEFAULTED_PARAMS[key]
                )
            else:
                raise ValueError('Search space is not supported.')

        # show all fixed and tuning params
        if trial.number == 0:
            params_fixed = [
                param for param in DEFAULTED_PARAMS.keys() if param not in trial.distributions.keys()
            ]
            print('---------> Fixed params:')
            for param in params_fixed:
                print(f"'{param}': {DEFAULTED_PARAMS[param]}")
            print('---------> Tuning params:')
            for param in trial.distributions:
                print(f"'{param}': {trial.distributions[param]}")
            print()


class LightGBM(DataPrep):

    def __init__(self, data: Union[pd.DataFrame, str] = None):
        super().__init__(data)
        self.study: optuna.study = None
        self.best_params: dict = None
        self.best_num_boost_round: int = None
        self.booster: lgb.Booster = None
        self.cvbooster: lgb.CVBooster = None

    def train(
        self,
        params: dict = None,
        num_boost_round: int = 500
    ):
        """Train lightgbm model.

        Args:
            params (dict, optional): Hyperparameters. If None, default hyperparameters will be set.
                Defaults to None.
            num_boost_round (int, optional): Number of boosting round. Defaults to 500.
        """
        train_data = lgb.Dataset(data=self.train_feat, label=self.train_label)
        DEFAULTED_PARAMS = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.1,
            'verbose': -1,
            'early_stopping_rounds': 50
        }
        if params:
            DEFAULTED_PARAMS.update(params)
        print(f'Booster is trained with parameters: {DEFAULTED_PARAMS}.')
        if len(self.valid_feat) == 0:
            print('No validation set. Strongly recommend to spare one to prevent overfitting.', end=' ')
            print('Or use method tune() with cross-validation instead.')
            booster = lgb.train(
                params=DEFAULTED_PARAMS,
                train_set=train_data,
                num_boost_round=num_boost_round,
                valid_sets=train_data,
                valid_names='train'
            )
        else:
            valid_data = lgb.Dataset(data=self.valid_feat, label=self.valid_label, reference=train_data)
            booster = lgb.train(
                params=DEFAULTED_PARAMS,
                train_set=train_data,
                num_boost_round=num_boost_round,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid']
            )
        self.booster = booster

    @staticmethod
    def _objective(
        trial,
        cv: bool = True,
        params: dict = None,
        train_label: pd.DataFrame = None,
        train_feat: pd.DataFrame = None,
        valid_label: pd.DataFrame = None,
        valid_feat: pd.DataFrame = None,
        folds: list = None,
        nfold: int = 5,
        num_boost_round: int = 500
    ) -> float:
        # default params search space
        DEFAULTED_PARAMS = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.1,
            'verbose': -1,
            'early_stopping_rounds': 50,
            'num_leaves': [15, 1023],
            'max_depth': [3, 12],
            'min_data_in_leaf': [10, 100],
            'subsample': [0.3, 1.0],
            'colsample_bytree': [0.3, 1.0],
            'reg_alpha': [1e-8, 10.0, 'log'],
            'reg_lambda': [1e-8, 10.0, 'log']
        }
        # update defaulted by inputed params
        DataPrep._update_params(trial, DEFAULTED_PARAMS, params)
        # if use cross validation
        if cv:
            # if there is a valid, merge valid to train before cross validation
            if len(valid_feat) > 0:
                train_feat = pd.concat([train_feat, valid_feat], axis=0)
                train_label = pd.concat([train_label, valid_label], axis=0)
            train_data = lgb.Dataset(data=train_feat, label=train_label)
            cvbooster = lgb.cv(
                params=DEFAULTED_PARAMS,
                train_set=train_data,
                num_boost_round=num_boost_round,
                folds=folds,
                nfold=nfold,
                return_cvbooster=True
            )
            metric_name = list(cvbooster)[0]  # the first metric name (if many)
            metric_values = cvbooster[metric_name]  # the first metric's values
            metric_value = metric_values[-1]  # the first metric's best value
            best_num_boost_round = len(metric_values)
        # if not use cross validation
        else:
            train_data = lgb.Dataset(data=train_feat, label=train_label)
            valid_data = lgb.Dataset(data=valid_feat, label=valid_label, reference=train_data)
            booster = lgb.train(
                params=DEFAULTED_PARAMS,
                train_set=train_data,
                num_boost_round=num_boost_round,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid']
            )
            metric_name = list(booster.best_score['valid'])[0]  # the first metric name (if many)
            metric_value = booster.best_score['valid'][metric_name]  # the first metric's best value
            best_num_boost_round = booster.current_iteration()
        trial.set_user_attr(key='best_num_boost_round', value=best_num_boost_round)
        return metric_value

    @staticmethod
    def _callback(study: optuna.study, trial: optuna.Trial):
        if study.best_trial.number == trial.number:
            study.set_user_attr(
                key='best_num_boost_round',
                value=trial.user_attrs['best_num_boost_round']
            )

    def tune(
        self,
        n_trials: int = 100,
        cv: bool = True,
        cv_method: str = 'random',
        params: dict = None,
        nfold: int = 5,
        valid_size: int = 1,
        gap: int = 2,
        num_boost_round: int = 500,
        direction: str = 'maximize'
    ):
        """Hyperparameter tunning lightgbm with optuna.

        Args:
            n_trials (int, optional): Number of tunning trials. Defaults to 100.
            cv (bool, optional): If use k-folds cross validation. Defaults to True.
            cv_method (str, optional): Method of cross validation. Must be either 'oot' or 'random'.
                Ignore when cv=False. Defaults to 'random'.
                Example of 3-fold ([train], [valid]) when cv_method='oot':
                ([202006, 202007, 202008, 202009, 202010], [202012])
                ([202006, 202007, 202008, 202009], [202011])
                ([202006, 202007, 202008], [202010])
            params (dict, optional): Hyperparameter search space. If None, pre-determined search
                space applies. Defaults to None. Example of params search space:
                {
                    'learning_rate': 0.1,
                    'max_depth': [3, 12],
                    'subsample': [0.3, 1.0],
                    'reg_alpha': [1e-8, 10.0, 'log'],
                    'boosting': ['gbdt', 'dart']
                }
            nfold (int, optional): Number of folds if cross validating. Defaults to 5.
            valid_size (int, optional): Number of ending periods in valid set when cv_method='oot'.
                Ignore if cv_method='random'. Defaults to 1.
            gap (int, optional): Gap in periods between the last period in train and the first in
                valid when cv_method='oot'. Example: 'gap=2' means use data upto now to predict
                label in 2 periods. Ignore if cv_method='random'. Defaults to 2.
            num_boost_round (int, optional): Number of boosting round. Defaults to 500.
            direction (str, optional): Direction to optimize the metric. Defaults to 'maximize'.

        Raises:
            ValueError: Must spare a valid set when spliting or registering data.
            ValueError: cv_method must be either 'oot' or 'random'.
        """
        if not cv and len(self.valid_feat) == 0:
            raise ValueError('Must spare a valid set when spliting or registering data.')
        if cv_method not in ['oot', 'random']:
            raise ValueError("cv_method must be either 'oot' or 'random'.")
        if cv and len(self.valid_feat) > 0:
            print(f'---------> {cv_method.capitalize()} cross validation on whole (train + valid).')
        folds = None
        if cv and cv_method == 'oot':
            time_index = pd.concat([self.train_label, self.valid_label], axis=0).index
            folds = time_folds_split(
                time_col=self.time_col.loc[time_index],
                nfold=nfold,
                valid_size=valid_size,
                gap=gap,
                verbose=1
            )
        self.folds = folds
        sampler = optuna.samplers.TPESampler(seed=0)
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(
            lambda trial: LightGBM._objective(
                trial,
                cv=cv,
                params=params,
                train_label=self.train_label,
                train_feat=self.train_feat,
                valid_label=self.valid_label,
                valid_feat=self.valid_feat,
                folds=folds,
                nfold=nfold,
                num_boost_round=num_boost_round
            ),
            n_trials=n_trials,
            callbacks=[LightGBM._callback]
        )
        self.study = study
        DEFAULTED_PARAMS = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.1,
            'verbose': -1,
            'early_stopping_rounds': 50
        }
        if params:
            DEFAULTED_PARAMS.update(params)
        DEFAULTED_PARAMS.update(study.best_params)
        self.best_params = DEFAULTED_PARAMS
        self.best_num_boost_round = self.study.user_attrs['best_num_boost_round']
        # merge valid (if any) to train before re-train
        if len(self.valid_feat) > 0:
            # if there is a valid set, merge it with train set
            train_feat = pd.concat([self.train_feat, self.valid_feat], axis=0)
            train_label = pd.concat([self.train_label, self.valid_label], axis=0)
            train_data = lgb.Dataset(data=train_feat, label=train_label)
        else:
            train_data = lgb.Dataset(data=self.train_feat, label=self.train_label)
        # re-train cv with best_params and best_num_boost_round
        self.cvbooster = None  # re-assign to None
        # If not do this, tune(cv=True) then tune(cv=False) remains old attribute cvbooster
        if cv:
            cvbooster = lgb.cv(
                params=self.best_params,
                train_set=train_data,
                num_boost_round=self.best_num_boost_round,
                folds=folds,
                nfold=nfold,
                return_cvbooster=True
            )
            self.cvbooster = cvbooster['cvbooster']
        # re-train with best_params and best_num_boost_round
        retrain_best_params = self.best_params.copy()
        del retrain_best_params['early_stopping_rounds']  # delete early_stopping_rounds
        booster = lgb.train(
            params=retrain_best_params,
            train_set=train_data,
            num_boost_round=self.best_num_boost_round
        )
        self.booster = booster

    def predict(self, data: Union[str, pd.DataFrame], cvbooster: bool = True) -> list:
        """Predict on new data.

        Args:
            data (Union[str, pd.DataFrame]): Data to be predicted. 'train' for train data, 'valid'
                for validation data, 'test' for test data.
            cvbooster (bool, optional): Use cvbooster if available. If cvbooster=True, predictions
                are average of scores from all boosters from cross validation. Defaults to True.

        Returns:
            list: Prediction score.
        """
        if not isinstance(data, pd.DataFrame):
            if data == 'train':
                data = self.train_feat
            elif data == 'valid':
                data = self.valid_feat
            elif data == 'test':
                data = self.test_feat
        if cvbooster and self.cvbooster:
            pred = self.cvbooster.predict(data)
            pred = np.mean(pred, axis=0)
        else:
            pred = self.booster.predict(data)
        return pred


class XGBoost(DataPrep):

    def __init__(self, data: Union[pd.DataFrame, str] = None):
        super().__init__(data)
        self.study: optuna.study = None
        self.best_params: dict = None
        self.best_num_boost_round: int = None
        self.booster: xgb.Booster = None

    def train(
        self,
        params: dict = None,
        num_boost_round: int = 500,
        verbose_eval: int = 10,
        early_stopping_rounds: int = 50
    ):
        """Train xgboost model.

        Args:
            params (dict, optional): Hyperparameters. If None, default hyperparameters will be set.
                Defaults to None.
            num_boost_round (int, optional): Number of boosting round. Defaults to 500.
            verbose_eval (int, optional): Evaluation metric is printed every verbose_eval rounds.
                Defaults to 10.
            early_stopping_rounds (int, optional): Will stop training if metric in validation data
                doesn't improve in last early_stopping_round. Only applied if there is valid set
                when splitting or registering data. Defaults to 50.
        """
        train_data = xgb.DMatrix(data=self.train_feat, label=self.train_label)
        DEFAULTED_PARAMS = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.1,
            'verbosity': 1
        }
        if params:
            DEFAULTED_PARAMS.update(params)
        print(f'Booster is trained with parameters: {DEFAULTED_PARAMS}.')
        if len(self.valid_feat) == 0:
            print('No validation set. Strongly recommend to spare one to prevent overfitting.', end=' ')
            print('Or use method tune() with cross-validation instead.')
            booster = xgb.train(
                params=DEFAULTED_PARAMS,
                dtrain=train_data,
                num_boost_round=num_boost_round,
                evals=[(train_data, 'train')],
                verbose_eval=verbose_eval
            )
        else:
            valid_data = xgb.DMatrix(data=self.valid_feat, label=self.valid_label)
            booster = xgb.train(
                params=DEFAULTED_PARAMS,
                dtrain=train_data,
                num_boost_round=num_boost_round,
                evals=[(train_data, 'train'), (valid_data, 'valid')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval
            )
        self.booster = booster

    @staticmethod
    def _objective(
        trial,
        cv: bool = True,
        params: dict = None,
        train_label: pd.DataFrame = None,
        train_feat: pd.DataFrame = None,
        valid_label: pd.DataFrame = None,
        valid_feat: pd.DataFrame = None,
        folds: list = None,
        nfold: int = 5,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50
    ) -> float:
        # default params search space
        DEFAULTED_PARAMS = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.1,
            'verbosity': 0,
            # 'num_leaves': [15, 1023],
            'max_depth': [3, 12],
            # 'min_data_in_leaf': [10, 100],
            'subsample': [0.3, 1.0],
            'colsample_bytree': [0.3, 1.0],
            'reg_alpha': [1e-8, 10.0, 'log'],
            'reg_lambda': [1e-8, 10.0, 'log']
        }
        # update defaulted by inputed params
        DataPrep._update_params(trial, DEFAULTED_PARAMS, params)
        # if use cross validation
        if cv:
            # if there is a valid, merge valid to train before cross validation
            if len(valid_feat) > 0:
                train_feat = pd.concat([train_feat, valid_feat], axis=0)
                train_label = pd.concat([train_label, valid_label], axis=0)
            train_data = xgb.DMatrix(data=train_feat, label=train_label)
            cvbooster = xgb.cv(
                params=DEFAULTED_PARAMS,
                dtrain=train_data,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
                folds=folds,
                nfold=nfold
            )
            metric_values = cvbooster.iloc[:, -2]  # the last metric's values (if many metrics)
            metric_value = max(metric_values)  # the last metric's best value
            best_num_boost_round = metric_values.idxmax() + 1
        # if not use cross validation
        else:
            train_data = xgb.DMatrix(data=train_feat, label=train_label)
            valid_data = xgb.DMatrix(data=valid_feat, label=valid_label)
            booster = xgb.train(
                params=DEFAULTED_PARAMS,
                dtrain=train_data,
                num_boost_round=num_boost_round,
                evals=[(train_data, 'train'), (valid_data, 'valid')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False
            )
            metric_value = booster.best_score  # the last metric's best value (if many metrics)
            best_num_boost_round = booster.best_iteration + 1
        trial.set_user_attr(key='best_num_boost_round', value=best_num_boost_round)
        return metric_value

    @staticmethod
    def _callback(study: optuna.study, trial: optuna.Trial):
        if study.best_trial.number == trial.number:
            study.set_user_attr(
                key='best_num_boost_round',
                value=trial.user_attrs['best_num_boost_round']
            )

    def tune(
        self,
        n_trials: int = 100,
        cv: bool = True,
        cv_method: str = 'random',
        params: dict = None,
        nfold: int = 5,
        valid_size: int = 1,
        gap: int = 2,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
        direction: str = 'maximize'
    ):
        """Hyperparameter tunning xgboost with optuna.

        Args:
            n_trials (int, optional): Number of tunning trials. Defaults to 100.
            cv (bool, optional): If use k-folds cross validation. Defaults to True.
            cv_method (str, optional): Method of cross validation. Must be either 'oot' or 'random'.
                Ignore when cv=False. Defaults to 'random'.
                Example of 3-fold ([train], [valid]) when cv_method='oot':
                ([202006, 202007, 202008, 202009, 202010], [202012])
                ([202006, 202007, 202008, 202009], [202011])
                ([202006, 202007, 202008], [202010])
            params (dict, optional): Hyperparameter search space. If None, pre-determined search
                space applies. Defaults to None. Example of params search space:
                {
                    'learning_rate': 0.1,
                    'max_depth': [3, 12],
                    'subsample': [0.3, 1.0],
                    'reg_alpha': [1e-8, 10.0, 'log']
                }
            nfold (int, optional): Number of folds if cross validating. Defaults to 5.
            valid_size (int, optional): Number of ending periods in valid set when cv_method='oot'.
                Ignore if cv_method='random'. Defaults to 1.
            gap (int, optional): Gap in periods between the last period in train and the first in
                valid when cv_method='oot'. Example: 'gap=2' means use data upto now to predict
                label in 2 periods. Ignore if cv_method='random'. Defaults to 2.
            num_boost_round (int, optional): Number of boosting round. Defaults to 500.
            early_stopping_rounds (int, optional): Will stop training if (any) metric in validation
                data doesn't improve in last early_stopping_round. Set first_metric_only=True in
                params if want to use only the first metric for early stopping. Defaults to 50.
            direction (str, optional): Direction to optimize the metric. Defaults to 'maximize'.

        Raises:
            ValueError: Must spare a valid set when spliting or registering data.
            ValueError: cv_method must be either 'oot' or 'random'.
        """
        if not cv and len(self.valid_feat) == 0:
            raise ValueError('Must spare a valid set when spliting or registering data.')
        if cv_method not in ['oot', 'random']:
            raise ValueError("'cv_method' must be either 'oot' or 'random'.")
        if cv and len(self.valid_feat) > 0:
            print(f'---------> {cv_method.capitalize()} cross validation on whole (train + valid).')
        folds = None
        if cv and cv_method == 'oot':
            time_index = pd.concat([self.train_label, self.valid_label], axis=0).index
            folds = time_folds_split(
                time_col=self.time_col.loc[time_index],
                nfold=nfold,
                valid_size=valid_size,
                gap=gap,
                verbose=1
            )
        self.folds = folds
        sampler = optuna.samplers.TPESampler(seed=0)
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(
            lambda trial: XGBoost._objective(
                trial,
                cv=cv,
                params=params,
                train_label=self.train_label,
                train_feat=self.train_feat,
                valid_label=self.valid_label,
                valid_feat=self.valid_feat,
                folds=folds,
                nfold=nfold,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds
            ),
            n_trials=n_trials,
            callbacks=[XGBoost._callback]
        )
        self.study = study
        DEFAULTED_PARAMS = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.1,
            'verbosity': 0,
        }
        if params:
            DEFAULTED_PARAMS.update(params)
        DEFAULTED_PARAMS.update(study.best_params)
        self.best_params = DEFAULTED_PARAMS
        self.best_num_boost_round = self.study.user_attrs['best_num_boost_round']
        # merge valid (if any) to train before re-train
        if len(self.valid_feat) > 0:
            # if there is a valid set, merge it with train set
            train_feat = pd.concat([self.train_feat, self.valid_feat], axis=0)
            train_label = pd.concat([self.train_label, self.valid_label], axis=0)
            train_data = xgb.DMatrix(data=train_feat, label=train_label)
        else:
            train_data = xgb.DMatrix(data=self.train_feat, label=self.train_label)
        # re-assign cvbooster to None
        # If not do this, tune(cv=True) then tune(cv=False) remains old attribute cvbooster
        self.cvbooster = None
        # re-train cv with best_params and best_num_boost_round
        # the re-train cv
        if cv:
            cvbooster = []
            if cv_method == 'oot':  # replicate the oot split (100% the same as xgb.cv())
                for i in range(nfold):
                    train_fold = train_data.slice(self.folds[i][0])
                    booster = xgb.train(
                        params=self.best_params,
                        dtrain=train_fold,
                        num_boost_round=self.best_num_boost_round
                    )
                    cvbooster.append(booster)
            elif cv_method == 'random':  # replicate the random split (not 100% the same as xgb.cv())
                indices = list(range(train_data.num_row()))
                random.seed(0)
                random.shuffle(indices)
                size = len(indices) / nfold
                for i in range(nfold):
                    fold_indices = range(int(size * i), int(size * (i + 1)))
                    fold_indices = list(fold_indices)
                    train_fold = train_data.slice(fold_indices)
                    booster = xgb.train(
                        params=self.best_params,
                        dtrain=train_fold,
                        num_boost_round=self.best_num_boost_round
                    )
                    cvbooster.append(booster)
            self.cvbooster = cvbooster
        # re-train with best_params and best_num_boost_round
        booster = xgb.train(
            params=self.best_params,
            dtrain=train_data,
            num_boost_round=self.best_num_boost_round
        )
        self.booster = booster

    def predict(self, data: Union[str, pd.DataFrame], cvbooster: bool = True) -> list:
        """Predict on new data.

        Args:
            data (Union[str, pd.DataFrame]): Data to be predicted. 'train' for train data, 'valid'
                for validation data, 'test' for test data.
            cvbooster (bool, optional): Use cvbooster if available. If cvbooster=True, predictions
                are average of scores from all boosters from cross validation. Defaults to True.

        Returns:
            list: Prediction score.
        """
        if not isinstance(data, pd.DataFrame):
            if data == 'train':
                data = xgb.DMatrix(self.train_feat)
            elif data == 'valid':
                data = xgb.DMatrix(self.valid_feat)
            elif data == 'test':
                data = xgb.DMatrix(self.test_feat)
        if cvbooster and self.cvbooster:
            pred = [booster.predict(data) for booster in self.cvbooster]
            pred = np.mean(pred, axis=0)
        else:
            pred = self.booster.predict(data)
        return pred


class MLFramework(LightGBM, XGBoost):

    def __init__(self, data: Union[pd.DataFrame, str] = None):
        super().__init__(data)
        self.algo = None
        self.calibrated_booster = None
        self.calibrated_cvbooster = None

    def train(
        self,
        params: dict = None,
        num_boost_round: int = 500,
        verbose_eval: int = 10,
        early_stopping_rounds: int = 50,
        algo: str = 'lgb'
    ):
        """Train machine learning model.

        Args:
            params (dict, optional): Hyperparameters. If None, default hyperparameters will be set.
                Defaults to None.
            num_boost_round (int, optional): Number of boosting round. Defaults to 500.
            verbose_eval (int, optional): Evaluation metric is printed every verbose_eval rounds.
                Only apply for xgboost. Defaults to 10.
            early_stopping_rounds (int, optional): Will stop training if metric in validation data
                doesn't improve in last early_stopping_round. Only applied if there is valid set
                when splitting or registering data. Only apply for xgboost. Defaults to 50.
            algo (str, optional): Algorithm. Must be either 'lgb' or 'xgb'. Defaults to 'lgb'.

        Raises:
            ValueError: algo must be either 'lgb' or 'xgb'.
        """
        if algo not in ['lgb', 'xgb']:
            raise ValueError("algo must be either 'lgb' or 'xgb'.")
        elif algo == 'lgb':
            LightGBM.train(self, params, num_boost_round, early_stopping_rounds)
        elif algo == 'xgb':
            XGBoost.train(self, params, num_boost_round, verbose_eval, early_stopping_rounds)
        self.algo = algo

    def tune(
        self,
        n_trials: int = 100,
        cv: bool = True,
        cv_method: str = 'random',
        params: dict = None,
        nfold: int = 5,
        valid_size: int = 1,
        gap: int = 2,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
        direction: str = 'maximize',
        algo: str = 'lgb'
    ):
        """Hyperparameter tunning lightgbm with optuna.

        Args:
            n_trials (int, optional): Number of tunning trials. Defaults to 100.
            cv (bool, optional): If use k-folds cross validation. Defaults to True.
            cv_method (str, optional): Method of cross validation. Must be either 'oot' or 'random'.
                Ignore when cv=False. Defaults to 'random'.
                Example of 3-fold ([train], [valid]) when cv_method='oot':
                ([202006, 202007, 202008, 202009, 202010], [202012])
                ([202006, 202007, 202008, 202009], [202011])
                ([202006, 202007, 202008], [202010])
            params (dict, optional): Hyperparameter search space. If None, pre-determined search
                space applies. Defaults to None. Example of params search space:
                {
                    'learning_rate': 0.1,
                    'max_depth': [3, 12],
                    'subsample': [0.3, 1.0],
                    'reg_alpha': [1e-8, 10.0, 'log'],
                    'boosting': ['gbdt', 'dart']
                }
            nfold (int, optional): Number of folds if cross validating. Defaults to 5.
            valid_size (int, optional): Number of ending periods in valid set when cv_method='oot'.
                Ignore if cv_method='random'. Defaults to 1.
            gap (int, optional): Gap in periods between the last period in train and the first in
                valid when cv_method='oot'. Example: 'gap=2' means use data upto now to predict
                label in 2 periods. Ignore if cv_method='random'. Defaults to 2.
            num_boost_round (int, optional): Number of boosting round. Defaults to 500.
            early_stopping_rounds (int, optional): Will stop training if (any) metric in validation
                data doesn't improve in last early_stopping_round. Set first_metric_only=True in
                params if want to use only the first metric for early stopping. Only apply for
                xgboost.Defaults to 50.
            direction (str, optional): Direction to optimize the metric. Defaults to 'maximize'.
            algo (str, optional): Algorithm. Must be either 'lgb' or 'xgb'. Defaults to 'lgb'.

        Raises:
            ValueError: Must spare a valid set when spliting or registering data.
            ValueError: cv_method must be either 'oot' or 'random'.
            ValueError: algo must be either 'lgb' or 'xgb'.
        """
        if algo not in ['lgb', 'xgb']:
            raise ValueError("algo must be either 'lgb' or 'xgb'.")
        elif algo == 'lgb':
            LightGBM.tune(
                self,
                n_trials,
                cv,
                cv_method,
                params,
                nfold,
                valid_size,
                gap,
                num_boost_round,
                direction
            )
        elif algo == 'xgb':
            XGBoost.tune(
                self,
                n_trials,
                cv,
                cv_method,
                params,
                nfold,
                valid_size,
                gap,
                num_boost_round,
                early_stopping_rounds,
                direction
            )
        self.algo = algo

    def calibrate(self, C: float = 1.0):
        """Calibrate the booster.

        Args:
            C (float, optional): Inverse of regularization strength. Defaults to 1.0.
        """
        if self.algo == 'xgb':
            test_feat = xgb.DMatrix(self.test_feat)
        else:
            test_feat = self.test_feat

        calibrated_booster = LogisticRegression(C=C)
        calibrated_booster.fit(
            X=self.booster.predict(test_feat).reshape(-1, 1),
            y=self.test_label)
        self.calibrated_booster = calibrated_booster

        self.calibrated_cvbooster = None
        if self.cvbooster:
            if self.algo == 'lgb':
                pred = self.cvbooster.predict(test_feat)
            if self.algo == 'xgb':
                pred = [booster.predict(test_feat) for booster in self.cvbooster]
            pred = np.mean(pred, axis=0)
            calibrated_cvbooster = LogisticRegression(C=C)
            calibrated_cvbooster.fit(
                X=pred.reshape(-1, 1),
                y=self.test_label)
            self.calibrated_cvbooster = calibrated_cvbooster
        print('Prediction scores have been probability calibrated.', end=' ')
        print('Set calibrate=False to get original scores when using method predict().')

    def predict(
        self,
        data: Union[str, pd.DataFrame],
        cvbooster: bool = True,
        calibrate: bool = True
    ) -> list:
        """Predict on new data.

        Args:
            data (Union[str, pd.DataFrame]): Data to be predicted. 'train' for train data, 'valid'
                for validation data, 'test' for test data.
            cvbooster (bool, optional): Use cvbooster if available. If cvbooster=True, predictions
                are average of scores from all boosters from cross validation. Defaults to True.
            calibrate (bool, optional): Use calibration model if available. Defaults to True.

        Returns:
            list: Prediction score.
        """
        if self.algo == 'lgb':
            pred = LightGBM.predict(self, data, cvbooster)
        elif self.algo == 'xgb':
            pred = XGBoost.predict(self, data, cvbooster)
        if calibrate:
            # if set cvbooster=True and calibrated_cvbooster is available
            if cvbooster and self.calibrated_cvbooster:
                pred = self.calibrated_cvbooster.predict_proba(pred.reshape(-1, 1))
                pred = np.array([x[1] for x in pred])
            # use calibration of single booster
            elif not cvbooster and self.calibrated_booster:
                pred = self.calibrated_booster.predict_proba(pred.reshape(-1, 1))
                pred = np.array([x[1] for x in pred])
        return pred

    def get_feature_importance(
        self,
        ntop: int = 10,
        importance_type: str = 'gain'
    ) -> pd.Series:
        """Get top feature importance from trained booster.

        Args:
            ntop (int, optional): Number of top features. Defaults to 10.
            importance_type (str, optional): 'gain' or 'split'. Defaults to 'gain'.

        Returns:
            pd.Series: Top feature importance.
        """
        if self.algo == 'lgb':
            importance = pd.Series(
                data=self.booster.feature_importance(importance_type=importance_type),
                index=self.booster.feature_name(),
                name='importance'
            )
        if self.algo == 'xgb':
            importance = self.booster.get_score(importance_type=importance_type)
            importance = pd.Series(importance)
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
        plt.title('Feature Importance', size=15)
        ax.bar_label(ax.containers[0], fmt=fmt, padding=padding)


def get_auc(
    true: list,
    pred: list,
) -> float:
    """Get AUC.

    Args:
        true (list): True label.
        pred (list): Prediction score.

    Returns:
        float: AUC
    """
    auc = roc_auc_score(true, pred)
    return auc


def get_confusion_matrix(
    true: list,
    pred: list,
    threshold: float = 0.5,
    percent: bool = False,
):
    """Plot confusion matrix.

    Args:
        true (list): True label.
        pred (list): Prediction score.
        threshold (float, optional): Threshold to convert score to label. Defaults to 0.5.
        percent (bool, optional): Display the quantities on confusion matrix as percentage.
            Defaults to False.
    """
    pred_label = np.where(pred < threshold, 0, 1)
    list1 = ['Predicted 1', 'Predicted 0']
    list2 = ['Actual 1', 'Actual 0']
    confusion = pd.DataFrame(confusion_matrix(pred_label, true), list1, list2)
    fmt = 'd'
    if percent:
        confusion = confusion / len(true)
        fmt = '.1%'
    sns.heatmap(confusion, annot=True, fmt=fmt, linewidths=.5, cmap='Blues')
    plt.title('Confusion Matrix', size=15)


def get_gain_table(true: list, pred: list, n_level: int = 10) -> pd.DataFrame:
    """Get gain table.

    Args:
        true (list): True label.
        pred (list): Prediction score.
        n_level (int, optional): Number of levels. Defaults to 10.

    Returns:
        pd.DataFrame: Gain table.
    """
    label_levels = reversed(range(1, 1 + n_level))
    label_levels = ['Level' + str(x) for x in label_levels]
    level = pd.qcut(pred, n_level, labels=label_levels, duplicates='drop')
    gain_df = pd.DataFrame({'true': true, 'predict': pred, 'level': level})
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


def plot_calibration_curve(true: list, pred: list, n_level: int = 10):
    """Plot calibration curve

    Args:
        true (list): True label.
        pred (list): Prediction score.
        n_level (int, optional): Number of levels. Defaults to 10.
    """
    gain_df = get_gain_table(true, pred, n_level)
    sns.lineplot(x='predict_mean', y='true_mean', data=gain_df, marker='o')
    max_predict_mean = max(gain_df['predict_mean'])
    max_true_mean = max(gain_df['true_mean'])
    lim_axis = max(max_predict_mean, max_true_mean)
    plt.plot([0.0, lim_axis], [0.0, lim_axis], linestyle='-.', color='grey', linewidth=0.5)
    plt.title('Calibration Curve', size=15)
