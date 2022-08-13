
# import project modules
from src.util import *
import src.constants as const

# import libraries
import os
from os.path import isfile, isdir, join
import numpy as np
import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from bs4 import BeautifulSoup
import re
from IPython.display import display
from zipfile import ZipFile
import pickle
import unicodedata
import pytz
from joblib import Parallel, delayed
import shutil
import difflib
import random
import math
from shutil import copyfile
import itertools
import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score

import lightgbm as lgbm
import optuna
from optuna import Trial, visualization

import matplotlib as mpl
from matplotlib import pyplot as plt


class LTRModel:

    def __init__(self):
        log(f'Initializing SignalAnalytics...')
        self.config = yaml.safe_load(open('config.yml'))
        self.n_day_trade = int(self.config['horizons'][self.config['ltr_h']])
        self.n_day_calendar = int(self.config['horizons'][self.config['ltr_h']] * 365/252)
        if self.config['gpu_enabled'] == True:
            self.device_type = 'GPU'
        else:
            self.device_type = 'CPU'


    # load excess return
    def load_exret(self):
        exret = load_pkl(os.path.join(const.INTERIM_DATA_PATH, 'exret.csv'))
        exret = (1+exret).rolling(self.n_day_trade).apply(np.prod, raw=True).shift(-self.n_day_trade) - 1
        exret = exret \
            .unstack() \
            .loc[lambda x: x.notnull()] \
            .reset_index() \
            .rename(columns={'level_0':'stock',0:'exret'})
        log(f'Shape of exret: {exret.shape}')
        self.exret = exret


    def load_signals(self):
        log(f'Loading 10-K signals...')
        feats_10k = load_pkl(os.path.join(const.INTERIM_DATA_PATH, 'feats_10k.pkl')) \
            .rename(columns={'filing_date':'date'}) \
            .assign(date = lambda x: pd.to_datetime(x.date),
                    cik = lambda x: x.cik.astype(str).str.zfill(10)) \
            .replace([np.inf, -np.inf], np.nan)
        self.feat_names = [c for c in feats_10k.columns if 'feat' in c]
        feats_10k = feats_10k.rename(columns={c:c+'_10k' for c in self.feat_names})
        feats_10k = feats_10k.loc[:,['stock','date'] + [c for c in feats_10k.columns if 'feat' in c]]
        log(f'Shape of 10-K feats: {feats_10k.shape}')

        # load 10-Q signals
        feats_10q = load_pkl(os.path.join(const.INTERIM_DATA_PATH, 'feats_10q.pkl')) \
            .rename(columns={'filing_date':'date'}) \
            .assign(date = lambda x: pd.to_datetime(x.date),
                    cik = lambda x: x.cik.astype(str).str.zfill(10)) \
            .replace([np.inf, -np.inf])
        self.feat_names = [c for c in feats_10q.columns if 'feat' in c]
        feats_10q = feats_10q.rename(columns={c:c+'_10q' for c in self.feat_names})
        feats_10q = feats_10q.loc[:,['stock','date'] + [c for c in feats_10q.columns if 'feat' in c]]
        log(f'Shape of 10-Q feats: {feats_10q.shape}')

        # load 8-k signal
        feats_8k = load_pkl(os.path.join(const.INTERIM_DATA_PATH, 'feats_8k.pkl'))
        log(f'Shape of 8-K feats: {feats_8k.shape}')

        # combine all signals into single df
        self.feats = feats_10k \
            .merge(feats_10q, how='outer', on=['stock','date']) \
            .merge(feats_8k, how='outer', on=['stock','date']) \
            .sort_values(['stock','date']) \
            .groupby('stock') \
            .apply(lambda x: x.ffill()) \
            .reset_index(drop=True)
        self.feat_names = [c for c in self.feats.columns if 'feat' in c]
        log(f'Shape of combined feats: {self.feats.shape}')
        self.feats = self.feats \
            .loc[lambda x: x[self.feat_names].notnull().sum(axis=1) > 0] \
            .reset_index(drop=True)


    # merge signal and rank to build the complete dataset
    def preprocess_signals(self):
        # merge with rank data    
        data = self.feats \
            .merge(self.exret, how='outer', on=['stock','date']) \
            .sort_values(['stock','date'])
        # fillna signal
        data[self.feat_names] = data.groupby('stock')[self.feat_names].ffill()
        data = data.loc[lambda x: x[self.feat_names].notnull().sum(axis=1) > 0]
        data[self.feat_names] = data[self.feat_names].fillna(data[self.feat_names].mean(axis=0))
        # derive rank of exret
        data = data.sort_values(['date','exret']).reset_index(drop=True)
        data['rank'] = data.groupby('date')['exret'].rank()
        data = data \
            .loc[lambda x: x['rank'].notnull()] \
            .sort_values(['date','rank']) \
            .drop('exret', axis=1) \
            .reset_index(drop=True)
        log(f'Shape of full dataset: {data.shape}')
        log(f'Number of nulls in full dataset: {data.isnull().sum().sum()}')
        # check group size distribution
        new_plot()
        data.groupby('date').size().hist(bins=50)
        plt.title('Group size distribution')
        plt.close()
        # scaling
        data[self.feat_names] = StandardScaler().fit_transform(data[self.feat_names])
        # fianl sort
        data = data.sort_values(['date','rank']).reset_index(drop=True)
        # DQ
        assert data[['stock','date']].drop_duplicates().shape[0]==data.shape[0]
        log(f'Shape of data: {data.shape}')
        log(f'Total number of groups: {data.date.nunique()}')
        log(f'Sample of data:')
        display(data.head())
        # save results
        self.data = data


    def prep_full_dataset(self):
        self.load_signals()
        self.load_exret()
        self.preprocess_signals()


    def train_model(self, date, params, hp_tune):
        # unpack model params
        lgbm_var_params, lgbm_fixed_params = params
        # define dataset
        date_trn_start, date_trn_end, date_val_start, date_val_end = get_trn_val_test_dates(date, self.config['n_day_trn'], self.config['n_day_val'], self.n_day_calendar)
        X_trn, y_trn, grp_trn, qid_trn, header_trn = get_dataset(self.data.loc[lambda x: x.date.between(date_trn_start,date_trn_end)], self.feat_names)
        X_val, y_val, grp_val, qid_val, header_val = get_dataset(self.data.loc[lambda x: x.date.between(date_val_start,date_val_end)], self.feat_names)
        # model fit
        model = lgbm.LGBMRanker(**lgbm_fixed_params, **lgbm_var_params)
        model.fit(
            X = X_trn,
            y = y_trn,
            group = grp_trn,
            eval_set = [(X_trn, y_trn),(X_val, y_val)],
            eval_names = ['Train', 'Validation'],
            eval_group = [grp_trn, grp_val],
            eval_at = 1000,
            verbose = False,
            eval_metric = 'ndcg'
        )
        # output
        if hp_tune:
            model_output = -1 * model.best_score_['Validation']['ndcg@1000']
        else:
            # prediction
            pred_trn = pred_rank(model, X_trn, qid_trn)
            pred_val = pred_rank(model, X_val, qid_val)
            pred_val_out = pd.concat([header_val, pred_val], axis=1)
            # evaluation metrics
            corr_trn = pd.concat([y_trn, pred_trn], axis=1).corr().iloc[0,1]
            corr_val = pd.concat([y_val, pred_val], axis=1).corr().iloc[0,1]
            # output dict
            model_output = dict(
                date = date,
                ndcg_trn = model.evals_result_['Train']['ndcg@1000'][-1],
                ndcg_val = model.evals_result_['Validation']['ndcg@1000'][-1],
                pred_val = pred_val,
                corr_trn = corr_trn,
                corr_val = corr_val,
                evals_result_ = model.evals_result_,
                pred_val_out = pred_val_out,
                feat_imp = pd.DataFrame(list(zip(model.feature_name_, model.feature_importances_)), columns=['feat','imp']),
            )
        return model_output


    '''
    Function to output -ve CV score given a parameter set; used for HP optimization
    '''
    def Objective(self, trial):
        lgbm_var_params = dict(
            max_depth = trial.suggest_int('max_depth', 2, 32, log=True),
            num_leaves = trial.suggest_int('num_leaves', 16, 64, log=True),
            learning_rate = trial.suggest_float("learning_rate", 0.5, 3),
            objective = trial.suggest_categorical("objective", ['rank_xendcg']),
            min_split_gain = 0,
            min_child_weight = trial.suggest_float("min_child_weight", 1, 4, log=True),
            min_child_samples = trial.suggest_int('min_child_samples', 10, 80, log=True),
            subsample = trial.suggest_float("subsample", 0.5, 1),
            subsample_freq = trial.suggest_categorical("subsample_freq", [1,2,4,6,8]),
            colsample_bytree = trial.suggest_float("subsample", 0.5, 1),
            reg_alpha = trial.suggest_float("reg_alpha", 1e-5, 3, log=True),
            reg_lambda = trial.suggest_float("reg_lambda", 1e-5, 3, log=True),
            n_estimators = trial.suggest_int('n_estimators', 5, 600, log=True),
        )
        
        lgbm_fixed_params = dict(
            boosting_type = 'gbdt',
            n_jobs = -1,
            random_state = 0,
            label_gain = [i for i in range(int(self.data['rank'].max()) + 1)],
            device_type = self.config['device_type']
        )
        
        # pack all params
        params = (lgbm_var_params, lgbm_fixed_params)
        # return average CV score
        cv_score = np.mean([self.train_model(date, params, hp_tune=True) for date in self.sampled_dates])
        return cv_score


    '''
    Function to run Hyperparameter Optimization
    '''
    def optimize_hp(self):
        # get sampled dates for CV
        s = pd.Series(self.data.date.unique()) \
            .loc[lambda x: x.between(x.min() + np.timedelta64(self.condif['n_day_trn'] + self.n_day_calendar + self.condif['n_day_val'], 'D'), self.condif['bt_start_date'])] \
            .tolist()
        self.sampled_dates = sorted(s[::len(s)//self.config['n_day_sample']])
        log(f'Sampled dates:')
        display(self.sampled_dates)

        if self.config['run_hp_tune'] == True:
            # run optimization
            study = optuna.create_study(direction="minimize", study_name='LGBM optimization')
            study.optimize(self.Objective, timeout=self.config['hp_opt_hrs']*60*60)
            
            # save results
            self.best_params = study.best_params
            self.best_score = study.best_value
            log(f'Best params: best_params')
            log(f'Highest NDCG is {-self.best_score}')
            trials = study.trials_dataframe()
            trials.to_csv(os.path.join(const.OUTPUT_DATA_PATH, 'hp_tuning_trials.csv'), index=False)
            save_pkl(self.best_params, os.path.join(const.INTERIM_DATA_PATH, 'lgbm_best_params.pkl'))
            save_pkl(study, os.path.join(const.INTERIM_DATA_PATH, 'study.pkl'))
            
            # visualise relationship between parameter and CV score
            for c in trials.columns:
                if c[:7]=='params_':
                    new_plot()
                    trials.plot.scatter(c, 'value')
                    plt.grid()
                    plt.title(c)
                    plt.show()
                    plt.close()
        else:
            self.best_params = self.config['best_params']


    def refit_best_model(self):
        # setup best params
        lgbm_fixed_params = dict(
            boosting_type = 'gbdt',
            n_jobs = -1,
            random_state = 0,
            label_gain = [i for i in range(int(self.data['rank'].max()) + 1)],
            device_type = self.device_type
        )
        params = (self.best_params, lgbm_fixed_params)

        # get full list of dates
        date_list = pd.Series(self.data.date.unique()) \
                .loc[lambda x: x.between(x.min() + np.timedelta64(self.config['n_day_trn'] + self.n_day_calendar + self.config['n_day_val'], 'D'), x.max())]
        n_group = (date_list.max() - date_list.min()) / np.timedelta64(1,'D') / self.config['n_day_val']
        n_group = math.ceil(n_group + 1 + 1e-5)
        date_list = pd.Series([date_list.min() + np.timedelta64(i * self.config['n_day_val'], 'D') for i in range(n_group)])
        date_list = date_list[date_list.apply(lambda d: self.data.loc[lambda x: x.date.between(d + np.timedelta64(-self.config['n_day_val']+1,'D'), d)].shape[0]) > 0]
        date_list = date_list.tolist()
        log(f'Total number of days to train model: {len(date_list)}')

        # generate outputs for all dates
        model_outputs = []
        for date in date_list[:self.config['n_day_final_predict']]:
            model_outputs.append(self.train_model(date, params, hp_tune=False))
            log(f'Completed model training for date {date}')
        self.model_outputs = model_outputs


    def analyze_model_perf(self):
        dates = pd.Series([x['date'] for x in self.model_outputs])

        # ndcg score
        ndcg_trn = pd.Series([x['ndcg_trn'] for x in self.model_outputs],index=dates)
        ndcg_val = pd.Series([x['ndcg_val'] for x in self.model_outputs],index=dates)
        new_plot()
        plt.figure(figsize=(12, 4))
        ndcg_trn.plot(label='trn')
        ndcg_val.plot(label='val')
        plt.grid()
        plt.legend()
        plt.title('NDCG score')
        plt.show()
        plt.close()

        # correlation score
        corr_trn = pd.Series([x['corr_trn'] for x in self.model_outputs],index=dates)
        corr_val = pd.Series([x['corr_val'] for x in self.model_outputs],index=dates)
        new_plot()
        plt.figure(figsize=(12, 4))
        corr_trn.plot(label='trn')
        corr_val.plot(label='val')
        plt.grid()
        plt.legend()
        plt.title('Correlation score')
        plt.show()
        plt.close()


        # model predictions
        pred_val_out = pd.concat([x['pred_val_out'] for x in self.model_outputs]).reset_index(drop=True)
        new_plot()
        plt.figure(figsize=(12, 4))
        pred_val_out.groupby('date').size().plot()
        plt.grid()
        plt.title('Number of predictions made per date')
        plt.show()
        plt.close()

        # feature importance
        feat_imp = pd.concat([x['feat_imp'] for x in self.model_outputs]).groupby('feat').mean().sort_values('imp')
        new_plot()
        feat_imp.plot.barh(figsize=(12, 4))
        plt.title('Feature Importance')
        plt.show()
        plt.close()

        # export
        save_pkl(self.model_outputs, os.path.join(const.INTERIM_DATA_PATH, 'model_outputs.pkl'))
        save_pkl(pred_val_out, os.path.join(const.INTERIM_DATA_PATH, 'pred_val_out.pkl'))



# Function to derive training/validation start/end dates
def get_trn_val_test_dates(date, n_day_trn, n_day_val, n_day_calendar):
    date_val_end = np.datetime64(date)
    date_val_start = date_val_end + np.timedelta64(-n_day_val+1,'D')
    date_horizon_end = date_val_start + np.timedelta64(-1,'D')
    date_horizon_start = date_horizon_end + np.timedelta64(-n_day_calendar+1,'D')
    date_trn_end = date_horizon_start + np.timedelta64(-1,'D')
    date_trn_start = date_trn_end + np.timedelta64(-n_day_trn+1,'D')
    return date_trn_start, date_trn_end, date_val_start, date_val_end

def get_dataset(data, selected_feats):
    df = data.sort_values(['date','rank']).reset_index(drop=True)
    grp = df.groupby('date').size().tolist()
    qid = df['date']
    X = df[selected_feats]
    y = df['rank']
    header = df[['stock','date']]
    return X, y, grp, qid, header

def pred_score(model, X, qid):
    X_ = X.assign(date=qid)
    scores = []
    for date in X_.date.unique():
        scores += pd.Series(model.predict(X_.loc[lambda x: x.date==date].drop('date',axis=1))).tolist()
    return pd.Series(scores)

def pred_rank(model, X, qid):
    X_ = X.assign(date=qid)
    rnk = []
    for date in X_.date.unique():
        rnk += pd.Series(model.predict(X_.loc[lambda x: x.date==date].drop('date',axis=1))).rank().tolist()
    return pd.Series(rnk)

def eval_corr(rank_pred, rank_true, qid):
    df = pd.DataFrame({'rank_pred':rank_pred, 'rank_true':rank_true, 'qid':qid})
    avg_corr = df.groupby('qid').apply(lambda x: x.corr().iloc[0,1]).mean()
    return avg_corr


