# import project modules
from src.util import *
import src.constants as const

# import libraries
import os
import numpy as np
import pandas as pd
from IPython.display import display
import itertools
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt




class SignalAnalytics:

    def __init__(self):
        log(f'Initializing SignalAnalytics...')
        self.config = yaml.safe_load(open('config.yml'))


    def load_returns(self):
        log(f'Loading returns...')
        # import data
        ret = load_pkl(os.path.join(const.INTERIM_DATA_PATH, 'ret.pkl'))
        exret = load_pkl(os.path.join(const.INTERIM_DATA_PATH, 'exret.pkl'))

        # filter returns to testing period
        exret = exret.loc[lambda x: (x.index>=self.config['bt_start_date']) & (x.index<=self.config['bt_end_date'])]
        exret = df_drop_na(exret)
        ret, exret = align_index((ret, exret))
        log(f'Shape of ret: {ret.shape}')
        log(f'Shape of exret: {exret.shape}')
        self.ret = ret
        self.exret = exret


    def load_signals(self):
        log(f'Loading 10-K signals...')
        feats_10k = load_pkl(os.path.join(const.INTERIM_DATA_PATH, 'feats_10k.pkl')) \
            .rename(columns={'filing_date':'date'}) \
            .assign(date = lambda x: pd.to_datetime(x.date),
                    cik = lambda x: x.cik.astype(str).str.zfill(10)) \
            .replace([np.inf, -np.inf], np.nan)
        feat_names = [c for c in feats_10k.columns if 'feat' in c]
        feats_10k = feats_10k.rename(columns={c:c+'_10k' for c in feat_names})
        feats_10k = feats_10k.loc[:,['stock','date'] + [c for c in feats_10k.columns if 'feat' in c]]
        log(f'Shape of 10-K feats: {feats_10k.shape}')

        # load 10-Q signals
        feats_10q = load_pkl(os.path.join(const.INTERIM_DATA_PATH, 'feats_10q.pkl')) \
            .rename(columns={'filing_date':'date'}) \
            .assign(date = lambda x: pd.to_datetime(x.date),
                    cik = lambda x: x.cik.astype(str).str.zfill(10)) \
            .replace([np.inf, -np.inf])
        feat_names = [c for c in feats_10q.columns if 'feat' in c]
        feats_10q = feats_10q.rename(columns={c:c+'_10q' for c in feat_names})
        feats_10q = feats_10q.loc[:,['stock','date'] + [c for c in feats_10q.columns if 'feat' in c]]
        log(f'Shape of 10-Q feats: {feats_10q.shape}')

        # load 8-k signal
        feats_8k = load_pkl(os.path.join(const.INTERIM_DATA_PATH, 'feats_8k.pkl'))
        log(f'Shape of 8-K feats: {feats_8k.shape}')

        # load LTR signal
        # feats_lgbm_ltr = load_pkl(f'../input/hkml-lightgbm-ltr/pred_val_out').rename(columns={0:'feat_lgbm_ltr_12m'})
        # log(f'Shape of LGBM Learning-to-Rank feats: {feats_lgbm_ltr.shape}')

        # combine all signals into single df
            # .merge(feats_lgbm_ltr, how='outer', on=['stock','date']) \
        self.feats = feats_10k \
            .merge(feats_10q, how='outer', on=['stock','date']) \
            .merge(feats_8k, how='outer', on=['stock','date']) \
            .sort_values(['stock','date']) \
            .groupby('stock') \
            .apply(lambda x: x.ffill()) \
            .loc[lambda x: (x.date>=self.config['bt_start_date']) & (x.date<=self.config['bt_end_date'])] \
            .reset_index(drop=True)

        # summary DQ
        self.feat_names = [c for c in self.feats.columns if 'feat' in c]
        log(f'Shape of combined feats: {self.feats.shape}')
        log(f'Sample total feats:')
        display(self.feats.head())


    def analyze_feat(self, feats, ret, exret, selected_feat):
        log(f'Analyzing signal {selected_feat}...')
        # create signal table as pivot
        signal = feats[['stock', 'date', selected_feat]] \
            .drop_duplicates() \
            .pivot('date', 'stock', selected_feat)

        # forward-fill and align index with returns
        dates = (signal.index | ret.index).sort_values().tolist()
        signal = signal.reindex(index=dates).ffill()
        signal = df_drop_na(signal)
        signal, ret_, exret_ = align_index((signal, ret, exret))

        log(f'Shape of ret: {ret_.shape}')
        log(f'Shape of exret: {exret_.shape}')
        log(f'Shape of signal: {signal.shape}')

        # calculate metrics per investiment horizon
        metric = []
        for h in self.config['bt_horizons']:
            metric.append([selected_feat, h, self.config['bt_horizons'][h]] + list(gen_metric(signal, ret, exret, self.config['bt_horizons'][h])))
        metric = pd.DataFrame(metric, columns=['feat', 'horizon', 'n_day', 'avg_rnk_corr', 'sharpe_exret', 'sharpe_ret', 'sharpe_ret_vol'])
        return metric


    def analyze_all_basic_feats(self):
        self.feat_metric = [self.analyze_feat(self.feats, self.ret, self.exret, f) for f in self.feat_names]
        self.feat_metric = pd.concat(self.feat_metric, axis=0).reset_index(drop=True)


    # based on first round analysis, compute various weighted averages of signals
    def add_weighted_avg_signals(self):
        log(f'Calculating weighted avg signals...')
        s_list = [0, 0.4]
        k_list = [0.05, 0.10, 0.15, 0.20, -0.05, -0.10, -0.15, -0.20, 0]
        t_dict = {'minmax': MinMaxScaler(),
                'uniform': QuantileTransformer(output_distribution='uniform', random_state=0),
                'normal': QuantileTransformer(output_distribution='normal', random_state=0)}

        list1 = list(itertools.product(s_list, [-0.1,-0.05, 0, 0.05, 0.1], ['normal']))
        list2 = list(itertools.product(s_list, [0,0.5,1,2,3], ['minmax','uniform']))

        for s, k, t in list1 + list2:
            weights = self.feat_metric \
                .loc[lambda x: (x.horizon=='12m') & (x.sharpe_ret>s) & (~x.feat.str.contains('avg')) & (~x.feat.str.contains('lgbm'))] \
                .loc[:, ['feat','sharpe_ret']] \
                .sort_values('sharpe_ret', ascending=False)
            weights['imp'] = np.exp(k * weights['sharpe_ret'])
            weights['weight'] = weights['imp'] / np.sum(weights['imp'])
            df = self.feats[weights.feat.tolist()]
            df = pd.DataFrame(t_dict[t].fit_transform(df), columns=df.columns)
            df = df.multiply(weights.weight.tolist(), axis=1)
            self.feats[f'feat_weighted_avg_s{s}_k{k}_{t}'] = df.sum(axis=1)
        log(f'Sample of weighted avg signals:')
        display(self.feats[[c for c in self.feats.columns if 'avg' in c]])



    def analyze_all_avg_feats(self):
        feat_avg_names = [c for c in self.feats.columns if 'feat_weighted_avg' in c]
        feat_metric_avg = [self.analyze_feat(self.feats, self.ret, self.exret, f) for f in feat_avg_names]
        feat_metric_avg = pd.concat(feat_metric_avg, axis=0).reset_index(drop=True)
        self.feat_metric = pd.concat([self.feat_metric, feat_metric_avg], axis=0).reset_index(drop=True)
        

    def display_all_results(self):
        # display all signal metric output
        feat_names = [c for c in self.feats.columns if 'feat' in c]
        for feat in feat_names:
            display(self.feat_metric.loc[lambda x: x.feat==feat])

        # display all feats
        log(f'Shape of combined feats: {self.feats.shape}')
        log(f'Sample of combined feats:')
        display(self.feats.head())

        # signal correlation plot
        corr = self.feats[feat_names].corr()
        corr.style.background_gradient(cmap='coolwarm')


    def export(self):
        save_pkl(self.feats, os.path.join(const.OUTPUT_DATA_PATH, 'feats.pkl'))
        self.feat_metric.to_csv(os.path.join(const.OUTPUT_DATA_PATH, 'feat_metric.csv'), index=False)


def get_portfolio_ret(signal, f_ret, n_day, div_vol=False):
    ranks = signal.rank(axis=1) - 1 / 2
    weights = 2 * ranks.divide(ranks.count(axis=1), axis=0) - 1
    std = f_ret.shift(n_day).rolling(252, 252//2).std() if div_vol else 1
    port_ret = (weights / std * f_ret).sum(axis=1)
    return port_ret


def get_sharpe(port, n_day):
    return port.mean() * np.sqrt(252/n_day) / port.std()


def gen_metric(signal, ret, exret, n_day):
    # future returns
    f_ret = (1+ret).rolling(n_day).apply(np.prod, raw=True).shift(-n_day) - 1
    f_exret = (1+exret).rolling(n_day).apply(np.prod, raw=True).shift(-n_day) - 1
    f_ret, f_exret = df_drop_na(f_ret), df_drop_na(f_exret)
    signal, f_ret, f_exret = align_index((signal, f_ret, f_exret))
    signal = signal.mask(f_ret.isnull())
    # ranked signal, returns
    signal_rnk, f_ret_rnk, f_exret_rnk = signal.rank(axis=1), f_ret.rank(axis=1), f_exret.rank(axis=1)
    # average correlation between signal and excess return (both ranked)
    avg_rnk_corr = pd.Series(signal_rnk.index).apply(lambda x: pd.concat([signal_rnk.loc[x], f_exret_rnk.loc[x]], axis=1).corr().iloc[0,1]).mean()
    # construct uniform weight portfolio
    port_ret = get_portfolio_ret(signal, f_ret, n_day)
    port_exret = get_portfolio_ret(signal, f_exret, n_day)
    port_ret_vol = get_portfolio_ret(signal, f_ret, n_day, div_vol=True)
    # calculate sharpe ratios
    sharpe_ret = get_sharpe(port_ret, n_day)
    sharpe_exret = get_sharpe(port_exret, n_day)
    sharpe_ret_vol = get_sharpe(port_ret_vol, n_day)
    return avg_rnk_corr, sharpe_exret, sharpe_ret, sharpe_ret_vol


