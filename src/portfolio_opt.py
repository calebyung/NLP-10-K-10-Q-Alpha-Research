# import project modules
from src.util import *
import src.constants as const

# import libraries
import os
import numpy as np
import pandas as pd
from IPython.display import display
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import cvxpy as cp


class PortfolioOpt:

    def __init__(self):
        log(f'Initializing PortfolioOpt...')
        self.config = yaml.safe_load(open('config.yml'))

    def load_data(self):
        log(f'Loading input data...')
        # load returns data
        self.ret = load_pkl(os.path.join(const.INTERIM_DATA_PATH, 'ret.pkl'))
        self.raw_ret_1d = self.ret.copy()
        self.exret = load_pkl(os.path.join(const.INTERIM_DATA_PATH, 'exret.pkl'))
        self.spy = load_pkl(os.path.join(const.INTERIM_DATA_PATH, 'spy.pkl'))['SPY']
        self.betas = load_pkl(os.path.join(const.INTERIM_DATA_PATH, 'betas.pkl'))
        log(f'Loaded ret: {self.ret.shape}')
        log(f'Loaded exret: {self.exret.shape}')
        log(f'Loaded spy: {self.spy.shape}')
        log(f'Loaded betas: {len(list(self.betas))}')

        # load combined signals
        feats = load_pkl(os.path.join(const.OUTPUT_DATA_PATH, 'feats.pkl'))
        log(f'Loaded feats: {self.feats.shape}')

        # get stock to cluster mapping
        doc_topics = load_pkl(os.path.join(const.INTERIM_DATA_PATH, 'doc_topics.pkl'))
        cik_map = load_pkl(os.path.join(const.INTERIM_DATA_PATH, 'cik_map.pkl'))
        log(f'Loaded doc_topics: {self.doc_topics.shape}')
        log(f'Loaded cik_map: {self.cik_map.shape}')
        doc_topics = doc_topics.merge(cik_map, how='outer', on='cik')
        self.topic_desc = doc_topics[['topic','topic_words']].loc[lambda x: x.topic.notnull()].drop_duplicates().sort_values('topic')
        doc_topics['topic'] = doc_topics['topic'].fillna(0)
        doc_topics['topic_words'] = doc_topics['topic'].map(dict(self.topic_desc.to_records(index=False)))
        doc_topics = doc_topics.loc[lambda x: x.stock.notnull()]
        doc_topics['topic'] = doc_topics['topic'].astype(int)
        clust_map = feats \
            .merge(doc_topics, how='inner', on='stock') \
            .loc[:,['stock','topic']] \
            .drop_duplicates() \
            .set_axis(['stock','cluster'], axis=1)
        clust_map = clust_map.assign(val=1).pivot('stock','cluster','val').fillna(0).astype(int)
        self.clust_map = clust_map
        log(f'Created clust_map: {clust_map.shape}')
        log(f'Sample clust_map:')
        display(self.clust_map.head())


    def preprocess_ret(self, ret, exret, spy, betas, h):
        # convert data to selected horizon
        n_day = self.config['horizons'][h]
        ret = (1+ret).rolling(n_day).apply(np.prod, raw=True) - 1
        exret = (1+exret).rolling(n_day).apply(np.prod, raw=True) - 1
        spy = (1+spy).rolling(n_day).apply(np.prod, raw=True) - 1
        beta = betas[h]

        # align index
        ret, exret, beta = align_col_index([ret, exret, beta])
        n_stock_cov = ret.shape[1]
        log(f'Number of stocks in ret, beta: {n_stock_cov}')

        # pre-compute variables needed for covariance matrix
        var_m = (spy.rolling(self.config['cov_window'], self.config['cov_window']//2).std())**2
        std_s = ret.rolling(self.config['cov_window'], self.config['cov_window']//2).std()
        var_s = std_s**2
        var_s_avg = (std_s.mean(axis=1))**2
        
        cov_params = [beta, n_stock_cov, var_m, var_s, var_s_avg]
        return ret, exret, spy, cov_params


    '''
    Create signal and raw weights
    '''
    def get_signal_and_raw_weights(self, selected_feat, n_day):
        # create signal table as pivot
        signal = self.feats[['stock', 'date', selected_feat]] \
            .drop_duplicates() \
            .pivot('date', 'stock', selected_feat)

        # forward-fill and align index with returns
        ret_ = df_drop_na(self.ret.shift(-n_day).loc[lambda x: (x.index>=self.config['bt_start_date']) & (x.index<=self.config['bt_end_date'])])
        dates = (signal.index | ret_.index).sort_values().tolist()
        signal = signal.reindex(index=dates).ffill()
        signal = df_drop_na(signal)
        signal, ret_ = align_index((signal, ret_))
        signal = signal.mask(ret_.isnull())
        log(f'Shape of signal: {signal.shape}')

        # ranked signal
        ranks = signal.rank(axis=1) - 1 / 2
        weights_raw = 2 * ranks.divide(ranks.count(axis=1), axis=0) - 1
        return signal, weights_raw



    def get_opt_gamma(self, date, weights_raw, sector_neutral, cov_params, cov_model):
        
        # prepare input values
        if cov_model=='cov_mkt_risk':
            sigma = get_cov_mkt_risk(date, self.params['cov_gamma'], cov_params)
        elif cov_model=='cov_simple':
            sigma = get_cov_simple(date, cov_params)        
        mu = weights_raw.loc[date].loc[lambda x: x.notnull()]
        stock_list = sorted((mu.index.intersection(sigma.index)).tolist())
        sigma = sigma.reindex(index=stock_list, columns=stock_list)
        mu = mu.reindex(index=stock_list).values.reshape(1,-1)
        n_stock_opt = len(stock_list)

        # setup optimizer
        w = cp.Variable(n_stock_opt)
        gamma = cp.Parameter(nonneg=True)
        ret_ = mu @ w 
        risk = cp.quad_form(w, sigma)
        basic_constraints = [cp.sum(w) == 0, cp.abs(w) <= 0.03, cp.norm(w, 1) <= 5]
        clust_constraints = [pd.Series(stock_list).map(self.clust_map[i]).values.reshape(1,-1) @ w == 0 for i in self.clust_map]
        constraints = basic_constraints + clust_constraints if sector_neutral==True else basic_constraints
        prob = cp.Problem(cp.Maximize(ret_ - gamma*risk), constraints)
        
        # grid search gamma
        n_sample = 20
        slope_data = np.zeros(n_sample)
        gamma_vals = np.logspace(3,7,n_sample)
        for i in range(n_sample):
            gamma.value = gamma_vals[i]
            prob.solve()
            slope_data[i] = ret_.value[0] / float(cp.sqrt(risk).value)
        slope_data = np.round(slope_data, 4)
        opt_gamma = gamma_vals[np.argmax(slope_data)]
        return date, opt_gamma, gamma_vals, slope_data



    def get_all_opt_gammas(self, weights_raw, sector_neutral, cov_params, cov_model, plot=False):
        opt_gammas = Parallel(n_jobs=-1)(delayed(self.get_opt_gamma)(date, weights_raw, sector_neutral, cov_params, cov_model) for date in weights_raw.index[::21])
        opt_gammas = [(x[0], x[1]) for x in opt_gammas]
        opt_gammas = pd.DataFrame(opt_gammas).set_axis(['date','opt_gamma'], axis=1).set_index('date').opt_gamma
        opt_gammas = weights_raw.merge(opt_gammas, how='left', left_index=True, right_index=True).opt_gamma.ffill()
        if plot:
            new_plot()
            np.log10(opt_gammas).plot(figsize=(10,5))
            plt.grid()
            plt.title('Optimal gamma (log10) over time (monthly sample)')
            plt.close()
        return opt_gammas



    def get_opt_weight(self, date, weights_raw, opt_gamma, sector_neutral, cov_params, cov_model):
        
        # prepare input values
        if cov_model=='cov_mkt_risk':
            sigma = df_drop_na(get_cov_mkt_risk(date, self.params['cov_gamma'], cov_params))
        elif cov_model=='cov_simple':
            sigma = df_drop_na(get_cov_simple(date, cov_params))
        mu = weights_raw.loc[date].loc[lambda x: x.notnull()]
        stock_list = sorted((mu.index.intersection(sigma.index)).tolist())
        sigma = sigma.reindex(index=stock_list, columns=stock_list)
        mu = mu.reindex(index=stock_list).values.reshape(1,-1)
        n_stock_opt = len(stock_list)

        # solve by optimizer
        w = cp.Variable(n_stock_opt)
        gamma = cp.Parameter(nonneg=True)
        ret_ = mu @ w 
        risk = cp.quad_form(w, sigma)
        basic_constraints = [cp.sum(w) == 0, cp.abs(w) <= 0.03, cp.norm(w, 1) <= 5]
        clust_constraints = [pd.Series(stock_list).map(self.clust_map[i]).values.reshape(1,-1) @ w == 0 for i in self.clust_map]
        constraints = basic_constraints + clust_constraints if sector_neutral==True else basic_constraints
        prob = cp.Problem(cp.Maximize(ret_ - gamma*risk), constraints)
        gamma.value = opt_gamma
        prob.solve()
        
        # output optimized weight
        w_opt = dict(zip(stock_list, w.value))
        w_opt = weights_raw.columns.map(w_opt)
        w_opt = pd.Series(w_opt, index=weights_raw.columns).rename(date)
        return w_opt



    def get_all_opt_weights(self, weights_raw, opt_gammas, sector_neutral, cov_params, cov_model, plot=False):
        weights_opt = Parallel(n_jobs=-1)(delayed(self.get_opt_weight)(date, weights_raw, opt_gammas.loc[date], sector_neutral, cov_params, cov_model) for date in weights_raw.index)
        weights_opt = pd.concat(weights_opt, axis=1).T.sort_index()

        if plot:
            new_plot()
            weights_opt.notnull().sum(axis=1).plot(figsize=(10,5))
            plt.grid()
            plt.title('Number of stocks per day')
            plt.close()

            new_plot()
            weights_opt.abs().sum(axis=1).plot(figsize=(10,5))
            plt.grid()
            plt.title('Total leverage per day')
            plt.close()

            new_plot()
            weights_opt.unstack().hist(figsize=(10,5), bins=50)
            plt.title('Weight distribution')
            plt.close()

            new_plot()
            weights_opt.sum(axis=1).plot(figsize=(10,5))
            plt.grid()
            plt.title('Total weight per day')
            plt.close()

            new_plot()
            for c in self.clust_map:
                weights_opt[self.clust_map.reindex(index=weights_opt.columns).loc[lambda x: x[c]==1].index.tolist()] \
                .sum(axis=1) \
                .plot(figsize=(10,5), label=f'Sector {c}: {self.topic_desc.set_index("topic")["topic_words"].loc[c]}', legend=True)
            plt.legend(bbox_to_anchor=(1.0, 1.0))
            plt.grid()
            plt.title('Sector total weights per day')
            plt.close()
        return weights_opt



    '''
    Portfolio optimization given a selected signal
    '''
    def optimize(self, selected_feat, h, cov_model, sector_neutral, plot):

        # load input data
        self.load_data()
        
        # optimize weights
        n_day = self.config['horizons'][h]
        ret, exret, spy, cov_params = self.preprocess_ret(self.ret, self.exret, self.spy, self.betas, h)
        signal, weights_raw = self.get_signal_and_raw_weights(selected_feat, n_day)
        opt_gammas = self.get_all_opt_gammas(weights_raw, sector_neutral, cov_params, cov_model, plot)
        opt_gammas = opt_gammas*0 + opt_gammas.mean() # use mean of optimized gamma for all periods
        weights_opt = self.get_all_opt_weights(weights_raw, opt_gammas, sector_neutral, cov_params, cov_model, plot)
        weights_corr = pd.Series(weights_raw.index).apply(lambda x: pd.concat([weights_raw.loc[x], weights_opt.loc[x]], axis=1).corr().iloc[0,1]).mean()
        
        # raw portfolio
        ret_, exret_, spy_ = ret.shift(-n_day), exret.shift(-n_day), spy.shift(-n_day)
        weights_raw, ret_ = align_index([weights_raw, ret_])
        port_ret_raw = (weights_raw * ret_).sum(axis=1)
        sharpe_raw = port_ret_raw.mean() * np.sqrt(252/n_day) / port_ret_raw.std()
        quintile_plot(weights_raw, ret_, desc='Raw Weights')
        pnl_raw = get_pnl(weights_raw, self.raw_ret_1d, n_day, plot_title='Raw PnL Curve (for $1 initial investment)')
        
        # optimized portfolio
        ret_, exret_, spy_ = ret.shift(-n_day), exret.shift(-n_day), spy.shift(-n_day)
        weights_opt, ret_ = align_index([weights_opt, ret_])
        port_ret_opt = (weights_opt * ret_).sum(axis=1)
        sharpe_opt = port_ret_opt.mean() * np.sqrt(252/n_day) / port_ret_opt.std()
        quintile_plot(weights_opt, ret_, desc='Optimized Weights')
        pnl_opt = get_pnl(weights_opt, self.raw_ret_1d, n_day, plot_title='Optimized PnL Curve (for $1 initial investment)')

        # report results
        log(f'Selected signal : {selected_feat}')
        log(f'Selected horizon : {h}')
        log(f'Selected n-days horizon : {n_day}')
        log(f'Mean optimized gamma: {opt_gammas.iloc[0]}')
        log(f'Sector neutral : {sector_neutral}')
        log(f'Risk model : {cov_model}')
        log(f'Avg correlation bewteen raw and optimized weights: {weights_corr}')
        log(f'Raw Sharpe : {sharpe_raw}')
        log(f'Optimized Sharpe : {sharpe_opt}')
        
        return weights_raw, port_ret_raw, weights_opt, port_ret_opt



'''
Risk models
'''
# function to calculate covariance matrix given a date
def get_cov_mkt_risk(date, cov_gamma, cov_params):
    beta, n_stock_cov, var_m, var_s, var_s_avg = cov_params
    i_var_m = var_m.loc[date]
    i_beta = pd.DataFrame(beta.loc[date])
    i_var_s_avg = var_s_avg.loc[date]
    i_var_s = var_s.loc[date]
    cov = i_var_m * (i_beta @ i_beta.T) + cov_gamma * i_var_s_avg * np.identity(n_stock_cov) + (1 - cov_gamma) * np.diag(i_var_s)
    return df_drop_na(cov)

def get_cov_simple(date, cov_params):
    _, _, _, var_s, _ = cov_params
    i_var_s = var_s.loc[date].loc[lambda x: x.notnull()]
    cov = pd.DataFrame(np.diag(i_var_s), index=i_var_s.index, columns=i_var_s.index)
    return cov


'''
PnL Curve functions
'''
# weights and ret must be aligned in index before calling function
# weights is a 1-d vector with shape of (n_stock, )
def get_single_trade_pnl(w, r, init_cash):
    ret_cum = (1 + r).cumprod() - 1
    weighted_ret_cum = ret_cum.multiply(w, axis=1).sum(axis=1)
    pnl = init_cash * (1 + weighted_ret_cum)
    return pnl

# weights (n_dates, n_stock)
# ret (n_dates, n_stock) - raw 1-day returns
def get_pnl(weights, ret, n_day, plot_title=None):
    total_pnl = []
    # for each starting date, build an individual portfolio and PnL
    # if each investment is 30-day long, we contruct 30 different potfolios by shifting start dates one-by-one
    # final PnL is based on the average of individual portfolios
    for i in range(n_day):
        # get consecutive investment periods
        i_weights = weights.iloc[i::n_day]
        bal = 1
        i_pnl = []
        for date in i_weights.index:
            w = i_weights.loc[date]
            r = ret.loc[lambda x: x.index>date].iloc[:n_day]
            stocks = sorted(list(set(w.index) & set(r.columns)))
            w = w.reindex(index=stocks)
            r = r.reindex(columns=stocks)
            pnl = get_single_trade_pnl(w=w, r=r, init_cash=bal)
            bal = pnl[-1]
            i_pnl.append(pnl)
        i_pnl = pd.concat(i_pnl, axis=0)
        total_pnl.append(i_pnl)
    total_pnl = pd.concat(total_pnl, axis=1)
    total_pnl = total_pnl.ffill().sum(axis=1) / n_day
    total_pnl = total_pnl.iloc[n_day:]
    if plot_title!=None:
        new_plot()
        total_pnl.plot(figsize=(10,5))
        plt.grid()
        plt.title(plot_title)
        plt.show()
        plt.close()
    return total_pnl


'''
Quintile Plots
'''
def quintile_plot(weights, ret, desc):
    
    # quintile plot
    new_plot()
    quintiles = [f'Q{i}' for i in range(1, 6)]
    q_portfolios = {}
    for i, quintile in enumerate(quintiles):
        lbound = weights.ge(weights.quantile(q=i/5, axis=1), axis=0)
        if i + 1 < 5:
            ubound = weights.lt(weights.quantile(q=(i+1)/5, axis=1), axis=0)
        else:
            ubound = weights.le(weights.quantile(q=(i+1)/5, axis=1), axis=0)
        q_weights = weights[lbound & ubound]
        q_weights = q_weights.mask(~q_weights.isnull(), 1)
        q_weights = q_weights.divide(q_weights.count(axis=1), axis=0)
        q_portfolios[quintile] = q_weights
    q_returns = {}
    for i, quintile in enumerate(quintiles):
        q_returns[quintile] = (q_portfolios[quintile] * ret).sum(axis=1)    
        q_returns[quintile].cumsum().plot(figsize=(10,5), label=quintile, legend=True)
    plt.title(f'Quintile plot ({desc})')
    plt.grid()
    plt.close()
    
    # Q5-Q1 plot
    new_plot()
    ls_returns = q_returns['Q5'] - q_returns['Q1']
    ls_returns.cumsum().plot(figsize=(10,5), label='Q5 - Q1', legend=True, color='k')
    plt.title(f'Q5-Q1 plot ({desc})')
    plt.grid()
    plt.close()
    return