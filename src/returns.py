# load main config
with open('config.yml') as file:
    config = yaml.safe_load(file)   
    file.close()

# import project modules
from re import X
from src.util import *
import constants as c

# import other packages
import os
import numpy as np
from datetime import datetime
import time
from dateutil.relativedelta import relativedelta
from IPython.display import display
from matplotlib import pyplot as plt
import random
import yaml
import pandas as pd
import warnings
import quandl
from alpha_vantage.timeseries import TimeSeries
import yfinance as yf

class ReturnData:
    def __init__(self, config):
        self.config = config
        self.START_DATETIME = datetime.strptime(c.DOWNLOAD_RETURN_START_DATE, '%Y-%m-%d')
        pd.set_option('display.max_columns', None)
        pd.options.mode.chained_assignment = None
        signal_.signal(signal_.SIGALRM, timeout_handler)
        warnings.filterwarnings("ignore")
        return

    def get_universe(self):
        # download from wiki
        wiki_tbl_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        curr_cons = wiki_tbl_list[0] # current constituents
        hist_changes = wiki_tbl_list[1] # historical changes in constituents

        # format columns
        hist_changes.columns = ['_'.join(sorted(set(x))).lower() for x in hist_changes.columns]
        hist_changes['date'] = pd.to_datetime(hist_changes['date'])
        hist_changes['date'] = hist_changes.date + np.timedelta64(-1,'D') # subtract 1 day for ease of backward constituents reconciliation
        hist_changes['added_ticker'] = hist_changes['added_ticker'].fillna('').apply(lambda x: x.replace('.','-'))
        hist_changes['removed_ticker'] = hist_changes['removed_ticker'].fillna('').apply(lambda x: x.replace('.','-'))
        curr_cons['Symbol'] = curr_cons['Symbol'].fillna('').apply(lambda x: x.replace('.','-'))

        # separate into "add" entries and "remove" entries, and then concat into single table
        add_df = hist_changes[['date','added_ticker']].rename(columns={'added_ticker':'symbol'})
        add_df['action'] = 'add'
        remove_df = hist_changes[['date','removed_ticker']].rename(columns={'removed_ticker':'symbol'})
        remove_df['action'] = 'remove'
        hist_changes = pd.concat([add_df, remove_df], axis=0).reset_index(drop=True)

        # remove NA entries
        hist_changes = hist_changes[(hist_changes['symbol'].notnull()) & (hist_changes['symbol']!='')]
        assert hist_changes.isnull().sum().sum() == 0

        # define total stock universe
        all_stocks = sorted(set(curr_cons['Symbol'].tolist() + hist_changes[hist_changes['date']>=c.DOWNLOAD_RETURN_START_DATE]['symbol'].tolist()))

        # generate a table of historical constituents
        # rows as all dates in the study preiod; columns as all stocks ever existed in S&P500 since the start of study period
        # the cell value True/False denoting whether the stock is within S&P500 on that date
        hist_cons = []
        cons = [x in curr_cons['Symbol'].tolist() for x in all_stocks]
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        n_days = (today - self.START_DATETIME).days + 1
        for i in range(n_days):
            date = today + relativedelta(days=-i)
            actions = hist_changes[hist_changes['date']==date]
            if i > 0 and len(actions) > 0:
                for i in range(len(actions)):
                    symbol = actions.iloc[i,:]['symbol']
                    action = actions.iloc[i,:]['action']
                    if action=='add':
                        cons[all_stocks.index(symbol)] = False
                    elif action=='remove':
                        cons[all_stocks.index(symbol)] = True
            hist_cons.append([date] + cons.copy())
        hist_cons = pd.DataFrame(hist_cons, columns=['date'] + all_stocks)

        # filter dates to within study period
        hist_cons = hist_cons \
                    .loc[lambda x: (x.date >= c.DOWNLOAD_RETURN_START_DATE) & (x.date <= c.DOWNLOAD_RETURN_END_DATE)] \
                    .set_index('date') \
                    .sort_index()

        # filter stocks to within study period
        stocks = hist_cons.sum(axis=0).loc[lambda x: x>0].index.tolist()
        hist_cons = hist_cons[stocks]
        display(hist_cons.head())
        log(f'Shape of historical constituents: {hist_cons.shape}')

        # plot for DQ check
        new_plot()
        hist_cons.sum(axis=1).plot()
        plt.title('Number of constituent stocks per day')
        new_plot()
        hist_cons.sum(axis=0).hist(bins=30)
        plt.title('Distribution of stock life span (days)')

        # save results
        self.hist_cons = hist_cons
        self.curr_cons = curr_cons


    # Use Quandl WIKI API to get all historical price (only up to 27 Mar 2018)
    def get_price_data(self):
        # init
        quandl.ApiConfig.api_key = self.config['quandl_key']
        hist_cons = self.hist_cons

        # download returns
        raw_ret = []
        for yr in hist_cons.index.year.unique().tolist():
            df = quandl.get_table('WIKI/PRICES', 
                                    ticker = ','.join(hist_cons.columns.tolist()), 
                                    qopts = { 'columns': ['ticker', 'date', 'adj_close'] }, 
                                    date = { 'gte': f'{yr}-01-01', 'lte':f'{yr}-12-31' }, 
                                    paginate=True)
            log(f'Downloaded {yr} price data')
            raw_ret.append(df)
        raw_ret = pd.concat(raw_ret)

        # convert to date x stock format
        ret = raw_ret.pivot(index='date', columns='ticker', values='adj_close')
        ret.index = pd.to_datetime(ret.index)
        ret = ret.sort_index()
        ret = ret[sorted(ret.columns.tolist())]

        # remove market holidays
        ret = ret.loc[lambda x: ~x.index.isin(c.MARKET_HOLIDAYS)]

        # mask against historical constituents
        msk = hist_cons.reindex(index=ret.index, columns=ret.columns)
        ret = ret.mask(~msk)

        # align trading days across return and hist_cons
        hist_cons = hist_cons.reindex(ret.index)

        # compare number of stocks per day
        new_plot()
        plt.plot(hist_cons.sum(axis=1))
        plt.plot(ret.notnull().sum(axis=1))
        plt.grid()
        plt.title('Downloaded vs Expected stocks per day')

        # list down the missing stocks
        missing_stocks = [x for x in hist_cons.columns.tolist() if x not in ret.columns.tolist()]
        log(f'Number of missing stocks: {len(missing_stocks)}')
        log(f'List of missing stocks: {", ".join(missing_stocks)}')

        # DQ
        log(f'Shape of return is {ret.shape}')
        log(f'Min date: {ret.index.min()}, Max date: {ret.index.max()}')

        # list of stock and date missing from Quandl
        missing = []
        for stock in hist_cons.columns:
            if stock not in ret.columns:
                missing.append((stock, str(ret.index.min())[:10], str(ret.index.max())[:10]))
            else:
                date_list = hist_cons[stock].loc[lambda x: x==True].index.tolist()
                missing_date_list = ret[stock].loc[lambda x: (x.index.isin(date_list)) & (x.isnull())].index.tolist()
                if len(missing_date_list) >= 10:
                    missing.append((stock, str(min(missing_date_list))[:10], str(max(missing_date_list))[:10]))
        missing_df = pd.DataFrame(missing, columns=['stock','start','end']).assign(diff=lambda x: (pd.to_datetime(x.end)-pd.to_datetime(x.start)).dt.days)
        log(f'Shape of missing_df: {missing_df.shape}')
        display(missing_df.sort_values('diff').head())
        display(missing_df['diff'].hist(bins=15))

        # download yahoo for missing data
        ret_yf = []
        for stock, start, end in missing:
            df = yf.download(stock, start=start, end=end, progress=False) \
                    .reset_index() \
                    .assign(stock = stock) \
                    .rename(columns={'Date':'date', 'Adj Close':'adj_close'}) \
                    .loc[:,['date','stock','adj_close']] \
                    .reset_index(drop=True)
            log(f'{stock}: start={start}, end={end}, records={df.shape[0]}')
            ret_yf.append(df)
        ret_yf = pd.concat(ret_yf)
        ret_yf = ret_yf.pivot(index='date', columns='stock', values='adj_close')
        ret_yf.index = pd.to_datetime(ret_yf.index)
        ret_yf = ret_yf.sort_index()
        ret_yf = ret_yf[sorted(ret_yf.columns.tolist())]

        # add extra price from yahoo
        for stock in ret_yf.columns.tolist():
            if stock not in ret.columns.tolist():
                ret[stock] = np.nan
            replace_idx = ret.loc[lambda x: x[stock].isnull()].index.intersection(ret_yf.loc[lambda x: x[stock].notnull()].index)
            ret.loc[replace_idx, stock] = ret_yf.loc[replace_idx, stock]
        ret = ret[sorted(ret.columns.tolist())]
        msk = hist_cons.reindex(index=ret.index, columns=ret.columns)
        ret = ret.mask(~msk)

        # compare number of stocks per day
        new_plot()
        plt.plot(hist_cons.sum(axis=1))
        plt.plot(ret.notnull().sum(axis=1))
        plt.grid()
        plt.title('Downloaded vs Expected stocks per day')
        new_plot()
        plt.plot(hist_cons.sum(axis=1)-ret.notnull().sum(axis=1))
        plt.grid()
        plt.title('Shortfall of stock per day')

        # list down the missing stocks
        missing_stocks = [x for x in hist_cons.columns.tolist() if x not in ret.columns.tolist()]
        log(f'Number of missing stocks: {len(missing_stocks)}')
        log(f'List of missing stocks: {", ".join(missing_stocks)}')

        # DQ
        log(f'Shape of return is {ret.shape}')
        log(f'Min date: {ret.index.min()}, Max date: {ret.index.max()}')

        # list of stock and date missing from Quandl & Yahoo
        missing = []
        for stock in hist_cons.columns:
            if stock not in ret.columns:
                missing.append((stock, str(ret.index.min())[:10], str(ret.index.max())[:10]))
            else:
                date_list = hist_cons[stock].loc[lambda x: x==True].index.tolist()
                missing_date_list = ret[stock].loc[lambda x: (x.index.isin(date_list)) & (x.isnull())].index.tolist()
                if len(missing_date_list) >= 10:
                    missing.append((stock, str(min(missing_date_list))[:10], str(max(missing_date_list))[:10]))
        missing_df = pd.DataFrame(missing, columns=['stock','start','end']).assign(diff=lambda x: (pd.to_datetime(x.end)-pd.to_datetime(x.start)).dt.days)
        log(f'Shape of missing_df: {missing_df.shape}')
        display(missing_df.sort_values('diff').head())
        display(missing_df['diff'].hist(bins=15))

        '''
        Get missing price data from Alpha Vantage
        '''
        ret_av = []
        ts = TimeSeries(key=self.config['av_key'], output_format='pandas')
        for stock, start, end in missing:
            try:
                df, _ = ts.get_daily(symbol=stock, outputsize='full')
                df = df \
                    .assign(stock=stock) \
                    .rename(columns={'4. close':'close'}) \
                    .reset_index() \
                    .loc[:,['date','stock','close']] \
                    .assign(date = lambda x: pd.to_datetime(x.date))
                log(f'{stock}: start={start}, end={end}, records={df.shape[0]}')
                ret_av.append(df)
                time.sleep(12)
            except:
                time.sleep(12)
                continue
        ret_av = pd.concat(ret_av)
        ret_av = ret_av.pivot(index='date', columns='stock', values='close')
        ret_av.index = pd.to_datetime(ret_av.index)
        ret_av = ret_av.sort_index()
        ret_av = ret_av[sorted(ret_av.columns.tolist())]

        # add extra price from AV
        for stock in ret_av.columns.tolist():
            if stock not in ret.columns.tolist():
                ret[stock] = np.nan
            replace_idx = ret.loc[lambda x: x[stock].isnull()].index.intersection(ret_av.loc[lambda x: x[stock].notnull()].index)
            ret.loc[replace_idx, stock] = ret_av.loc[replace_idx, stock]
        ret = ret[sorted(ret.columns.tolist())]
        msk = hist_cons.reindex(index=ret.index, columns=ret.columns)
        ret = ret.mask(~msk)

        # compare number of stocks per day
        new_plot()
        plt.plot(hist_cons.sum(axis=1))
        plt.plot(ret.notnull().sum(axis=1))
        plt.grid()
        plt.title('Downloaded vs Expected stocks per day')
        new_plot()
        plt.plot(hist_cons.sum(axis=1)-ret.notnull().sum(axis=1))
        plt.grid()
        plt.title('Shortfall of stock per day')

        # list down the missing stocks
        missing_stocks = [x for x in hist_cons.columns.tolist() if x not in ret.columns.tolist()]
        log(f'Number of missing stocks: {len(missing_stocks)}')
        log(f'List of missing stocks: {", ".join(missing_stocks)}')

        # DQ
        log(f'Shape of return is {ret.shape}')
        log(f'Min date: {ret.index.min()}, Max date: {ret.index.max()}')

        # # Final price check
        # check stocks that does not populate all days in life span
        df = []
        for stock in ret.columns.tolist():
            s = ret[stock].loc[lambda x: x.notnull()].index
            min_dt, max_dt = s.min(), s.max()
            date_span = ret[stock].loc[lambda x: (x.index >= min_dt) & (x.index <= max_dt)].shape[0]
            date_count = ret[stock].notnull().sum()
            df.append((stock, date_count/date_span))
        df = pd.DataFrame(df, columns=['stock','ratio'])
        df.ratio.hist(bins=20)
        display(df.loc[lambda x: x.ratio < 0.999].sort_values('ratio'))

        # visualise single stock to check continuity
        stock = 'CBE'
        ret[stock].plot()
        print(ret[stock].count())

        # remove low data quality stock
        rm = ['RX','WFR']
        ret = ret.drop(rm, axis=1)

        # save results
        self.ret = ret
        self.hist_cons = hist_cons

    def cal_returns(self):
        
        # init
        ret = self.ret
        hist_cons = self.hist_cons

        # convert price to return
        msk = hist_cons.reindex(index=ret.index, columns=ret.columns)
        ret = ret \
                .sort_index() \
                .pct_change() \
                .mask(~msk)

        # check extreme values
        s = pd.Series(ret.values.flatten())
        s = s.loc[lambda x: x.notnull()]
        q = c.Q_TAIL
        display(s.quantile([q, 1-q]))

        # clipping
        ratio_thsld = config['ratio_thsld']
        clip_thsld = config['clip_thsld']
        ret.loc[:,:] = np.select([ret > ratio_thsld-1, ret < 1/ratio_thsld-1, ret > clip_thsld, ret < -clip_thsld, True], [0, 0, clip_thsld, -clip_thsld, ret])
        display(ret.head())
        pd.Series(ret.values.flatten()).hist(bins=100)

        # # Calculate Beta and Excess Returns
        # download SPY
        df = yf.download('SPY', start=c.DOWNLOAD_RETURN_START_DATE, end=c.DOWNLOAD_RETURN_END_DATE, progress=False) \
                .reset_index() \
                .assign(stock = 'SPY') \
                .rename(columns={'Date':'date', 'Adj Close':'adj_close'}) \
                .loc[:,['date','stock','adj_close']] \
                .reset_index(drop=True)
        spy = df.pivot(index='date', columns='stock', values='adj_close')
        spy.index = pd.to_datetime(spy.index)
        spy = spy.sort_index().pct_change()

        # align stock and SPY dates
        dates = (spy['SPY'].loc[lambda x: x.notnull()].index & ret.index).sort_values()
        ret = ret.reindex(index=dates)
        spy = spy.reindex(index=dates)

        log(f'Shape of SPY: {spy.shape}')
        log(f'Number of nulls of SPY: {spy.SPY.isnull().sum()}')

        # add SPY to returns table
        ret = ret.merge(spy, how='inner', left_index=True, right_index=True)


        def get_beta(ret, n_day, window):
            '''ret is a returns table with last columns as SPY'''
            
            # convert return to n-day horizon
            ret = ret.rolling(n_day).sum()
            
            # individual stock volatility
            stacked_vols = ret.rolling(window, min_periods=window//2).std() \
                .clip(config['stock_volatility_min'], config['stock_volatility_max']) \
                .stack().reset_index() \
                .set_axis(['date', 'stock', 'stock_vol'], axis=1)
            
            # correlation between stock and market
            stacked_corrs = ret['SPY'] \
                .rolling(window, min_periods=window//2) \
                .corr(other=ret, pairwise=False) \
                .stack().reset_index() \
                .set_axis(['date', 'stock', 'correl'], axis=1)
                
            # beta calculation
            beta = stacked_vols \
                .merge(stacked_corrs, how='inner', on=['date', 'stock']) \
                .set_index('date')
            market_vol = pd.DataFrame(beta.loc[lambda x: x.stock=='SPY']['stock_vol']).rename(columns={'stock_vol':'market_vol'})
            beta = beta.merge(market_vol, how='inner', left_index=True, right_index=True)
            beta['beta'] = beta['correl'] * beta['stock_vol'] / beta['market_vol']
            beta = beta.loc[lambda x: x.stock!='SPY']
            beta = beta.reset_index().pivot('date','stock','beta')
            return beta
            
            
        def get_excess_return(ret, beta, n_day):
            '''ret is a returns table with last columns as SPY'''
            
            # convert return to n-day horizon
            ret = ret.rolling(n_day).sum()
            
            # calculate excess return
            beta = beta.merge(ret['SPY'], how='inner', left_index=True, right_index=True)
            pred_ret = beta.drop('SPY', axis=1).multiply(beta['SPY'], axis=0)
            ret, pred_ret = align_index([ret, pred_ret])
            exret = ret - pred_ret
            return ret, exret

        # calculate beta for all investment horizons
        horizons = config['horizons']
        betas = {}
        for h in horizons:
            betas[h] = get_beta(ret=ret, n_day=horizons[h], window=config['beta_window'])
            
        # get 1-day excess return
        ret, exret = get_excess_return(ret=ret, beta=betas['1d'], n_day=horizons['1d'])

        # check market PnL
        spy.cumsum().plot()

        # check beta distribution
        for h in ['1d','1w','1m']:
            new_plot()
            betas[h].iloc[-1].hist(bins=20)

        # cumulative returns before removing market return
        sample_stock = random.sample(ret.columns.tolist(),20)
        new_plot()
        for c in sample_stock:
            ret[c].cumsum().plot()
            
        # cumulative returns after removing market return
        sample_stock = random.sample(exret.columns.tolist(),20)
        new_plot()
        for c in sample_stock:
            exret[c].cumsum().plot()

        # check excess return is uncorrelated with market return
        corr = []
        for stock in ret:
            corr.append(pd.concat([ret[stock], spy.SPY], axis=1).dropna().corr().iloc[0,1])
        new_plot()
        pd.Series(corr).hist(bins=50)

        corr = []
        for stock in exret:
            corr.append(pd.concat([exret[stock], spy.SPY], axis=1).dropna().corr().iloc[0,1])
        new_plot()
        pd.Series(corr).hist(bins=50)

        # export
        ret.to_csv('ret.csv')
        exret.to_csv('exret.csv')
        save_pkl(spy, 'spy')
        save_pkl(betas, 'betas')




