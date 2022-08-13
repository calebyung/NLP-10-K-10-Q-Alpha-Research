# import project modules
from src.util import *
import src.constants as const

# import other packages
import os
import numpy as np
from datetime import datetime
import time
from dateutil.relativedelta import relativedelta
from IPython.display import display
from matplotlib import pyplot as plt
import pandas as pd
import yaml



class Universe:
    def __init__(self):
        self.config = yaml.safe_load(open('config.yml'))
        self.START_DATETIME = datetime.strptime(const.DOWNLOAD_RETURN_START_DATE, '%Y-%m-%d')
        return

    def get_wiki_tables(self):
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

        # save results
        self.curr_cons = curr_cons
        self.hist_changes = hist_changes

    def get_hist_cons(self):
        # load data
        curr_cons = self.curr_cons
        hist_changes = self.hist_changes

        # define total stock universe
        all_stocks = sorted(set(curr_cons['Symbol'].tolist() + hist_changes[hist_changes['date']>=const.DOWNLOAD_RETURN_START_DATE]['symbol'].tolist()))

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
                    .loc[lambda x: (x.date >= const.DOWNLOAD_RETURN_START_DATE) & (x.date <= const.DOWNLOAD_RETURN_END_DATE)] \
                    .set_index('date') \
                    .sort_index()

        # filter stocks to within study period
        stocks = hist_cons.sum(axis=0).loc[lambda x: x>0].index.tolist()
        hist_cons = hist_cons[stocks]
        display(hist_cons.head())
        log(f'Shape of historical constituents: {hist_cons.shape}')

        # filter stocks sampled based on config
        hist_cons = hist_cons.iloc[:, :self.config['n_stock_return']]

        # plot for DQ check
        new_plot()
        hist_cons.sum(axis=1).plot()
        plt.title('Number of constituent stocks per day')
        plt.show()
        plt.close()
        new_plot()
        hist_cons.sum(axis=0).hist(bins=30)
        plt.title('Distribution of stock life span (days)')
        plt.show()
        plt.close()

        # save results
        self.hist_cons = hist_cons
        self.curr_cons = curr_cons

    def export(self):
        save_pkl(self.curr_cons, f'{const.INTERIM_DATA_PATH}/curr_cons.pkl')
        save_pkl(self.hist_changes, f'{const.INTERIM_DATA_PATH}/hist_changes.pkl')
        save_pkl(self.hist_cons, f'{const.INTERIM_DATA_PATH}/hist_cons.pkl')