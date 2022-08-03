# import project modules
from src.util import *
import constants as const

# import libraries
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from IPython.display import display
from joblib import Parallel, delayed
import requests
import edgar
import time
import yaml


def get_stock_cik_map():
    # current S&P500 CIK mapping based on wikipedia
    wiki_tbl_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    curr_cons = wiki_tbl_list[0] \
                .assign(stock = lambda x: x.Symbol,
                        cik = lambda x: x.CIK.astype(str).str.zfill(10)) \
                .loc[:,['stock','cik']]

    # Official Edgar Symbol-to-CIK mapping
    cik_map = pd.read_csv('https://www.sec.gov/include/ticker.txt', sep='\t', names=['stock','cik']) \
                    .assign(stock = lambda x: x.stock.str.upper(),
                            cik = lambda x: x.cik.astype(str).str.zfill(10))

    # combine the two sources
    cik_map = pd.concat([cik_map, curr_cons.loc[lambda x: ~x.stock.isin(cik_map.stock)]], axis=0).drop_duplicates()

    # load full stock list based on returns table
    ret = pd.read_csv(f'{const.DATA_OUTPUT_PATH}/ret.csv')
    ret = ret.set_index('date')
    ret.index = pd.to_datetime(ret.index)

    # derive EDGAR filing start (ret start date - 400 days) and end date
    df = []
    for stock in ret:
        s = ret[stock].loc[lambda x: x.notnull()].index
        df.append((stock, s.min(), s.max()))
    df = pd.DataFrame(df, columns=['stock','start_date','end_date'])
    df['start_date'] = df['start_date'] + np.timedelta64(-365*2,'D')

    # map to CIK
    stock_map = df.merge(cik_map, how='left', on='stock')
    assert stock_map.stock.nunique()==stock_map.shape[0]

    # import manual mapping
    manual_cik_map = pd.read_csv(f'{const.DATA_INPUT_PATH}/missing_stock_map.csv') \
        .assign(cik = lambda x: x.cik.astype(str).str.zfill(10)) \
        .rename(columns={'cik':'missing_cik'}) \
        .loc[:,['stock','missing_cik']]

    # fill in missing CIK
    stock_map = stock_map.merge(manual_cik_map, how='left', on='stock') \
        .assign(cik = lambda x: np.select([(x.cik.isnull()) & (x.missing_cik.notnull()), True],[x.missing_cik, x.cik])) \
        .drop('missing_cik', axis=1)

    # check if still missing any CIK
    log(f'Missing CIK:')
    display(stock_map.loc[lambda x: x.cik.isnull()])
    log(f'Printing stock_map')
    display(stock_map.head())
    return stock_map


def download_master_index(stock_map, config):
    # download all index
    edgar.download_index(dest=f'{const.DATA_OUTPUT_PATH}/', since_year=config['edgar_index_start_year'], user_agent=config['edgar_user_agent'], skip_all_present_except_last=False)

    # combin index
    master_idx = []
    for f in os.listdir(f'{const.DATA_OUTPUT_PATH}/'):
        if '.tsv' in f:
            df = pd.read_csv(f'{const.DATA_OUTPUT_PATH}/{f}', sep='|', names=['cik','entity','filing_type','filing_date','full_submission_filename','index_url'])
            master_idx.append(df)
            os.remove(f'{const.DATA_OUTPUT_PATH}/{f}')
    master_idx = pd.concat(master_idx)

    # cleaning and filter with only filings required
    master_idx = master_idx \
        .assign(cik = lambda x: x.cik.astype(str).str.zfill(10),
                filing_date = lambda x: pd.to_datetime(x.filing_date)) \
        .merge(stock_map, how='inner', on='cik') \
        .loc[lambda x: (x.filing_date >= x.start_date) & (x.filing_date <= x.end_date) & (x.filing_type=='10-K')] \
        .reset_index(drop=True)

    # remove stocks with only 1 filing
    ciks = master_idx.groupby('cik')['full_submission_filename'].nunique().loc[lambda x: x==1].index
    master_idx = master_idx.loc[lambda x: ~x.cik.isin(ciks)]

    # save the CIK-stock mapping
    cik_map = master_idx[['cik','stock']].drop_duplicates()

    # final clean
    master_idx = master_idx \
        .drop(['stock','start_date','end_date'], axis=1) \
        .drop_duplicates() \
        .sort_values(['cik','filing_date']) \
        .reset_index(drop=True)

    # get full html link
    results = Parallel(n_jobs=-1)(delayed(get_html_link)(i, master_idx.iloc[i]['full_submission_filename'], master_idx.iloc[i]['index_url']) for i in range(len(master_idx)))
    results = pd.DataFrame(results, columns=['i','url_10k']).set_index('i')
    master_idx = master_idx.merge(results, how='left', left_index=True, right_index=True)

    # remove nulls and pdf
    log(f'Percentage of null: {master_idx["url_10k"].isnull().sum() / master_idx.shape[0]}')
    log(f'Percentage of PDF: {(master_idx["url_10k"].fillna("").str.lower().str[-3:]=="pdf").sum() / master_idx.shape[0]}')
    master_idx = master_idx.loc[lambda x: (x.url_10k.fillna('').str.lower().str[-3:].isin(['htm','tml']))].reset_index(drop=True)

    # check again CIK with single doc
    ciks = master_idx.groupby('cik')['filing_date'].count().loc[lambda x: x<2].index.tolist()
    master_idx = master_idx.loc[lambda x: ~x.cik.isin(ciks)].reset_index(drop=True)

    # assign doc_id
    master_idx = master_idx \
        .assign(doc_id = lambda x: x.cik + '_' + x.filing_date.apply(lambda y: str(y)[:10].replace('-',''))) \
        .sort_values('doc_id') \
        .reset_index(drop=True)
    assert master_idx.doc_id.nunique()==master_idx.shape[0]
    log(f'Shape of master_idx: {master_idx.shape}')
    log(f'Avg number of filing per stock: {master_idx.shape[0] / master_idx.cik.nunique()}')
    log(f'Sample master_idx:')
    display(master_idx.sample(5))
    log(f'Shape of cik_map: {cik_map.shape}')
    return cik_map, master_idx


# function to contruct full 10-K HTML URLs
def get_html_link(i, full_submission_filename, index_url):
    time.sleep(0.1)
    try: 
        # get 10-K document name
        url = f'https://www.sec.gov/Archives/{index_url}'
        html = requests.get(url, headers={"user-agent": f"chan_tai_man_{int(float(np.random.rand(1)) * 1e7)}@gmail.com"}).content
        doc_name = pd.read_html(html)[0] \
            .loc[lambda x: x.Type=='10-K'] \
            .sort_values('Size', ascending=False) \
            .Document \
            .tolist()[0]

        # construct full URL
        filing_id = full_submission_filename.replace('.txt','').replace('-','')
        full_url = f'https://www.sec.gov/Archives/{filing_id}/{doc_name}'
    except:
        full_url = None
    
    log(f'[{i}] {full_url}') if i%200==0 else None
    return i, full_url

# class to build master index
class MasterIndex:

    def __init__(self):
        self.config = yaml.safe_load(open('config.yml'))

    def build_master_idx(self):
        self.stock_map = get_stock_cik_map()
        self.cik_map, self.master_idx = download_master_index(self.stock_map, self.config)

    def export(self):
        self.stock_map.to_csv(f'{const.DATA_OUTPUT_PATH}/stock_map.csv', index=False)
        self.cik_map.to_csv(f'{const.DATA_OUTPUT_PATH}/cik_map.csv', index=False)
        self.master_idx.to_csv(f'{const.DATA_OUTPUT_PATH}/master_idx.csv', index=False)
    