# import project modules
from src.util import *
import src.constants as const

# import libraries
import numpy as np
import pandas as pd
from IPython.display import display
from joblib import Parallel, delayed
import requests
# import edgar
import time
import yaml
import zipfile
import tempfile
import io
import multiprocessing


# class to build master index
class MasterIndex:

    def __init__(self):
        self.config = yaml.safe_load(open('config.yml'))

    def get_stock_cik_map(self):
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
        ret = load_pkl(f'{const.INTERIM_DATA_PATH}/ret.pkl')

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
        manual_cik_map = pd.read_csv(f'{const.INPUT_DATA_PATH}/missing_stock_map.csv') \
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
        
        # save results
        self.stock_map = stock_map


    def download_master_index(self):
        # download all index
        years = range(self.config['edgar_index_start_year'], self.config['edgar_index_end_year'] + 1)
        quarters = ["QTR1", "QTR2", "QTR3", "QTR4"]
        history = list((y, q) for y in years for q in quarters)
        history.reverse()
        tasks = [("https://www.sec.gov/Archives/edgar/full-index/%s/%s/master.zip" % (x[0], x[1]),
                "%s-%s.tsv" % (x[0], x[1]),)
                for x in history]
        for i, file in enumerate(tasks):
            log(f'Downloading index: {file[0]} >> {file[1]}')
            _download(file, const.INTERIM_DATA_PATH, self.config['edgar_user_agent'])

        # combin index
        master_idx = []
        for f in os.listdir(f'{const.INTERIM_DATA_PATH}/'):
            if '.tsv' in f:
                df = pd.read_csv(f'{const.INTERIM_DATA_PATH}/{f}', sep='|', names=['cik','entity','filing_type','filing_date','full_submission_filename','index_url'])
                master_idx.append(df)
                os.remove(f'{const.INTERIM_DATA_PATH}/{f}')
        master_idx = pd.concat(master_idx)

        # cleaning and filter with only filings required
        master_idx = master_idx \
            .assign(cik = lambda x: x.cik.astype(str).str.zfill(10),
                    filing_date = lambda x: pd.to_datetime(x.filing_date)) \
            .merge(self.stock_map, how='inner', on='cik') \
            .loc[lambda x: (x.filing_date >= x.start_date) & (x.filing_date <= x.end_date)] \
            .reset_index(drop=True)

        # if duplicate, take last entry
        master_idx = master_idx \
            .sort_values(['filing_type','cik','filing_date','full_submission_filename']) \
            .groupby(['filing_type','cik','filing_date']) \
            .last() \
            .reset_index()

        # remove stocks with only 1 10-k or 10-Q filing
        ciks_10k = master_idx.loc[lambda x: x.filing_type=='10-K'].groupby('cik')['full_submission_filename'].nunique().loc[lambda x: x==1].index
        ciks_10q = master_idx.loc[lambda x: x.filing_type=='10-Q'].groupby('cik')['full_submission_filename'].nunique().loc[lambda x: x==1].index
        master_idx = master_idx.loc[lambda x: (~x.cik.isin(ciks_10k)) & (~x.cik.isin(ciks_10q))]

        # save the CIK-stock mapping
        cik_map = master_idx[['cik','stock']].drop_duplicates()

        # final clean
        master_idx = master_idx \
            .drop(['stock','start_date','end_date'], axis=1) \
            .drop_duplicates() \
            .sort_values(['filing_type','cik','filing_date']) \
            .reset_index(drop=True)

        # separate 10-K, 10-Q, 8-K
        master_idx_10q = master_idx.loc[lambda x: x.filing_type=='10-Q'].reset_index(drop=True)
        master_idx_8k = master_idx.loc[lambda x: x.filing_type=='8-K'].reset_index(drop=True)
        master_idx_10k = master_idx.loc[lambda x: x.filing_type=='10-K'].reset_index(drop=True)

        log(f'Shape of 10-K master_idx: {master_idx_10k.shape}')
        log(f'Shape of 10-Q master_idx: {master_idx_10q.shape}')
        log(f'Shape of 8-K master_idx: {master_idx_8k.shape}')
        log(f'Avg number of 10-K filing per stock: {master_idx_10k.shape[0] / master_idx_10k.cik.nunique()}')
        log(f'Avg number of 10-Q filing per stock: {master_idx_10q.shape[0] / master_idx_10q.cik.nunique()}')
        log(f'Avg number of 8-K filing per stock: {master_idx_8k.shape[0] / master_idx_8k.cik.nunique()}')
        log(f'Sample of master_idx_10k:')
        display(master_idx_10k.sample(3))
        log(f'Distribution of number of 10-K per CIK:')
        display(master_idx_10k.groupby('cik')['full_submission_filename'].nunique().value_counts())
        log(f'Sample of master_idx_10q:')
        display(master_idx_10q.sample(3))
        log(f'Distribution of number of 10-Q per CIK:')
        display(master_idx_10q.groupby('cik')['full_submission_filename'].nunique().value_counts())
        log(f'Sample of master_idx_8k:')
        display(master_idx_8k.sample(3))
        log(f'Distribution of number of 8-K per CIK:')
        display(master_idx_8k.groupby('cik')['full_submission_filename'].nunique().value_counts())

        # save results
        self.cik_map = cik_map
        self.master_idx_10k = master_idx_10k
        self.master_idx_10q = master_idx_10q
        self.master_idx_8k = master_idx_8k


    # function to contruct full 10-K HTML URLs
    def get_html_link(self, arg):
        i, full_submission_filename, index_url, type = arg
        time.sleep(0.1)
        try: 
            # get 10-K document name
            url = f'https://www.sec.gov/Archives/{index_url}'
            head = {'Host': 'www.sec.gov', 'Connection': 'close',
                    'Accept': 'application/json, text/javascript, */*; q=0.01', 'X-Requested-With': 'XMLHttpRequest',
                    'User-Agent': f"{self.config['edgar_user_agent']}{int(float(np.random.rand(1)) * 1e7)}"
                    }
            html = requests.get(url, headers=head).content
            doc_name = pd.read_html(html)[0] \
                .loc[lambda x: x.Type==type] \
                .sort_values('Size', ascending=False) \
                .Document \
                .tolist()[0]

            # construct full URL
            filing_id = full_submission_filename.replace('.txt','').replace('-','')
            full_url = f'https://www.sec.gov/Archives/{filing_id}/{doc_name}'
        except:
            full_url = None
        
        log(f'[{i}] Constructed URL: {full_url}') if (i%200==0 and i>0) else None
        return i, full_url


    def append_full_html_link_10k(self): 
        master_idx_10k = self.master_idx_10k
        pool = multiprocessing.Pool(processes=self.config['n_jobs'])
        results = pool.map(self.get_html_link, zip(list(range(len(master_idx_10k))), 
                                                    master_idx_10k.full_submission_filename.tolist(), 
                                                    master_idx_10k.index_url.tolist(), 
                                                    ['10-K']*len(master_idx_10k)))
        results = pd.DataFrame(results, columns=['i','url_10k']).set_index('i')
        master_idx_10k = master_idx_10k.merge(results, how='left', left_index=True, right_index=True)

        # remove nulls and pdf
        log(f'Percentage of null (10-K): {master_idx_10k["url_10k"].isnull().sum() / master_idx_10k.shape[0]}')
        log(f'Percentage of PDF (10-K): {(master_idx_10k["url_10k"].fillna("").str.lower().str[-3:]=="pdf").sum() / master_idx_10k.shape[0]}')
        master_idx_10k = master_idx_10k.loc[lambda x: (x.url_10k.fillna('').str.lower().str[-3:].isin(['htm','tml']))].reset_index(drop=True)

        # check again CIK with single doc
        ciks = master_idx_10k.groupby('cik')['filing_date'].count().loc[lambda x: x<2].index.tolist()
        log(f'Number of CIK with single doc (10-K): {len(ciks)}')
        master_idx_10k = master_idx_10k.loc[lambda x: ~x.cik.isin(ciks)].reset_index(drop=True)

        # assign doc_id
        master_idx_10k = master_idx_10k \
            .assign(doc_id = lambda x: x.cik + '_' + x.filing_date.apply(lambda y: str(y)[:10].replace('-',''))) \
            .sort_values('doc_id') \
            .reset_index(drop=True)

        # logging
        assert master_idx_10k.doc_id.nunique()==master_idx_10k.shape[0]
        log(f'Shape of master_idx_10k: {master_idx_10k.shape}')
        log(f'master_idx_10k')
        display(master_idx_10k.sample(5))

        # save results
        self.master_idx_10k = master_idx_10k

    def append_full_html_link_10q(self): 
        master_idx_10q = self.master_idx_10q
        pool = multiprocessing.Pool(processes=self.config['n_jobs'])
        results = pool.map(self.get_html_link, zip(list(range(len(master_idx_10q))), 
                                                    master_idx_10q.full_submission_filename.tolist(), 
                                                    master_idx_10q.index_url.tolist(), 
                                                    ['10-Q']*len(master_idx_10q)))
        results = pd.DataFrame(results, columns=['i','url_10q']).set_index('i')
        master_idx_10q = master_idx_10q.merge(results, how='left', left_index=True, right_index=True)

        # remove nulls and pdf
        log(f'Percentage of null (10-Q): {master_idx_10q["url_10q"].isnull().sum() / master_idx_10q.shape[0]}')
        log(f'Percentage of PDF (10-Q): {(master_idx_10q["url_10q"].fillna("").str.lower().str[-3:]=="pdf").sum() / master_idx_10q.shape[0]}')
        master_idx_10q = master_idx_10q.loc[lambda x: (x.url_10q.fillna('').str.lower().str[-3:].isin(['htm','tml']))].reset_index(drop=True)

        # check again CIK with single doc
        ciks = master_idx_10q.groupby('cik')['filing_date'].count().loc[lambda x: x<2].index.tolist()
        log(f'Number of CIK with single doc (10-Q): {len(ciks)}')
        master_idx_10q = master_idx_10q.loc[lambda x: ~x.cik.isin(ciks)].reset_index(drop=True)

        # assign doc_id
        master_idx_10q = master_idx_10q \
            .assign(doc_id = lambda x: x.cik + '_' + x.filing_date.apply(lambda y: str(y)[:10].replace('-',''))) \
            .sort_values('doc_id') \
            .reset_index(drop=True)

        # logging
        assert master_idx_10q.doc_id.nunique()==master_idx_10q.shape[0]
        log(f'Shape of master_idx_10q: {master_idx_10q.shape}')
        log(f'Sample of master_idx_10q:')
        display(master_idx_10q.sample(5))

        # save results
        self.master_idx_10q = master_idx_10q


    def export(self):
        save_pkl(self.stock_map, f'{const.INTERIM_DATA_PATH}/stock_map.pkl')
        save_pkl(self.cik_map, f'{const.INTERIM_DATA_PATH}/cik_map.pkl')
        save_pkl(self.master_idx_10k, f'{const.INTERIM_DATA_PATH}/master_idx_10k.pkl')
        save_pkl(self.master_idx_10q, f'{const.INTERIM_DATA_PATH}/master_idx_10q.pkl')
        save_pkl(self.master_idx_8k, f'{const.INTERIM_DATA_PATH}/master_idx_8k.pkl')



def _append_html_version(line):
    sep = "|"
    chunks = line.split(sep)
    return line + sep + chunks[-1].replace(".txt", "-index.html")


def _skip_header(f):
    for x in range(0, 11):
        f.readline()


def _url_get(url, user_agent):
    heads = {'Host': 'www.sec.gov', 'Connection': 'close',
         'Accept': 'application/json, text/javascript, */*; q=0.01', 'X-Requested-With': 'XMLHttpRequest',
         'User-Agent': user_agent,
         }
    content = requests.get(url, headers=heads).content
    return content


def _download(file, dest, user_agent):
    """
    Download an idx archive from EDGAR
    This will read idx files and unzip
    archives + read the master.idx file inside
    """
    if not dest.endswith("/"):
        dest = "%s/" % dest

    url = file[0]
    dest_name = file[1]

    if url.endswith("zip"):
        with tempfile.TemporaryFile(mode="w+b") as tmp:
            tmp.write(_url_get(url, user_agent))
            with zipfile.ZipFile(tmp).open("master.idx") as z:
                with io.open(dest + dest_name, "w+", encoding="utf-8") as idxfile:
                    _skip_header(z)
                    lines = z.read()
                    lines = lines.decode("latin-1")
                    lines = map(
                        lambda line: _append_html_version(line), lines.splitlines()
                    )
                    idxfile.write("\n".join(lines)+"\n")
                    logging.info("> downloaded %s to %s%s" % (url, dest, dest_name))
    else:
        raise logging.error("python-edgar only supports zipped index files")

    