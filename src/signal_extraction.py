# import project modules
from src.util import *
import constants as const

# import libraries
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from bs4 import BeautifulSoup
import re
from IPython.display import display
import unicodedata
from joblib import Parallel, delayed
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import edgar
from polyleven import levenshtein
import nltk
from nltk import tokenize
nltk.download('punkt')


mode = ['full','cpu','gpu','wv'][1]

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
    ret = pd.read_csv('./data/ret.csv')
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
    manual_cik_map = pd.read_csv('./data/missing_stock_map.csv') \
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


def download_master_index(stock_map):
    # download all index
    edgar.download_index(dest='./', since_year=config['edgar_index_start_year'], user_agent=config['edgar_user_agent'], skip_all_present_except_last=False)

    # combin index
    master_idx = []
    for f in os.listdir('./data/'):
        if '.tsv' in f:
            df = pd.read_csv(f'./data/{f}', sep='|', names=['cik','entity','filing_type','filing_date','full_submission_filename','index_url'])
            master_idx.append(df)
            os.remove(f'./data/{f}')
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


def remove_unicode1(txt):
    chars = {
        r'[\xc2\x82]' : ',',        # High code comma
         r'[\xc2\x84]' : ',,',       # High code double comma
         r'[\xc2\x85]' : '...',      # Tripple dot
         r'[\xc2\x88]' : '^',        # High carat
         r'[\xc2\x91]' : "'",     # Forward single quote
         r'[\xc2\x92]' : "'",     # Reverse single quote
         r'[\xc2\x93]' : '"',     # Forward double quote
         r'[\xc2\x94]' : '"',     # Reverse double quote
         r'[\xc2\x95]' : ' ',
         r'[\xc2\x96]' : '-',        # High hyphen
         r'[\xc2\x97]' : '--',       # Double hyphen
         r'[\xc2\x99]' : ' ',
         r'[\xc2\xa0]' : ' ',
         r'[\xc2\xa6]' : '|',        # Split vertical bar
         r'[\xc2\xab]' : '<<',       # Double less than
         r'[\xc2\xbb]' : '>>',       # Double greater than
         r'[\xc2\xbc]' : '1/4',      # one quarter
         r'[\xc2\xbd]' : '1/2',      # one half
         r'[\xc2\xbe]' : '3/4',      # three quarters
         r'[\xca\xbf]' : "'",     # c-single quote
         r'[\xcc\xa8]' : '',         # modifier - under curve
         r'[\xcc\xb1]' : '',          # modifier - under line
         r"[\']" : "'"
    }
    for ptrn in chars:
        txt = re.sub(ptrn, chars[ptrn], txt)
    return txt

def remove_unicode2(txt):
    txt = txt. \
        replace('\\xe2\\x80\\x99', "'"). \
        replace('\\xc3\\xa9', 'e'). \
        replace('\\xe2\\x80\\x90', '-'). \
        replace('\\xe2\\x80\\x91', '-'). \
        replace('\\xe2\\x80\\x92', '-'). \
        replace('\\xe2\\x80\\x93', '-'). \
        replace('\\xe2\\x80\\x94', '-'). \
        replace('\\xe2\\x80\\x94', '-'). \
        replace('\\xe2\\x80\\x98', "'"). \
        replace('\\xe2\\x80\\x9b', "'"). \
        replace('\\xe2\\x80\\x9c', '"'). \
        replace('\\xe2\\x80\\x9c', '"'). \
        replace('\\xe2\\x80\\x9d', '"'). \
        replace('\\xe2\\x80\\x9e', '"'). \
        replace('\\xe2\\x80\\x9f', '"'). \
        replace('\\xe2\\x80\\xa6', '...'). \
        replace('\\xe2\\x80\\xb2', "'"). \
        replace('\\xe2\\x80\\xb3', "'"). \
        replace('\\xe2\\x80\\xb4', "'"). \
        replace('\\xe2\\x80\\xb5', "'"). \
        replace('\\xe2\\x80\\xb6', "'"). \
        replace('\\xe2\\x80\\xb7', "'"). \
        replace('\\xe2\\x81\\xba', "+"). \
        replace('\\xe2\\x81\\xbb', "-"). \
        replace('\\xe2\\x81\\xbc', "="). \
        replace('\\xe2\\x81\\xbd', "("). \
        replace('\\xe2\\x81\\xbe', ")")
    return txt

def clean_doc1(txt):

    # remove all special fields e.g. us-gaap:AccumulatedOtherComprehensiveIncomeMember
    txt = re.sub(r'\b' + re.escape('us-gaap:') + r'\w+\b', '', txt)
    txt = re.sub(r'\b\w+[:]\w+\b', '', txt)

    # remove unicode characters
    txt = unicodedata.normalize("NFKD", txt)
    txt = remove_unicode1(txt)
    txt = remove_unicode2(txt)

    # standardize spaces
    txt = txt.replace('\\n',' ').replace('\n',' ').replace('\\t','|').replace('\t','|')
    txt = re.sub(r'\| +', '|', txt)
    txt = re.sub(r' +\|', '|', txt)
    txt = re.sub(r'\|+', '|', txt)
    txt = re.sub(r' +', ' ', txt)
    return txt

# Function to clean txt; applied only after Item extraction
def clean_doc2(txt):
    # lowercase all strings
    txt = txt.lower()
    # replace sep with space
    txt = txt.replace('|',' ')
    # remove tags
    txt = re.sub('<.+>', '', txt)
    # remove unwanted characters, numbers, dots
    txt = re.sub(r'([a-z]+\d+)+([a-z]+)?(\.+)?', '', txt) # aa12bb33. y3y
    txt = re.sub(r'(\d+[a-z]+)+(\d+)?(\.+)?', '', txt) # 1a2b. 1a1a1
    txt = re.sub(r'\b\$?\d+\.(\d+)?', '', txt) # $2.14 999.8 123.
    txt = re.sub(r'\$\d+', '', txt) # $88
    txt = re.sub(r'(\w+\.){2,}(\w+)?', '', txt) # W.C. ASD.ASD.c
    txt = re.sub(r"\bmr\.|\bjr\.|\bms\.|\bdr\.|\besq\.|\bhon\.|\bmrs\.|\bprof\.|\brev\.|\bsr\.|\bst\.|\bno\.", '', txt) # titles and common abbreviations
    txt = re.sub(r'\b[a-z]\.', '', txt) #  L.
    txt = re.sub(r'(\w+)?\.\w+', '', txt) # .net .123 www.123
    txt = re.sub(r'[\$\%\d]+', '', txt) # remove all $/%/numbers
    # final clean format
    txt = re.sub(r'[\.\:\;]', '.', txt) # standardize all sentence separators
    txt = re.sub(r'( ?\. ?)+', '. ', txt) # replace consecutive sentence separators
    txt = re.sub(r' +', ' ', txt) # replace consecutive spaces
    txt = re.sub(r'( ?, ?)+', ', ', txt) # replace consecutive ","
    return txt


# function to convert txt to re pattern allowing any | between characters
def w(txt):
    txt = r''.join([x + r'\|?' for x in list(txt)])
    return txt

def wu(txt):
    txt = r''.join([x + r'\|?' for x in list(txt)])
    return r'(?:' + txt + r'|' + txt.upper() + r')'

def s(x='.'):
    return x + r'{0,5}'

# defining search patterns
def get_item_ptrn1():
    item_ptrn1 = dict()
    item_ptrn1['item_1'] = rf"\|(?:{wu('Item')}{s()}1{s()}){w('Business')}{s('[^a-z]')}\|"
    item_ptrn1['item_1a'] = rf"\|(?:{wu('Item')}{s()}{wu('1a')}{s()}){w('Risk')}{s()}{w('Factors')}{s()}\|"
    item_ptrn1['item_1b'] = rf"\|(?:{wu('Item')}{s()}{wu('1b')}{s()}){w('Unresolved')}{s()}(?:{w('Staff')}|{w('SEC')}){s()}{w('Comment')}{s()}\|"
    item_ptrn1['item_2'] = rf"\|(?:{wu('Item')}{s()}2{s()}){w('Properties')}{s()}\|"
    item_ptrn1['item_3'] = rf"\|(?:{wu('Item')}{s()}3{s()}){w('Legal')}{s()}{w('Proceeding')}{s()}\|"
    item_ptrn1['item_4'] = r'|'.join([rf"(?:\|(?:{wu('Item')}{s()}4{s()}){w('Mine')}{s()}{w('Safety')}{s()}{w('Disclosure')}{s()}\|)", 
                                    rf"(?:\|(?:{wu('Item')}{s()}4{s()}){w('Submission')}{s()}{w('f')}{s()}{w('Matter')}{s()}{w('o')}{s()}{wu('a')}{s()}{w('Vote')}{s()}{w('f')}{s()}{w('Security')}{s()}{w('Holder')}{s()}\|)",
                                    rf"(?:\|(?:{wu('Item')}{s()}4{s()})(?:{w('Removed')}{s()}{w('nd')}{s()})?{w('Reserved')}{s()}\|)"])
    item_ptrn1['item_5'] = rf"\|(?:{wu('Item')}{s()}5{s()}){w('Market')}{s()}{w('or')}{s()}{w('Registrant')}{s()}{w('Common')}{s()}{w('Equit')}(?:{w('y')}|{w('ies')}){s()}{w('Related')}{s()}{w('Stockholder')}{s()}{w('Matter')}{s()}{w('nd')}{s()}{w('Issuer')}{s()}{w('Purchase')}{s()}{w('f')}{s()}{w('Equit')}(?:{w('y')}|{w('ies')}){s()}{w('Securities')}{s()}\|"
    item_ptrn1['item_6'] = rf"\|(?:{wu('Item')}{s()}6{s()}){w('Selected')}{s()}(?:{w('Consolidated')}{s()})?{w('Financial')}{s()}{w('Data')}{s()}\|"
    item_ptrn1['item_7'] = r'|'.join([rf"\|(?:{wu('Item')}{s()}7{s()}){w('Management')}{s()}{w('Discussion')}{s()}{w('nd')}{s()}{w('Analy')}(?:{w('sis')}|{w('ses')}){s()}{w('f')}{s()}{w('Financial')}{s()}{w('Condition')}{s()}{w('nd')}{s()}{w('Result')}{s()}{w('f')}{s()}{w('Operation')}{s()}\|",
                                    rf"\|(?:{wu('Item')}{s()}7{s()}){w('Management')}{s()}{w('Discussion')}{s()}{w('nd')}{s()}{w('Analy')}(?:{w('sis')}|{w('ses')}){s()}{w('f')}{s()}{w('Result')}{s()}{w('f')}{s()}{w('Operation')}{s()}{w('nd')}{s()}{w('Financial')}{s()}{w('Condition')}{s()}\|"])
    item_ptrn1['item_7a'] = r'|'.join([rf"\|(?:{wu('Item')}{s()}{wu('7a')}{s()}){w('Quantitative')}{s()}{w('nd')}{s()}{w('Qualitative')}{s()}{w('Disclosure')}{s()}{w('bout')}{s()}{w('Market')}{s()}{w('Risk')}{s()}\|",
                                    rf"\|(?:{wu('Item')}{s()}{wu('7a')}{s()}){w('Qualitative')}{s()}{w('nd')}{s()}{w('Quantitative')}{s()}{w('Disclosure')}{s()}{w('bout')}{s()}{w('Market')}{s()}{w('Risk')}{s()}\|"])
    item_ptrn1['item_8'] = rf"\|(?:{wu('Item')}{s()}8{s()}){w('Financial')}{s()}{w('Statement')}{s()}{w('nd')}{s()}{w('Supplementary')}{s()}{w('Data')}{s()}\|"
    item_ptrn1['item_9'] = rf"\|(?:{wu('Item')}{s()}9{s()}){w('Change')}{s()}{w('n')}{s()}{w('nd')}{s()}{w('Disagreement')}{s()}{w('ith')}{s()}{w('Accountant')}{s()}{w('n')}{s()}{w('Accounting')}{s()}{w('nd')}{s()}{w('Financial')}{s()}{w('Disclosure')}{s()}\|"
    item_ptrn1['item_9a'] = rf"\|(?:{wu('Item')}{s()}{wu('9a')}{s()}){w('Control')}{s()}{w('nd')}{s()}{w('Procedure')}{s()}\|"
    item_ptrn1['item_9b'] = rf"\|(?:{wu('Item')}{s()}{wu('9b')}{s()}){w('Other')}{s()}{w('Information')}{s()}\|"
    item_ptrn1['item_10'] = rf"\|(?:{wu('Item')}{s()}10{s()}){w('Director')}{s()}{w('Executive')}{s()}{w('Officer')}{s()}{w('nd')}{s()}{w('Corporate')}{s()}{w('Governance')}{s()}\|"
    item_ptrn1['item_11'] = rf"\|(?:{wu('Item')}{s()}11{s()}){w('Executive')}{s()}{w('Compensation')}{s()}\|"
    item_ptrn1['item_12'] = rf"\|(?:{wu('Item')}{s()}12{s()}){w('Security')}{s()}{w('Ownership')}{s()}{w('f')}{s()}{w('Certain')}{s()}{w('Beneficial')}{s()}{w('Owner')}{s()}{w('nd')}{s()}{w('Management')}{s()}{w('nd')}{s()}{w('Related')}{s()}{w('Stockholder')}{s()}{w('Matter')}s?{s()}\|"
    item_ptrn1['item_13'] = rf"\|(?:{wu('Item')}{s()}13{s()}){w('Certain')}{s()}{w('Relationship')}{s()}{w('nd')}{s()}{w('Related')}{s()}{w('Transaction')}{s()}{w('nd')}{s()}{w('Director')}{s()}{w('Independence')}{s()}\|"
    item_ptrn1['item_14'] = rf"\|(?:{wu('Item')}{s()}14{s()}){w('Principal')}{s()}{w('Account')}(?:{w('ant')}|{w('ing')}){s()}{w('Fee')}{s()}{w('nd')}{s()}{w('Service')}{s()}\|"
    item_ptrn1['item_15'] = rf"\|(?:{wu('Item')}{s()}15{s()}){w('Exhibits')}{s()}{w('Financial')}{s()}{w('Statement')}{s()}{w('Schedule')}{s()}\|"
    return item_ptrn1

def get_item_ptrn2():
    item_ptrn2 = dict()
    item_ptrn2['item_1'] = rf"\|(?:{wu('Item')}{s()}1{s()})?{w('Business')}{s('[^a-z]')}\|"
    item_ptrn2['item_1a'] = rf"\|(?:{wu('Item')}{s()}{wu('1a')}{s()})?{w('Risk')}{s()}{w('Factors')}{s()}\|"
    item_ptrn2['item_1b'] = rf"\|(?:{wu('Item')}{s()}{wu('1b')}{s()})?{w('Unresolved')}{s()}(?:{w('Staff')}|{w('SEC')}){s()}{w('Comment')}{s()}\|"
    item_ptrn2['item_2'] = rf"\|(?:{wu('Item')}{s()}2{s()})?{w('Properties')}{s()}\|"
    item_ptrn2['item_3'] = rf"\|(?:{wu('Item')}{s()}3{s()})?{w('Legal')}{s()}{w('Proceeding')}{s()}\|"
    item_ptrn2['item_4'] = r'|'.join([rf"(?:\|(?:{wu('Item')}{s()}4{s()})?{w('Mine')}{s()}{w('Safety')}{s()}{w('Disclosure')}{s()}\|)", 
                                    rf"(?:\|(?:{wu('Item')}{s()}4{s()})?{w('Submission')}{s()}{w('f')}{s()}{w('Matter')}{s()}{w('o')}{s()}{wu('a')}{s()}{w('Vote')}{s()}{w('f')}{s()}{w('Security')}{s()}{w('Holder')}{s()}\|)",
                                    rf"(?:\|(?:{wu('Item')}{s()}4{s()})(?:{w('Removed')}{s()}{w('nd')}{s()})?{w('Reserved')}{s()}\|)"])
    item_ptrn2['item_5'] = rf"\|(?:{wu('Item')}{s()}5{s()})?{w('Market')}{s()}{w('or')}{s()}{w('Registrant')}{s()}{w('Common')}{s()}{w('Equit')}(?:{w('y')}|{w('ies')}){s()}{w('Related')}{s()}{w('Stockholder')}{s()}{w('Matter')}{s()}{w('nd')}{s()}{w('Issuer')}{s()}{w('Purchase')}{s()}{w('f')}{s()}{w('Equit')}(?:{w('y')}|{w('ies')}){s()}{w('Securities')}{s()}\|"
    item_ptrn2['item_6'] = rf"\|(?:{wu('Item')}{s()}6{s()})?{w('Selected')}{s()}(?:{w('Consolidated')}{s()})?{w('Financial')}{s()}{w('Data')}{s()}\|"
    item_ptrn2['item_7'] = r'|'.join([rf"\|(?:{wu('Item')}{s()}7{s()})?{w('Management')}{s()}{w('Discussion')}{s()}{w('nd')}{s()}{w('Analy')}(?:{w('sis')}|{w('ses')}){s()}{w('f')}{s()}{w('Financial')}{s()}{w('Condition')}{s()}{w('nd')}{s()}{w('Result')}{s()}{w('f')}{s()}{w('Operation')}{s()}\|",
                                    rf"\|(?:{wu('Item')}{s()}7{s()})?{w('Management')}{s()}{w('Discussion')}{s()}{w('nd')}{s()}{w('Analy')}(?:{w('sis')}|{w('ses')}){s()}{w('f')}{s()}{w('Result')}{s()}{w('f')}{s()}{w('Operation')}{s()}{w('nd')}{s()}{w('Financial')}{s()}{w('Condition')}{s()}\|"])
    item_ptrn2['item_7a'] = r'|'.join([rf"\|(?:{wu('Item')}{s()}{wu('7a')}{s()})?{w('Quantitative')}{s()}{w('nd')}{s()}{w('Qualitative')}{s()}{w('Disclosure')}{s()}{w('bout')}{s()}{w('Market')}{s()}{w('Risk')}{s()}\|",
                                    rf"\|(?:{wu('Item')}{s()}{wu('7a')}{s()})?{w('Qualitative')}{s()}{w('nd')}{s()}{w('Quantitative')}{s()}{w('Disclosure')}{s()}{w('bout')}{s()}{w('Market')}{s()}{w('Risk')}{s()}\|"])
    item_ptrn2['item_8'] = rf"\|(?:{wu('Item')}{s()}8{s()})?{w('Financial')}{s()}{w('Statement')}{s()}{w('nd')}{s()}{w('Supplementary')}{s()}{w('Data')}{s()}\|"
    item_ptrn2['item_9'] = rf"\|(?:{wu('Item')}{s()}9{s()})?{w('Change')}{s()}{w('in')}{s()}{w('nd')}{s()}{w('Disagreement')}{s()}{w('ith')}{s()}{w('Accountant')}{s()}{w('n')}{s()}{w('Accounting')}{s()}{w('nd')}{s()}{w('Financial')}{s()}{w('Disclosure')}{s()}\|"
    item_ptrn2['item_9a'] = rf"\|(?:{wu('Item')}{s()}{wu('9a')}{s()})?{w('Control')}{s()}{w('nd')}{s()}{w('Procedure')}{s()}\|"
    item_ptrn2['item_9b'] = rf"\|(?:{wu('Item')}{s()}{wu('9b')}{s()})?{w('Other')}{s()}{w('Information')}{s()}\|"
    item_ptrn2['item_10'] = rf"\|(?:{wu('Item')}{s()}10{s()})?{w('Director')}{s()}{w('Executive')}{s()}{w('Officer')}{s()}{w('nd')}{s()}{w('Corporate')}{s()}{w('Governance')}{s()}\|"
    item_ptrn2['item_11'] = rf"\|(?:{wu('Item')}{s()}11{s()})?{w('Executive')}{s()}{w('Compensation')}{s()}\|"
    item_ptrn2['item_12'] = rf"\|(?:{wu('Item')}{s()}12{s()})?{w('Security')}{s()}{w('Ownership')}{s()}{w('f')}{s()}{w('Certain')}{s()}{w('Beneficial')}{s()}{w('Owner')}{s()}{w('nd')}{s()}{w('Management')}{s()}{w('nd')}{s()}{w('Related')}{s()}{w('Stockholder')}{s()}{w('Matter')}s?{s()}\|"
    item_ptrn2['item_13'] = rf"\|(?:{wu('Item')}{s()}13{s()})?{w('Certain')}{s()}{w('Relationship')}{s()}{w('nd')}{s()}{w('Related')}{s()}{w('Transaction')}{s()}{w('nd')}{s()}{w('Director')}{s()}{w('Independence')}{s()}\|"
    item_ptrn2['item_14'] = rf"\|(?:{wu('Item')}{s()}14{s()})?{w('Principal')}{s()}{w('Account')}(?:{w('ant')}|{w('ing')}){s()}{w('Fee')}{s()}{w('nd')}{s()}{w('Service')}{s()}\|"
    item_ptrn2['item_15'] = rf"\|(?:{wu('Item')}{s()}15{s()})?{w('Exhibits')}{s()}{w('Financial')}{s()}{w('Statement')}{s()}{w('Schedule')}{s()}\|"
    return item_ptrn2

def get_item_ptrn3():
    item_ptrn3 = dict()
    item_ptrn3['item_1'] = r'|'.join([rf"\W{w('Business')}\W", 
                                    rf"\W{w('BUSINESS')}\W"])
    item_ptrn3['item_1a'] = r'|'.join([rf"\W{w('Risk')}{s()}{w('Factors')}\W", 
                                    rf"\W{w('RISK')}{s()}{w('FACTORS')}\W"])
    item_ptrn3['item_1b'] = r'|'.join([rf"\W{w('Unresolved')}{s()}(?:{w('Staff')}|{w('SEC')}|{w('Sec')}){s()}{w('Comment')}s?\W", 
                                    rf"\W{w('UNRESOLVED')}{s()}(?:{w('STAFF')}|{w('SEC')}){s()}{w('COMMENT')}S?\W"])
    item_ptrn3['item_2'] = r'|'.join([rf"\W{w('Properties')}\W", 
                                    rf"\W{w('PROPERTIES')}\W"])
    item_ptrn3['item_3'] = r'|'.join([rf"\W{w('Legal')}{s()}{w('Proceeding')}s?", 
                                    rf"\W{w('LEGAL')}{s()}{w('PROCEEDING')}S?"])
    item_ptrn3['item_4'] = r'|'.join([rf"\W{w('Mine')}{s()}{w('Safety')}{s()}{w('Disclosure')}s?\W",
                                    rf"\W{w('MINE')}{s()}{w('SAFETY')}{s()}{w('DISCLOSURE')}S?\W",
                                    rf"\W(?:{w('Removed')}{s()}(?:{w('A')}|{w('a')}){w('nd')}{s()})?{w('Reserved')}\W",
                                    rf"\W(?:{w('REMOVED')}{s()}{w('AND')}{s()})?{w('RESERVED')}\W",
                                    rf"\W{w('Submission')}{s()}(?:{w('O')}|{w('o')}){w('f')}{s()}{w('Matter')}{s()}(?:{w('T')}|{w('t')}){w('o')}{s()}(?:{w('A')}|{w('a')}){s()}{w('Vote')}{s()}(?:{w('O')}|{w('o')}){w('f')}{s()}{w('Security')}{s()}{w('Holder')}s?\W",
                                    rf"\W{w('SUBMISSION')}{s()}{w('OF')}{s()}{w('MATTER')}{s()}{w('TO')}{s()}{w('A')}{s()}{w('VOTE')}{s()}{w('OF')}{s()}{w('SECURITY')}{s()}{w('HOLDER')}S?\W"])
    item_ptrn3['item_5'] = r'|'.join([rf"\W{w('Market')}{s()}(?:{w('F')}|{w('f')}){w('or')}{s()}{w('Registrant')}{s()}{w('Common')}{s()}{w('Equit')}(?:{w('y')}|{w('ies')}){s()}{w('Related')}{s()}{w('Stockholder')}{s()}{w('Matter')}{s()}(?:{w('A')}|{w('a')}){w('nd')}{s()}{w('Issuer')}{s()}{w('Purchase')}{s()}(?:{w('O')}|{w('o')}){w('f')}{s()}{w('Equit')}(?:{w('y')}|{w('ies')}){s()}{w('Securities')}\W", 
                                    rf"\W{w('MARKET')}{s()}{w('FOR')}{s()}{w('REGISTRANT')}{s()}{w('COMMON')}{s()}{w('EQUIT')}(?:{w('Y')}|{w('IES')}){s()}{w('RELATED')}{s()}{w('STOCKHOLDER')}{s()}{w('MATTER')}{s()}{w('AND')}{s()}{w('ISSUER')}{s()}{w('PURCHASE')}{s()}{w('OF')}{s()}{w('EQUIT')}(?:{w('Y')}|{w('IES')}){s()}{w('SECURITIES')}\W"])
    item_ptrn3['item_6'] = r'|'.join([rf"\W{w('Selected')}{s()}(?:{w('Consolidated')}{s()})?{w('Financial')}{s()}{w('Data')}\W", 
                                    rf"\W{w('SELECTED')}{s()}(?:{w('CONSOLIDATED')}{s()})?{w('FINANCIAL')}{s()}{w('DATA')}\W"])
    item_ptrn3['item_7'] = r'|'.join([rf"\W{w('Management')}{s()}{w('Discussion')}{s()}(?:{w('A')}|{w('a')}){w('nd')}{s()}{w('Analy')}(?:{w('sis')}|{w('ses')}){s()}(?:{w('O')}|{w('o')}){w('f')}{s()}{w('Financial')}{s()}{w('Condition')}{s()}(?:{w('A')}|{w('a')}){w('nd')}{s()}{w('Result')}{s()}(?:{w('O')}|{w('o')}){w('f')}{s()}{w('Operation')}s?\W", 
                                    rf"\W{w('MANAGEMENT')}{s()}{w('DISCUSSION')}{s()}{w('AND')}{s()}{w('ANALY')}(?:{w('SIS')}|{w('SES')}){s()}{w('OF')}{s()}{w('FINANCIAL')}{s()}{w('CONDITION')}{s()}{w('AND')}{s()}{w('RESULT')}{s()}{w('OF')}{s()}{w('OPERATION')}S?\W",
                                    rf"\W{w('Management')}{s()}{w('Discussion')}{s()}(?:{w('A')}|{w('a')}){w('nd')}{s()}{w('Analy')}(?:{w('sis')}|{w('ses')}){s()}(?:{w('O')}|{w('o')}){w('f')}{s()}{w('Result')}{s()}(?:{w('O')}|{w('o')}){w('f')}{s()}{w('Operation')}{s()}(?:{w('A')}|{w('a')}){w('nd')}{s()}{w('Financial')}{s()}{w('Condition')}s?\W", 
                                    rf"\W{w('MANAGEMENT')}{s()}{w('DISCUSSION')}{s()}{w('AND')}{s()}{w('ANALY')}(?:{w('SIS')}|{w('SES')}){s()}{w('OF')}{s()}{w('RESULT')}{s()}{w('OF')}{s()}{w('OPERATION')}{s()}{w('AND')}{s()}{w('FINANCIAL')}{s()}{w('CONDITION')}S?\W"])
    item_ptrn3['item_7a'] = '|'.join([rf"\W{w('Quantitative')}{s()}(?:{w('A')}|{w('a')}){w('nd')}{s()}{w('Qualitative')}{s()}{w('Disclosure')}{s()}(?:{w('A')}|{w('a')}){w('bout')}{s()}{w('Market')}{s()}{w('Risk')}s?\W",
                                    rf"\W{w('QUANTITATIVE')}{s()}{w('AND')}{s()}{w('QUALITATIVE')}{s()}{w('DISCLOSURE')}{s()}{w('ABOUT')}{s()}{w('MARKET')}{s()}{w('RISK')}S?\W",
                                    rf"\W{w('Qualitative')}{s()}(?:{w('A')}|{w('a')}){w('nd')}{s()}{w('Quantitative')}{s()}{w('Disclosure')}{s()}(?:{w('A')}|{w('a')}){w('bout')}{s()}{w('Market')}{s()}{w('Risk')}s?\W",
                                    rf"\W{w('QUALITATIVE')}{s()}{w('AND')}{s()}{w('QUANTITATIVE')}{s()}{w('DISCLOSURE')}{s()}{w('ABOUT')}{s()}{w('MARKET')}{s()}{w('RISK')}S?\W"])
    item_ptrn3['item_8'] = r'|'.join([rf"\W{w('Financial')}{s()}{w('Statement')}s?{s()}(?:{w('A')}|{w('a')}){w('nd')}{s()}{w('Supplementary')}{s()}{w('Data')}\W",
                                    rf"\W{w('FINANCIAL')}{s()}{w('STATEMENT')}S?{s()}{w('AND')}{s()}{w('SUPPLEMENTARY')}{s()}{w('DATA')}\W"])
    item_ptrn3['item_9'] = r'|'.join([rf"\W{w('Change')}{s()}(?:{w('I')}|{w('i')}){w('n')}{s()}(?:{w('A')}|{w('a')}){w('nd')}{s()}{w('Disagreement')}{s()}(?:{w('W')}|{w('w')}){w('ith')}{s()}{w('Accountant')}{s()}(?:{w('O')}|{w('o')}){w('n')}{w('Accounting')}{s()}(?:{w('A')}|{w('a')}){w('nd')}{s()}{w('Financial')}{s()}{w('Disclosure')}s?\W",
                                    rf"\W{w('CHANGE')}{s()}{w('IN')}{s()}{w('AND')}{s()}{w('DISAGREEMENT')}{s()}{w('WITH')}{s()}{w('ACCOUNTANT')}{s()}{w('ON')}{w('ACCOUNTING')}{s()}{w('AND')}{s()}{w('FINANCIAL')}{s()}{w('DISCLOSURE')}S?\W"])
    item_ptrn3['item_9a'] = r'|'.join([rf"\W{w('Control')}s?{s()}(?:{w('A')}|{w('a')}){w('nd')}{s()}{w('Procedure')}s?\W",
                                    rf"\W{w('CONTROL')}S?{s()}{w('AND')}{s()}{w('PROCEDURE')}S?\W"])
    item_ptrn3['item_9b'] = r'|'.join([rf"\W{w('Other')}{s()}{w('Information')}\W",
                                    rf"\W{w('OTHER')}{s()}{w('INFORMATION')}\W"])
    item_ptrn3['item_10'] = r'|'.join([rf"\W{w('Director')}{s()}{w('Executive')}{s()}{w('Officer')}{s()}(?:{w('A')}|{w('a')}){w('nd')}{s()}{w('Corporate')}{s()}{w('Governance')}s?\W",
                                    rf"\W{w('DIRECTOR')}{s()}{w('EXECUTIVE')}{s()}{w('OFFICER')}{s()}{w('AND')}{s()}{w('CORPORATE')}{s()}{w('GOVERNANCE')}S?\W"])
    item_ptrn3['item_11'] = r'|'.join([rf"\W{w('Executive')}{s()}{w('Compensation')}s?\W",
                                    rf"\W{w('EXECUTIVE')}{s()}{w('COMPENSATION')}S?\W"])
    item_ptrn3['item_12'] = r'|'.join([rf"\W{w('Security')}{s()}{w('Ownership')}{s()}(?:{w('O')}|{w('o')}){w('f')}{s()}{w('Certain')}{s()}{w('Beneficial')}{s()}{w('Owner')}{s()}(?:{w('A')}|{w('a')}){w('nd')}{s()}{w('Management')}{s()}(?:{w('A')}|{w('a')}){w('nd')}{s()}{w('Related')}{s()}{w('Stockholder')}{s()}{w('Matter')}s?\W",
                                    rf"\W{w('SECURITY')}{s()}{w('OWNERSHIP')}{s()}{w('OF')}{s()}{w('CERTAIN')}{s()}{w('BENEFICIAL')}{s()}{w('OWNER')}{s()}{w('AND')}{s()}{w('MANAGEMENT')}{s()}{w('AND')}{s()}{w('RELATED')}{s()}{w('STOCKHOLDER')}{s()}{w('MATTER')}S?\W"])
    item_ptrn3['item_13'] = r'|'.join([rf"\W{w('Certain')}{s()}{w('Relationship')}{s()}(?:{w('A')}|{w('a')}){w('nd')}{s()}{w('Related')}{s()}{w('Transaction')}{s()}(?:{w('A')}|{w('a')}){w('nd')}{s()}{w('Director')}{s()}{w('Independence')}\W",
                                    rf"\W{w('CERTAIN')}{s()}{w('RELATIONSHIP')}{s()}{w('AND')}{s()}{w('RELATED')}{s()}{w('TRANSACTION')}{s()}{w('AND')}{s()}{w('DIRECTOR')}{s()}{w('INDEPENDENCE')}\W"])
    item_ptrn3['item_14'] = r'|'.join([rf"\W{w('Principal')}{s()}{w('Account')}(?:{w('ant')}|{w('ing')}){s()}{w('Fee')}{s()}(?:{w('A')}|{w('a')}){w('nd')}{s()}{w('Service')}s?\W",
                                    rf"\W{w('PRINCIPAL')}{s()}{w('ACCOUNT')}(?:{w('ANT')}|{w('IND')}){s()}{w('FEE')}{s()}{w('AND')}{s()}{w('SERVICE')}S?\W"])
    item_ptrn3['item_15'] = r'|'.join([rf"\W{w('Exhibits')}{s()}{w('Financial')}{s()}{w('Statement')}{s()}{w('Schedule')}s?\W",
                                    rf"\W{w('EXHIBITS')}{s()}{w('FINANCIAL')}{s()}{w('STATEMENT')}{s()}{w('SCHEDULE')}S?\W"])
    return item_ptrn3

"""
Given a document, extract start and end position of each Item
"""

def dedup_pos(pos):
    return list(pd.DataFrame({0:[x[0] for x in pos], 1:[x[1] for x in pos]}).drop_duplicates(subset=[0]).to_records(index=False))

def find_item_pos(doc, log_mode=False):
    item_pos = []
    
    # loop througn all items
    for item in item_ptrn1:
        
        # pattern 1 (normal + upper)
        pos = [(m.start(), m.end()) for m in re.finditer(item_ptrn1[item], doc)] + [(m.start(), m.end()) for m in re.finditer(item_ptrn1[item].upper(), doc)]
        pos = dedup_pos(pos)
        log(f'[{item}] After attempt 1 yielded {len(pos)} matches') if log_mode==True else None

        # pattern 2 ("Item" as optional, normal + upper)
        if len(pos) == 0 or (len(pos) == 1 and len(re.findall(rf"{w('table')}{s()}{w('of')}{s()}{w('content')}", doc.lower())) > 0 and pos[0][0] < 7000):
            pos = pos + [(m.start(), m.end()) for m in re.finditer(item_ptrn2[item], doc)] + [(m.start(), m.end()) for m in re.finditer(item_ptrn2[item].upper(), doc)]
            pos = dedup_pos(pos)
            log(f'[{item}] After attempt 2 yielded {len(pos)} matches') if log_mode==True else None

        # pattern 3
        if len(pos) == 0 or (len(pos) == 1 and len(re.findall(rf"{w('table')}{s()}{w('of')}{s()}{w('content')}", doc.lower())) > 0 and pos[0][0] < 7000):
            pos = pos + [(m.start(), m.end()) for m in re.finditer(item_ptrn3[item], doc)]
            pos = dedup_pos(pos)
            log(f'[{item}] After attempt 3 yielded {len(pos)} matches') if log_mode==True else None


        # remove first entry due to table of contents
        if len(pos) >= 2  \
        and len(re.findall(rf"{w('table')}{s()}{w('of')}{s()}{w('content')}", doc.lower())) > 0 \
        and pos[0][0] < 6000 \
        and item != 'item_1':
            pos = pos[1:]
            log(f'[{item}] Removed first result due to Table of Contents') if log_mode==True else None

        # remove occurrance due to references
        pos_filtered = []
        for p in pos:
            match = doc[p[0]:p[1]]
            pre = doc[p[0]-20:p[0]].lower()
            suf = doc[p[1]:p[1]+20].lower()
            log(f'[{item}] pos {p} : <<{pre}....{match}....{suf}>>') if log_mode==True else None
            pre_ptrn = r"""(\W"$|\W“$|('s\W)$|\Wsee\W$|\Win\W$|\Wthe\W$|\Wour\W$|\Wthis\W$|\Wwithin\W$|\Wherein\W$|\Wrefer to\W$|\Wreferring\W$)"""
            suf_ptrn = r"""(^\Wshould\W|^\Wshall\W|^\Wmust\W|^\Wwas\W|^\Wwere\W|^\Whas\W|^\Whad\W|^\Wis\W|^\Ware\W)"""
            if re.search(pre_ptrn, pre) or re.search(suf_ptrn, suf):
                log(f'[{item}] removed the above match') if log_mode==True else None
            else:
                pos_filtered.append(p)
        pos = pos_filtered.copy()

        # save position as dataframe
        pos = pd.DataFrame({'item':[item]*len(pos), 'pos_start':[x[0] for x in pos]})
        item_pos.append(pos)

    # combine positions for all items
    item_pos = pd.concat(item_pos).sort_values('pos_start').reset_index(drop=True)
    # define ending position
    item_pos['pos_end'] = item_pos.pos_start.shift(-1).fillna(len(doc))
    # define length
    item_pos['len'] = item_pos.pos_end - item_pos.pos_start
    # for each item, select the match with longest length
    item_pos = item_pos.sort_values(['item','len','pos_start'], ascending=[1,0,0]).drop_duplicates(subset=['item']).sort_values('pos_start')
    item_pos = pd.concat([item_pos[item_pos.item==item][['pos_start','pos_end']].reset_index(drop=True).rename(columns={'pos_start':f'{item}_pos_start','pos_end':f'{item}_pos_end'}) for item in item_ptrn1], axis=1)
    # fillna with zero
    item_pos = item_pos.fillna(0).astype(int)
    # if item_pos is empty due to no item found, put all zeros as a row
    if item_pos.shape[0] == 0:
        item_pos.loc[0,:] = [0] * 2 * len(item_ptrn1)
    # record the full document length
    item_pos['full_doc_len'] = len(doc)
    # check if non empty df is returned
    assert item_pos.shape[0]==1
    return item_pos


# function to sample check item extraction quality
def show_item(doc_dict):
    n = 100
    for item in item_ptrn1:
        print(f'{item}: {doc_dict[item][:n]}........{doc_dict[item][-n:]}')
    return

# urls = ['https://www.sec.gov/Archives/edgar/data/1166691/000119312508034239/d10k.htm',
#        'https://www.sec.gov/Archives/edgar/data/922224/000092222411000029/form10k.htm',
#        'https://www.sec.gov/Archives/edgar/data/1283699/000119312511051403/d10k.htm']
# docs = {}
# for i in range(len(urls)):

#     url = urls[i]
#     doc_id = i
#     txt = requests.get(url, headers={"user-agent": f"chan_tai_man_{int(float(np.random.rand(1)) * 1e7)}@gmail.com"}).text

#     # clean doc, extract items
#     txt = soup = BeautifulSoup(txt, 'lxml').get_text('|', strip=True)
#     txt = clean_doc1(txt)
#     item_pos = find_item_pos(txt)
#     doc_dict = {}
#     doc_dict['full'] = txt[item_pos.iloc[0]['item_1_pos_start'] :]
#     for item in item_ptrn1:
#         doc_dict[item] = txt[item_pos.iloc[0][f'{item}_pos_start'] : item_pos.iloc[0][f'{item}_pos_end']]
#     for x in doc_dict:
#         doc_dict[x] = clean_doc2(doc_dict[x])
#     docs[doc_id] = doc_dict

# # Signal Extraction

'''
Download NLP pretrained models
'''

if mode in ['full','gpu']:
    !pip install sentence-transformers
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    import torch
    fb_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    fb_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    fb_model = fb_model.to("cuda:0")

# if mode in ['full','wv']:
#     import gensim.downloader as api
#     wv = api.load('word2vec-google-news-300')

# '''
# Generate a global TFIDF model
# '''
# if mode in ['full','cpu']:
#     # sample and clean doc
#     n_sample_per_cik = 1
#     df = master_idx.groupby('cik').sample(n_sample_per_cik).sort_values('filing_date').reset_index(drop=True)
#     doc_list = []
#     for i in range(len(df)):
#         url = df.iloc[i]['url_10k']
#         doc_id = df.iloc[i]['doc_id']
#         txt = requests.get(url, headers={"user-agent": f"chan_tai_man_{int(float(np.random.rand(1)) * 1e7)}@gmail.com"}).text
#         txt = soup = BeautifulSoup(txt, 'lxml').get_text('|', strip=True)
#         txt = clean_doc1(txt)
#         txt = clean_doc2(txt)
#         doc_list.append(txt)

#     # build tfidf for 1 and 2 gram
#     global_tfidf_1g = TfidfVectorizer(ngram_range=(1,1), norm='l2', use_idf=True, binary=False, token_pattern=r"(?u)\b[a-z]{3,}\b").fit(doc_list)
#     global_tfidf_2g = TfidfVectorizer(ngram_range=(1,2), norm='l2', min_df=0.0, max_df=0.7, use_idf=True, binary=False, token_pattern=r"(?u)\b[a-z]{3,}\b").fit(doc_list)
#     log(f'Vocab size of TFIDF (1-gram): {len(global_tfidf_1g.vocabulary_)}')
#     log(f'Vocab size of TFIDF (2-gram): {len(global_tfidf_2g.vocabulary_)}')

#     # release memory
#     del txt, doc_list
#     gc.collect()

#     # get the column index for vocab overlapping with Word2Vec
#     wv_vocab_list = list(wv.key_to_index)
#     tfidf_vocab = global_tfidf_1g.vocabulary_
#     tfidf_vocab_swap = {v: k for k, v in tfidf_vocab.items()}
#     tfidf_1g_wv_idx = sorted([global_tfidf_1g.vocabulary_[x] for x in list(global_tfidf_1g.vocabulary_) if x in wv_vocab_list])
#     tfidf_1g_wv_word = [tfidf_vocab_swap[x] for x in tfidf_1g_wv_idx]
#     log(f'Vocab size of TFIDF overlapped with Word2Vec: {len(tfidf_1g_wv_idx)}')

# load the pre-computed global TFIDF, and subset of word2vec list
global_tfidf_1g = load_pkl('../input/nlp10k-signal-extraction-pre/global_tfidf_1g')
global_tfidf_2g = load_pkl('../input/nlp10k-signal-extraction-pre/global_tfidf_2g')
tfidf_1g_wv_idx = load_pkl('../input/nlp10k-signal-extraction-pre/tfidf_1g_wv_idx')
tfidf_1g_wv_word = load_pkl('../input/nlp10k-signal-extraction-pre/tfidf_1g_wv_word')

wv_subset = load_pkl('../input/nlp10k-signal-extraction-pre/wv_subset')
assert list(wv_subset) == tfidf_1g_wv_word
wv_subset = np.concatenate(list(wv_subset.values())).reshape(len(wv_subset), 300)
log(f'Shape of wv_subset: {wv_subset.shape}')

'''
Loughran and McDonald’s Master Dictionary
'''
# load Loughran and McDonald’s Master Dictionary (2020)
master_dict = pd.read_csv('../input/loughranmcdonald-masterdictionary-2020/LoughranMcDonald_MasterDictionary_2020.csv')
master_dict.columns = ['_'.join([y.lower() for y in x.split()]) for x in master_dict.columns]
master_dict.word = master_dict.word.str.lower()

# extract specific word lists
negative_word_list = master_dict.loc[lambda x: x.negative!=0].word.tolist()
positive_word_list = master_dict.loc[lambda x: x.positive!=0].word.tolist()
uncertainty_word_list = master_dict.loc[lambda x: x.uncertainty!=0].word.tolist()
litigious_word_list = master_dict.loc[lambda x: x.litigious!=0].word.tolist()
strong_modal_word_list = master_dict.loc[lambda x: x.strong_modal!=0].word.tolist()
weak_modal_word_list = master_dict.loc[lambda x: x.weak_modal!=0].word.tolist()
constraining_word_list = master_dict.loc[lambda x: x.constraining!=0].word.tolist()
complexity_word_list = master_dict.loc[lambda x: x.complexity!=0].word.tolist()

'''
Change in length
'''
# full doc
def gen_feat_ch_full_len(docs):
    feat = pd.Series([len(doc_dict['full']) for doc_dict in docs.values()])
    feat = np.log(feat).diff()
    feat = feat * -1
    return feat.rename('feat_ch_full_len')

# Item 1A - Risk Factors
def gen_feat_ch_item_1a_len(docs):
    feat = pd.Series([len(doc_dict['item_1a']) for doc_dict in docs.values()])
    feat = np.log(feat).diff()
    feat = feat * -1
    return feat.rename('feat_ch_item_1a_len')

# Item 1B - Unresolved Staff Comments
def gen_feat_ch_item_1b_len(docs):
    feat = pd.Series([len(doc_dict['item_1b']) for doc_dict in docs.values()])
    feat = np.log(feat).diff()
    feat = feat * -1
    return feat.rename('feat_ch_item_1b_len')

# Item 3 - Legal Proceedings
def gen_feat_ch_item_3_len(docs):
    feat = pd.Series([len(doc_dict['item_3']) for doc_dict in docs.values()])
    feat = np.log(feat).diff()
    feat = feat * -1
    return feat.rename('feat_ch_item_3_len')

'''
Document Similarity
'''
# full doc, cosine similarity, 1 gram
def gen_feat_full_cos_1gram(docs):
    doc_list = [doc_dict['full'] for doc_dict in docs.values()]
    tf_vectors = global_tfidf_1g.transform(doc_list)
    feat = pd.Series([cosine_similarity(tf_vectors[i-1:i+1,:])[0][1] if i > 0 else np.NaN for i in range(len(doc_list))])
    return feat.rename('feat_full_cos_1gram')

# full doc, cosine similarity, 2 gram
def gen_feat_full_cos_2gram(docs):
    doc_list = [doc_dict['full'] for doc_dict in docs.values()]
    tf_vectors = global_tfidf_2g.transform(doc_list)
    feat = pd.Series([cosine_similarity(tf_vectors[i-1:i+1,:])[0][1] if i > 0 else np.NaN for i in range(len(doc_list))])
    return feat.rename('feat_full_cos_2gram')

# full doc, jaccard similarity, 1 gram
def gen_feat_full_jac_1gram(docs):
    doc_list = [doc_dict['full'] for doc_dict in docs.values()]
    vectorizer = CountVectorizer(ngram_range=(1,1), binary=True, token_pattern=r"(?u)\b[a-z]{3,}\b")
    tf_vectors = vectorizer.fit_transform(doc_list)
    feat = pd.Series([jaccard_score(tf_vectors[i-1,:].toarray().flatten(), tf_vectors[i,:].toarray().flatten()) if i > 0 else np.NaN for i in range(len(doc_list))])
    return feat.rename('feat_full_jac_1gram')

# full doc, jaccard similarity, 2 gram
def gen_feat_full_jac_2gram(docs):
    doc_list = [doc_dict['full'] for doc_dict in docs.values()]
    vectorizer = CountVectorizer(ngram_range=(1,2), binary=True, token_pattern=r"(?u)\b[a-z]{3,}\b")
    tf_vectors = vectorizer.fit_transform(doc_list)
    feat = pd.Series([jaccard_score(tf_vectors[i-1,:].toarray().flatten(), tf_vectors[i,:].toarray().flatten()) if i > 0 else np.NaN for i in range(len(doc_list))])
    return feat.rename('feat_full_jac_2gram')

# Levenshtein similarity
def fast_lev_ratio(s1,s2):
    total_len = len(s1) + len(s2)
    if total_len > 0:
        return 1 - levenshtein(s1, s2) / total_len
    else:
        return np.NaN
def gen_feat_item_1a_lev(docs):
    doc_list = [doc_dict['item_1a'] for doc_dict in docs.values()]
    feat = pd.Series([fast_lev_ratio(doc_list[i-1], doc_list[i]) if i > 0 else np.NaN for i in range(len(doc_list))])
    return feat.rename('feat_item_1a_lev')
def gen_feat_item_7_lev(docs):
    doc_list = [doc_dict['item_7'] for doc_dict in docs.values()]
    feat = pd.Series([fast_lev_ratio(doc_list[i-1], doc_list[i]) if i > 0 else np.NaN for i in range(len(doc_list))])
    return feat.rename('feat_item_7_lev')

'''
Dictionary based sentiment (Loughran and McDonald)
'''
# net postive words change in proportion
def gen_feat_lm_postive(docs):
    doc_list = [doc_dict['full'] for doc_dict in docs.values()]
    doc_len = pd.Series([len(x) for x in doc_list])
    vectorizer = CountVectorizer(ngram_range=(1,1), binary=False, token_pattern=r"(?u)\b[a-z]{3,}\b").fit(doc_list)
    tf_vectors = vectorizer.transform(doc_list)
    pos_target_cols = [vectorizer.vocabulary_[x] for x in positive_word_list if x in list(vectorizer.vocabulary_.keys())]
    neg_target_cols = [vectorizer.vocabulary_[x] for x in negative_word_list if x in list(vectorizer.vocabulary_.keys())]
    feat = pd.Series([(tf_vectors[i,pos_target_cols].sum() - tf_vectors[i,neg_target_cols].sum()) / doc_len[i] for i in range(len(doc_list))]).diff()
    return feat.rename('feat_lm_postive')

# uncertainty words change in proportion
def gen_feat_lm_uncertainty(docs):
    doc_list = [doc_dict['full'] for doc_dict in docs.values()]
    doc_len = pd.Series([len(x) for x in doc_list])
    vectorizer = CountVectorizer(ngram_range=(1,1), binary=False, token_pattern=r"(?u)\b[a-z]{3,}\b").fit(doc_list)
    tf_vectors = vectorizer.transform(doc_list)
    target_cols = [vectorizer.vocabulary_[x] for x in uncertainty_word_list if x in list(vectorizer.vocabulary_.keys())]
    feat = pd.Series([tf_vectors[i,target_cols].sum() / doc_len[i] for i in range(len(doc_list))]).diff()
    feat = feat * -1
    return feat.rename('feat_lm_uncertainty')

# uncertainty words change in proportion
def gen_feat_lm_litigious(docs):
    doc_list = [doc_dict['full'] for doc_dict in docs.values()]
    doc_len = pd.Series([len(x) for x in doc_list])
    vectorizer = CountVectorizer(ngram_range=(1,1), binary=False, token_pattern=r"(?u)\b[a-z]{3,}\b").fit(doc_list)
    tf_vectors = vectorizer.transform(doc_list)
    target_cols = [vectorizer.vocabulary_[x] for x in litigious_word_list if x in list(vectorizer.vocabulary_.keys())]
    feat = pd.Series([tf_vectors[i,target_cols].sum() / doc_len[i] for i in range(len(doc_list))]).diff()
    feat = feat * -1
    return feat.rename('feat_lm_litigious')

'''
Sentence encoding
'''
def gen_feat_sen_enc(docs):
    doc_list = [doc_dict['full'] for doc_dict in docs.values()]
    vecs = []
    for doc in doc_list:
        sen_list = [x for x in tokenize.sent_tokenize(doc) if len(x)>=30 and len(x)<=1000 and re.match(r'[a-z]', x)]
        vecs.append(st_model.encode(sentences=sen_list, show_progress_bar=False).mean(axis=0).flatten())
    vecs = np.concatenate(vecs).reshape(-1, vecs[0].shape[0])
    feat = pd.Series([cosine_similarity(vecs[i-1:i+1,:])[0][1] if i > 0 else np.NaN for i in range(len(doc_list))])
    return feat.rename('feat_sen_enc')


'''
Finbert Sentiment on Item 1A & 7
'''
def gen_feat_item_sentiment(docs):
    doc_list = [doc_dict['item_1a'] + '.' + doc_dict['item_7'] for doc_dict in docs.values()]
    sentiment = []
    for doc in doc_list:
        sen_list = [x for x in tokenize.sent_tokenize(doc) if len(x)>=30 and len(x)<=1000 and re.match(r'[a-z]', x)]
        if len(sen_list)==0:
            sentiment.append(np.NaN)
            continue
        batch_size = 8
        n_batch = math.ceil(len(sen_list)/batch_size)
        sentiment_sum = 0
        for i in range(n_batch):
            inputs = fb_tokenizer(sen_list[batch_size*i:batch_size*(i+1)], padding=True, truncation=True, return_tensors='pt').to("cuda:0")
            with torch.no_grad():
                outputs = fb_model(**inputs)
            sentiment_sum += float(torch.nn.functional.softmax(outputs.logits, dim=-1)[:,0].sum())
            torch.cuda.empty_cache()
        sentiment.append(sentiment_sum / len(sen_list))
    feat = pd.Series(sentiment).ffill().diff()
    return feat.rename('feat_item_sentiment')


'''
Finbert Sentiment on Forward-Looking Statements
'''
def gen_feat_fls_sentiment(docs):
    doc_list = [doc_dict['item_1a'] + '.' + doc_dict['item_7'] for doc_dict in docs.values()]
    fls_ptrn = r"(\baim\b|\banticipate\b|\bbelieve\b|\bcould\b|\bcontinue\b|\bestimate\b|\bexpansion\b|\bexpect\b|\bexpectation\b|\bexpected to be\b|\bfocus\b|\bforecast\b|\bgoal\b|\bgrow\b|\bguidance\b|\bintend\b|\binvest\b|\bis expected\b|\bmay\b|\bobjective\b|\bplan\b|\bpriority\b|\bproject\b|\bstrategy\b|\bto be\b|\bwe'll\b|\bwill\b|\bwould\b)"
    sentiment = []
    for doc in doc_list:
        sen_list = [x for x in tokenize.sent_tokenize(doc) if len(x)>=30 and len(x)<=1000 and re.match(r'[a-z]', x) and re.search(fls_ptrn, x)]
        if len(sen_list)==0:
            sentiment.append(np.NaN)
            continue
        batch_size = 8
        n_batch = math.ceil(len(sen_list)/batch_size)
        sentiment_sum = 0
        for i in range(n_batch):
            inputs = fb_tokenizer(sen_list[batch_size*i:batch_size*(i+1)], padding=True, truncation=True, return_tensors='pt').to("cuda:0")
            with torch.no_grad():
                outputs = fb_model(**inputs)
            sentiment_sum += float(torch.nn.functional.softmax(outputs.logits, dim=-1)[:,0].sum())
            torch.cuda.empty_cache()
        sentiment.append(sentiment_sum / len(sen_list))
    feat = pd.Series(sentiment).ffill().diff()
    return feat.rename('feat_fls_sentiment')


'''
Word2Vec
'''
def gen_feat_word2vec(docs):
    doc_list = [doc_dict['full'] for doc_dict in docs.values()]
    tf_vectors = global_tfidf_1g.transform(doc_list)[:,tfidf_1g_wv_idx]
    tf_vectors = (tf_vectors.toarray() > 0).astype(int)
#     tf_vectors = tf_vectors / tf_vectors.sum(axis=1)
    avg_vecs = tf_vectors @ wv_subset
    feat = pd.Series([cosine_similarity(avg_vecs[i-1:i+1,:])[0][1] if i > 0 else np.NaN for i in range(len(doc_list))])
    return feat.rename('feat_word2vec')

# # Run All per CIK

def gen_signal(cik):
    log(f'{cik}: Started signal generation')
    df = master_idx.loc[lambda x: x.cik==cik].sort_values('filing_date').reset_index(drop=True)
    docs = {}
    for i in range(len(df)):
        
        # load 10-K text from EDGAR html url
        url = df.iloc[i]['url_10k']
        doc_id = df.iloc[i]['doc_id']
        
        # url request
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        txt = session.get(url, headers={"user-agent": f"chan_tai_man_{int(float(np.random.rand(1)) * 1e7)}@gmail.com"}).text

        # clean doc, extract items
        txt = soup = BeautifulSoup(txt, 'lxml').get_text('|', strip=True)
        txt = clean_doc1(txt)
        item_pos = find_item_pos(txt)
        doc_dict = {}
        doc_dict['full'] = txt[item_pos.iloc[0]['item_1_pos_start'] :]
        for item in item_ptrn1:
            doc_dict[item] = txt[item_pos.iloc[0][f'{item}_pos_start'] : item_pos.iloc[0][f'{item}_pos_end']]
        for x in doc_dict:
            doc_dict[x] = clean_doc2(doc_dict[x])
        docs[doc_id] = doc_dict
        
    # generate signal
    feat_vecs = [pd.Series(list(docs.keys())).rename('doc_id')]
    if mode in ['full','cpu']:
        feat_vecs += [gen_feat_ch_full_len(docs),
                        gen_feat_ch_item_1a_len(docs),
                        gen_feat_ch_item_1b_len(docs),
                        gen_feat_ch_item_3_len(docs),
                        gen_feat_full_cos_1gram(docs),
                        gen_feat_full_cos_2gram(docs),
                        gen_feat_full_jac_1gram(docs),
                        gen_feat_full_jac_2gram(docs),
                        gen_feat_item_1a_lev(docs),
                        gen_feat_item_7_lev(docs),
                        gen_feat_lm_postive(docs),
                        gen_feat_lm_uncertainty(docs),
                        gen_feat_lm_litigious(docs),
                        gen_feat_word2vec(docs)]
    if mode in ['full','gpu']:
        feat_vecs += [gen_feat_sen_enc(docs),
                        gen_feat_item_sentiment(docs),
                        gen_feat_fls_sentiment(docs)]
    feats = pd.concat(feat_vecs, axis=1)
    log(f'Completed signal generation for CIK {cik}')
    return feats


# generate signal per CIK
feats = Parallel(n_jobs=-1)(delayed(gen_signal)(cik) for cik in master_idx.cik.unique())
feats = pd.concat(feats).sort_values('doc_id').reset_index(drop=True)

# map back to stock
df = master_idx[['doc_id','cik','entity','filing_date']].drop_duplicates()
feats = feats.merge(df, how='inner', on='doc_id')
feats = feats.merge(cik_map, how='inner', on='cik')
cols = [c for c in feats if c[:5]=='feat_']
feats = feats[[c for c in feats if c not in cols] + cols]
display(feats.loc[lambda x: x.isnull().sum(axis=1) > 0].groupby('cik')['doc_id'].count().loc[lambda x: x>1])
display(feats.head())

# export
feats.to_csv('feats.csv', index=False)

# # show sample item extraction
# df = master_idx.sample(10).sort_values('filing_date').reset_index(drop=True)
# # df = master_idx.sort_values('filing_date').reset_index(drop=True)

# for i in range(len(df)):

#     print(df.iloc[i]['cik'])
#     print(df.iloc[i]['doc_id'])
#     print(df.iloc[i]['url_10k'])
    
#     # load 10-K text from EDGAR html url
#     url = df.iloc[i]['url_10k']
#     doc_id = df.iloc[i]['doc_id']
#     txt = requests.get(url, headers={"user-agent": f"chan_tai_man_{int(float(np.random.rand(1)) * 1e7)}@gmail.com"}).text

#     # clean doc, extract items
#     txt = soup = BeautifulSoup(txt, 'lxml').get_text('|', strip=True)
#     txt = clean_doc1(txt)
#     item_pos = find_item_pos(txt, log_mode=False)
#     doc_dict = {}
#     doc_dict['full'] = txt[item_pos.iloc[0]['item_1_pos_start'] :]
#     for item in item_ptrn1:
#         doc_dict[item] = txt[item_pos.iloc[0][f'{item}_pos_start'] : item_pos.iloc[0][f'{item}_pos_end']]
#     for x in doc_dict:
#         doc_dict[x] = clean_doc2(doc_dict[x])
#     show_item(doc_dict)

'''
Count of 8-K filings
'''
master_idx_8k = load_pkl('../input/nlp10k-signal-extraction-pre/master_idx_8k')
dates = pd.to_datetime(pd.Series(['2008-01-01'] * 365*11)) \
    .to_frame() \
    .rename(columns={0:'filing_date'})
dates['filing_date'] = dates['filing_date'] + pd.Series([np.timedelta64(i, 'D') for i in range(len(dates))])

feat_cnt_8k = []
for cik in master_idx_8k.cik.unique():
    df = pd.merge(dates, master_idx_8k.loc[lambda x: x.cik==cik], how='left', on='filing_date') \
        .assign(filed=lambda x: x.cik.notnull().astype(int)) \
        .assign(cik=cik) \
        .loc[:,['cik','filing_date','filed']] \
        .assign(feat_cnt_8k = lambda x: x.rolling(365).filed.sum()) \
        .dropna()
    feat_cnt_8k.append(df)
feat_cnt_8k = pd.concat(feat_cnt_8k).reset_index(drop=True)
display(feat_cnt_8k.head())
save_pkl(feat_cnt_8k, 'feat_cnt_8k')


