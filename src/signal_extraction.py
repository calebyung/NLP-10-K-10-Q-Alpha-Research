# import project modules
from src.util import *
import src.constants as const
from src.text_processing import *
from src.signal_10k_def_func import *
from src.signal_10q_def_func import *

# import libraries
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from IPython.display import display
from joblib import Parallel, delayed
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from gensim import downloader as api
import gc
import multiprocessing
import shutil


class SignalExtraction:

    def __init__(self):
        log(f'Initializing SignalExtraction...')
        nltk.download('punkt')
        self.config = yaml.safe_load(open('config.yml'))
        self.master_idx_10k = load_pkl(f'{const.DATA_OUTPUT_PATH}/master_idx_10k.pkl')
        self.master_idx_10q = load_pkl(f'{const.DATA_OUTPUT_PATH}/master_idx_10q.pkl')
        self.master_idx_8k = load_pkl(f'{const.DATA_OUTPUT_PATH}/master_idx_8k.pkl')
        self.cik_map = load_pkl(f'{const.DATA_OUTPUT_PATH}/cik_map.pkl')


    def load_deep_learning_models(self):
        if self.config['gpu_enabled']:
            log(f'Loading deep learning models...')
            self.st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
            self.fb_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.fb_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.fb_model = self.fb_model.to("cuda:0")
        else:
            log(f'Skipped loading deep learning models')


    def load_master_dict(self):
        log(f'Loading Loughran and McDonald’s Master Dictionary...')
        # load Loughran and McDonald’s Master Dictionary (2020)
        master_dict = pd.read_csv(f'{const.DATA_INPUT_PATH}/LoughranMcDonald_MasterDictionary_2020.csv')
        master_dict.columns = ['_'.join([y.lower() for y in x.split()]) for x in master_dict.columns]
        master_dict.word = master_dict.word.str.lower()

        # extract specific word lists
        self.negative_word_list = master_dict.loc[lambda x: x.negative!=0].word.tolist()
        self.positive_word_list = master_dict.loc[lambda x: x.positive!=0].word.tolist()
        self.uncertainty_word_list = master_dict.loc[lambda x: x.uncertainty!=0].word.tolist()
        self.litigious_word_list = master_dict.loc[lambda x: x.litigious!=0].word.tolist()
        self.strong_modal_word_list = master_dict.loc[lambda x: x.strong_modal!=0].word.tolist()
        self.weak_modal_word_list = master_dict.loc[lambda x: x.weak_modal!=0].word.tolist()
        self.constraining_word_list = master_dict.loc[lambda x: x.constraining!=0].word.tolist()
        self.complexity_word_list = master_dict.loc[lambda x: x.complexity!=0].word.tolist()


    def build_tfidf_models(self):
        log(f'Building global TFIDF models...')
        # sample and clean doc
        master_idx_sampled = self.master_idx_10k \
            .sort_values(['cik','filing_date']).reset_index(drop=True) \
            .groupby('cik').last().reset_index() \

        def download_doc(i):
            url = master_idx_sampled.iloc[i]['url_10k']
            headers = {'Host': 'www.sec.gov', 'Connection': 'close',
                    'Accept': 'application/json, text/javascript, */*; q=0.01', 'X-Requested-With': 'XMLHttpRequest',
                    'User-Agent': f"{self.config['edgar_user_agent']}{int(float(np.random.rand(1)) * 1e7)}"
                    }
            txt = requests.get(url, headers=headers).text
            txt = soup = BeautifulSoup(txt, 'lxml').get_text('|', strip=True)
            txt = clean_doc1(txt)
            item_pos = find_item_pos(txt)
            item_1 = clean_doc2(txt[item_pos.iloc[0]['item_1_pos_start'] : item_pos.iloc[0]['item_1_pos_end']])
            full = clean_doc2(txt[item_pos.iloc[0]['item_1_pos_start'] :])
            log(f'Completed downloading doc {i}') if (i>0 and i%100==0) else None
            return {'full':full, 'item_1':item_1}
        docs = {}
        for i in range(len(master_idx_sampled)):
            cik = master_idx_sampled.iloc[i]['cik']
            docs[cik] = download_doc(i)
        doc_list = [docs[cik]['full'] for cik in docs]

        # build tfidf for 1 and 2 gram
        self.global_tfidf_1g = TfidfVectorizer(ngram_range=(1,1), norm='l2', min_df=0.0, max_df=self.config['max_df'], use_idf=True, binary=False, token_pattern=r"(?u)\b[a-z]{3,}\b").fit(doc_list)
        self.global_tfidf_2g = TfidfVectorizer(ngram_range=(1,2), norm='l2', min_df=0.0, max_df=self.config['max_df'], use_idf=True, binary=False, token_pattern=r"(?u)\b[a-z]{3,}\b").fit(doc_list)
        log(f'Vocab size of TFIDF (1-gram): {len(self.global_tfidf_1g.vocabulary_)}')
        log(f'Vocab size of TFIDF (2-gram): {len(self.global_tfidf_2g.vocabulary_)}')

        # save sampled docs for Top2Vec
        save_pkl(docs, f'{const.DATA_OUTPUT_PATH}/sampled_docs.pkl')
        
        # release memory
        del doc_list, docs
        gc.collect()


    def load_word2vec(self):
        # download word2vec
        log(f'Downloading word2vec...')
        for i in range(20):
            try:
                wv = api.load('word2vec-google-news-300')
                break
            except:
                continue

        # get the column index for vocab overlapping with Word2Vec
        wv_vocab_list = list(wv.key_to_index)
        tfidf_vocab = self.global_tfidf_1g.vocabulary_
        tfidf_vocab_swap = {v: k for k, v in tfidf_vocab.items()}
        tfidf_1g_wv_idx = sorted([self.global_tfidf_1g.vocabulary_[x] for x in list(self.global_tfidf_1g.vocabulary_) if x in wv_vocab_list])
        tfidf_1g_wv_word = [tfidf_vocab_swap[x] for x in tfidf_1g_wv_idx]
        log(f'Vocab size of TFIDF overlapped with Word2Vec: {len(tfidf_1g_wv_idx)}')

        # extract smaller word2vec dict
        self.tfidf_1g_wv_idx = tfidf_1g_wv_idx
        wv_subset = {w : wv[w] for w in tfidf_1g_wv_word}
        self.wv_subset = np.concatenate(list(wv_subset.values())).reshape(len(wv_subset), 300)
        log(f'Shape of wv_subset: {self.wv_subset.shape}')
        del wv
        gc.collect()


    def run_preparation(self):
        log(f'Running preparation...')
        self.load_deep_learning_models()
        self.load_master_dict()
        if self.config['load_prep_data'] == False:
            self.build_tfidf_models()
            self.load_word2vec()
        else:
            log(f'Loading precomputed prep...')
            self.global_tfidf_1g = load_pkl(os.path.join(const.DATA_INPUT_PATH, self.config['global_tfidf_1g']))
            self.global_tfidf_2g = load_pkl(os.path.join(const.DATA_INPUT_PATH, self.config['global_tfidf_2g']))
            self.tfidf_1g_wv_idx = load_pkl(os.path.join(const.DATA_INPUT_PATH, self.config['tfidf_1g_wv_idx']))
            self.wv_subset = load_pkl(os.path.join(const.DATA_INPUT_PATH, self.config['wv_subset']))
            load_pkl(os.path.join(const.DATA_INPUT_PATH, self.config['sampled_docs']))
            shutil.copyfile(os.path.join(const.DATA_INPUT_PATH, self.config['sampled_docs']), os.path.join(const.DATA_OUTPUT_PATH, 'sampled_docs.pkl'))
        log(f'Completed preparation')


    def gen_signal_10k(self, cik):
        log(f'{cik}: Extracting 10-K signal...')
        df = self.master_idx_10k.loc[lambda x: x.cik==cik].sort_values('filing_date').reset_index(drop=True)
        docs = {}
        for i in range(len(df)):
            
            # load 10-K text from EDGAR html url
            url = df.iloc[i]['url_10k']
            doc_id = df.iloc[i]['doc_id']
            
            # url request
            session = requests.Session()
            retry = Retry(connect=self.config['retry_connect'], backoff_factor=self.config['retry_backoff_factor'])
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            headers = {'Host': 'www.sec.gov', 'Connection': 'close',
                    'Accept': 'application/json, text/javascript, */*; q=0.01', 'X-Requested-With': 'XMLHttpRequest',
                    'User-Agent': f"{self.config['edgar_user_agent']}{int(float(np.random.rand(1)) * 1e7)}"
                    }
            txt = session.get(url, headers=headers).text

            # clean doc, extract items
            txt = BeautifulSoup(txt, 'lxml').get_text('|', strip=True)
            txt = clean_doc1(txt)
            item_pos = find_item_pos(txt)
            doc_dict = {}
            doc_dict['full'] = txt[item_pos.iloc[0]['item_1_pos_start'] :]
            item_ptrn1 = get_item_ptrn1()
            for item in item_ptrn1:
                doc_dict[item] = txt[item_pos.iloc[0][f'{item}_pos_start'] : item_pos.iloc[0][f'{item}_pos_end']]
            for x in doc_dict:
                doc_dict[x] = clean_doc2(doc_dict[x])
            docs[doc_id] = doc_dict
            
        # generate signal
        feat_vecs = [pd.Series(list(docs.keys())).rename('doc_id')]
        feat_vecs += [gen_feat_ch_full_len(docs),
                        gen_feat_ch_item_1a_len(docs),
                        gen_feat_ch_item_1b_len(docs),
                        gen_feat_ch_item_3_len(docs),
                        gen_feat_full_cos_1gram(docs, self.global_tfidf_1g),
                        gen_feat_full_cos_2gram(docs, self.global_tfidf_2g),
                        gen_feat_full_jac_1gram(docs),
                        gen_feat_full_jac_2gram(docs),
                        gen_feat_item_1a_lev(docs),
                        gen_feat_item_7_lev(docs),
                        gen_feat_lm_postive(docs, self.positive_word_list, self.negative_word_list),
                        gen_feat_lm_uncertainty(docs, self.uncertainty_word_list),
                        gen_feat_lm_litigious(docs, self.litigious_word_list),
                        gen_feat_word2vec(docs, self.global_tfidf_1g, self.tfidf_1g_wv_idx, self.wv_subset)]
        if self.config['gpu_enabled']:
            feat_vecs += [gen_feat_sen_enc(docs, self.st_model),
                            gen_feat_item_sentiment(docs, self.fb_tokenizer, self.fb_model),
                            gen_feat_fls_sentiment(docs, self.fb_tokenizer, self.fb_model)]
        feats = pd.concat(feat_vecs, axis=1)
        return feats


    def gen_signal_10k_all_stocks(self):
        # generate signal per CIK
        pool = multiprocessing.Pool(processes=self.config['n_jobs'])
        feats = pool.map(self.gen_signal_10k, self.master_idx_10k.cik.unique().tolist())
        # feats = [self.gen_signal_10k(x) for x in self.master_idx_10k.cik.unique().tolist()]
        feats = pd.concat(feats).sort_values('doc_id').reset_index(drop=True)

        # map back to stock
        df = self.master_idx_10k[['doc_id','cik','entity','filing_date']].drop_duplicates()
        feats = feats.merge(df, how='inner', on='doc_id')
        feats = feats.merge(self.cik_map, how='inner', on='cik')
        cols = [c for c in feats if c[:5]=='feat_']
        feats = feats[[c for c in feats if c not in cols] + cols]
        log(f'Number of stocks with more than 1 row containing null 10-K signals:')
        display(feats.loc[lambda x: x.isnull().sum(axis=1) > 0].groupby('cik')['doc_id'].count().loc[lambda x: x>1])
        log(f'Sample of 10-K feats:')
        display(feats.head())

        # export
        self.feats_10k = feats
        save_pkl(feats, os.path.join(const.DATA_OUTPUT_PATH, 'feats_10k.pkl'))


    def get_10q_doc_pairs(self, docs):
        df = pd.DataFrame({'doc_id':list(docs),
                        'date':[x[11:15]+'-'+x[15:17]+'-'+x[17:19] for x in list(docs)]}) \
            .assign(date = lambda x: pd.to_datetime(x.date))
        prev_doc_id = []
        for i in range(len(df)):
            curr_date = df.iloc[i]['date']
            eps = 30
            target_date = curr_date + np.timedelta64(-365, 'D')
            lb = curr_date + np.timedelta64(-365-eps, 'D')
            ub = curr_date + np.timedelta64(-365+eps, 'D')
            target_df = df.loc[lambda x: (x.date>=lb) & (x.date<=ub)].assign(diff = lambda x: (x.date - target_date)/ np.timedelta64(1, 's'))
            if target_df.shape[0] > 0:
                prev_doc_id.append(target_df.sort_values('diff').iloc[0,0])
            else:
                prev_doc_id.append(np.NaN)
        df['prev_doc_id'] = prev_doc_id
        df = df[['doc_id','prev_doc_id']].dropna().reset_index(drop=True)
        return df


    def gen_signal_10q(self, cik):
        log(f'{cik}: Extracting 10-Q signal...')
        df = self.master_idx_10q.loc[lambda x: x.cik==cik].sort_values('filing_date').reset_index(drop=True)
        docs = {}
        for i in range(len(df)):
            
            # load 10-K text from EDGAR html url
            url = df.iloc[i]['url_10q']
            doc_id = df.iloc[i]['doc_id']
            
            # url request
            session = requests.Session()
            retry = Retry(connect=self.config['retry_connect'], backoff_factor=self.config['retry_backoff_factor'])
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            headers = {'Host': 'www.sec.gov', 'Connection': 'close',
                    'Accept': 'application/json, text/javascript, */*; q=0.01', 'X-Requested-With': 'XMLHttpRequest',
                    'User-Agent': f"{self.config['edgar_user_agent']}{int(float(np.random.rand(1)) * 1e7)}"
                    }
            txt = session.get(url, headers=headers).text

            # clean doc, extract items
            txt = BeautifulSoup(txt, 'lxml').get_text('|', strip=True)
            txt = clean_doc1(txt)
            doc_dict = {}
            doc_dict['full'] = txt
            for x in doc_dict:
                doc_dict[x] = clean_doc2(doc_dict[x])
            docs[doc_id] = doc_dict
            
        # generate year-on-year pairs of 10-Q
        doc_pairs = self.get_10q_doc_pairs(docs)
            
        # generate signal
        feat_vecs = [doc_pairs.doc_id]
        feat_vecs += [gen_feat_ch_full_len_10q(docs, doc_pairs),
                        gen_feat_full_cos_1gram_10q(docs, doc_pairs),
                        gen_feat_full_jac_1gram_10q(docs, doc_pairs),
                        gen_feat_word2vec_10q(docs, doc_pairs, self.global_tfidf_1g, self.tfidf_1g_wv_idx, self.wv_subset),
                        gen_feat_lm_postive_10q(docs, doc_pairs, self.positive_word_list, self.negative_word_list)]
        feats = pd.concat(feat_vecs, axis=1)
        log(f'Completed signal generation for CIK {cik}')
        return feats


    def gen_signal_10q_all_stocks(self):
        # generate signal per CIK
        pool = multiprocessing.Pool(processes=self.config['n_jobs'])
        feats = pool.map(self.gen_signal_10q, self.master_idx_10q.cik.unique().tolist())
        # feats = [self.gen_signal_10q(x) for x in self.master_idx_10q.cik.unique().tolist()]
        feats = pd.concat(feats).sort_values('doc_id').reset_index(drop=True)

        # map back to stock
        df = self.master_idx_10q[['doc_id','cik','entity','filing_date']].drop_duplicates()
        feats = feats.merge(df, how='inner', on='doc_id')
        feats = feats.merge(self.cik_map, how='inner', on='cik')
        cols = [c for c in feats if c[:5]=='feat_']
        feats = feats[[c for c in feats if c not in cols] + cols]
        log(f'Number of stocks with more than 1 row containing null 10-Q signals:')
        display(feats.loc[lambda x: x.isnull().sum(axis=1) > 0].groupby('cik')['doc_id'].count().loc[lambda x: x>1])
        log(f'Sample of 10-Q feats:')
        display(feats.head())

        # export
        self.feats_10q = feats
        save_pkl(feats, os.path.join(const.DATA_OUTPUT_PATH, 'feats_10q.pkl'))


    def gen_8k_feats(self):
        # get a vector of all calendar dates
        dates = pd.to_datetime(pd.Series(['2008-01-01'] * 365*11)) \
            .to_frame() \
            .rename(columns={0:'filing_date'})
        dates['filing_date'] = dates['filing_date'] + pd.Series([np.timedelta64(i, 'D') for i in range(len(dates))])

        # count the rolling 1-year number of 8-K filings
        feats_8k = []
        for cik in self.master_idx_8k.cik.unique():
            df = pd.merge(dates, self.master_idx_8k.loc[lambda x: x.cik==cik], how='left', on='filing_date') \
                .assign(filed=lambda x: x.cik.notnull().astype(int)) \
                .assign(cik=cik) \
                .loc[:,['cik','filing_date','filed']] \
                .assign(feat_cnt_8k = lambda x: x.rolling(365).filed.sum()) \
                .dropna()
            feats_8k.append(df)
        feats_8k = pd.concat(feats_8k).reset_index(drop=True)

        # export
        self.feats_8k = feats_8k
        save_pkl(feats_8k, os.path.join(const.DATA_OUTPUT_PATH, 'feats_8k.pkl'))


