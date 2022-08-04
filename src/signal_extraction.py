# import project modules
from src.util import *
import constants as const
from src.text_processing import *

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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import edgar
from polyleven import levenshtein
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
import nltk
from nltk import tokenize
from gensim import downloader as api
import gc


class SignalExtraction():

    def __init__(self):
        nltk.download('punkt')
        self.config = yaml.safe_load(open('config.yml'))
        self.master_idx = pd.read_csv('./data/output/master_idx.csv')
        

    def load_deep_learning_models(self):
        if self.config['gpu_enabled']:
            self.st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
            self.fb_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.fb_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.fb_model = self.fb_model.to("cuda:0")


    def load_master_dict(self):
        # load Loughran and McDonald’s Master Dictionary (2020)
        master_dict = pd.read_csv('../input/loughranmcdonald-masterdictionary-2020/LoughranMcDonald_MasterDictionary_2020.csv')
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
        # sample and clean doc
        master_idx_sampled = master_idx.groupby('cik').last().sort_values('filing_date').reset_index(drop=True)
        master_idx_sampled = master_idx \
            .sort_values(['cik','filing_date']).reset_index(drop=True) \
            .groupby('cik').last().reset_index() \

        def download_doc(i):
            url = master_idx_sampled.iloc[i]['url_10k']
            txt = requests.get(url, headers={"user-agent": f"chan_tai_man_{int(float(np.random.rand(1)) * 1e7)}@gmail.com"}).text
            txt = soup = BeautifulSoup(txt, 'lxml').get_text('|', strip=True)
            txt = clean_doc1(txt)
            item_pos = find_item_pos(txt)
            item_1 = clean_doc2(txt[item_pos.iloc[0]['item_1_pos_start'] : item_pos.iloc[0]['item_1_pos_end']])
            full = clean_doc2(txt[item_pos.iloc[0]['item_1_pos_start'] :])
            log(f'Completed downloading doc {i}')
            return {'full':full, 'item_1':item_1}
        docs = {}
        for i in range(len(master_idx_sampled)):
            cik = master_idx_sampled.iloc[i]['cik']
            docs[cik] = download_doc(i)
        doc_list = [docs[cik]['full'] for cik in docs]

        # build tfidf for 1 and 2 gram
        self.global_tfidf_1g = TfidfVectorizer(ngram_range=(1,1), norm='l2', min_df=0.0, max_df=0.7, use_idf=True, binary=False, token_pattern=r"(?u)\b[a-z]{3,}\b").fit(doc_list)
        self.global_tfidf_2g = TfidfVectorizer(ngram_range=(1,2), norm='l2', min_df=0.0, max_df=0.7, use_idf=True, binary=False, token_pattern=r"(?u)\b[a-z]{3,}\b").fit(doc_list)
        log(f'Vocab size of TFIDF (1-gram): {len(self.global_tfidf_1g.vocabulary_)}')
        log(f'Vocab size of TFIDF (2-gram): {len(self.global_tfidf_2g.vocabulary_)}')

        # release memory
        save_pkl(docs, 'docs')
        del doc_list, docs
        gc.collect()


    def load_word2vec(self):
        # download word2vec
        wv = load_pkl('../input/word2vecgooglenews300/wv')

        # get the column index for vocab overlapping with Word2Vec
        wv_vocab_list = list(wv.key_to_index)
        tfidf_vocab = global_tfidf_1g.vocabulary_
        tfidf_vocab_swap = {v: k for k, v in tfidf_vocab.items()}
        tfidf_1g_wv_idx = sorted([global_tfidf_1g.vocabulary_[x] for x in list(global_tfidf_1g.vocabulary_) if x in wv_vocab_list])
        tfidf_1g_wv_word = [tfidf_vocab_swap[x] for x in tfidf_1g_wv_idx]
        log(f'Vocab size of TFIDF overlapped with Word2Vec: {len(tfidf_1g_wv_idx)}')

        # extract smaller word2vec dict
        self.wv_subset = {w : wv[w] for w in tfidf_1g_wv_word}
        del wv
        gc.collect()


    def gen_8k_feats(self):
        # get a vector of all calendar dates
        dates = pd.to_datetime(pd.Series(['2008-01-01'] * 365*11)) \
            .to_frame() \
            .rename(columns={0:'filing_date'})
        dates['filing_date'] = dates['filing_date'] + pd.Series([np.timedelta64(i, 'D') for i in range(len(dates))])

        # count the rolling 1-year number of 8-K filings
        feats_8k = []
        for cik in master_idx_8k.cik.unique():
            df = pd.merge(dates, master_idx_8k.loc[lambda x: x.cik==cik], how='left', on='filing_date') \
                .assign(filed=lambda x: x.cik.notnull().astype(int)) \
                .assign(cik=cik) \
                .loc[:,['cik','filing_date','filed']] \
                .assign(feat_cnt_8k = lambda x: x.rolling(365).filed.sum()) \
                .dropna()
            feats_8k.append(df)
        feats_8k = pd.concat(feats_8k).reset_index(drop=True)

        # calculate Year-on-Year change
        feats_8k = feats_8k \
            .merge(cik_map, how='inner', on='cik') \
            .rename(columns={'filing_date':'date'}) \
            .loc[:,['stock','date','feat_cnt_8k']] \
            .sort_values(['stock','date']) \
            .assign(feat_cnt_8k_prev = lambda x: x.groupby('stock').feat_cnt_8k.shift(365)) \
            .assign(feat_cnt_8k_diff = lambda x: x.feat_cnt_8k - x.feat_cnt_8k_prev) \
            .assign(feat_cnt_8k = lambda x: x.feat_cnt_8k * -1,
                    feat_cnt_8k_diff = lambda x: x.feat_cnt_8k_diff * -1) \
            .loc[lambda x: x.date.isin(ret.index), ['stock','date','feat_cnt_8k','feat_cnt_8k_diff']] \
            .dropna()
        log(f'Shape of 8-K feats: {feats_8k.shape}')



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


