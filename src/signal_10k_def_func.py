# import project modules
from src.util import *
import src.constants as const
from src.text_processing import *

# import libraries
import numpy as np
import pandas as pd
import re
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import torch
from nltk import tokenize



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
def gen_feat_full_cos_1gram(docs, global_tfidf_1g):
    doc_list = [doc_dict['full'] for doc_dict in docs.values()]
    tf_vectors = global_tfidf_1g.transform(doc_list)
    feat = pd.Series([cosine_similarity(tf_vectors[i-1:i+1,:])[0][1] if i > 0 else np.NaN for i in range(len(doc_list))])
    return feat.rename('feat_full_cos_1gram')

# full doc, cosine similarity, 2 gram
def gen_feat_full_cos_2gram(docs, global_tfidf_2g):
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
def gen_feat_lm_postive(docs, positive_word_list, negative_word_list):
    doc_list = [doc_dict['full'] for doc_dict in docs.values()]
    doc_len = pd.Series([len(x) for x in doc_list])
    vectorizer = CountVectorizer(ngram_range=(1,1), binary=False, token_pattern=r"(?u)\b[a-z]{3,}\b").fit(doc_list)
    tf_vectors = vectorizer.transform(doc_list)
    pos_target_cols = [vectorizer.vocabulary_[x] for x in positive_word_list if x in list(vectorizer.vocabulary_.keys())]
    neg_target_cols = [vectorizer.vocabulary_[x] for x in negative_word_list if x in list(vectorizer.vocabulary_.keys())]
    feat = pd.Series([(tf_vectors[i,pos_target_cols].sum() - tf_vectors[i,neg_target_cols].sum()) / doc_len[i] for i in range(len(doc_list))]).diff()
    return feat.rename('feat_lm_postive')

# uncertainty words change in proportion
def gen_feat_lm_uncertainty(docs, uncertainty_word_list):
    doc_list = [doc_dict['full'] for doc_dict in docs.values()]
    doc_len = pd.Series([len(x) for x in doc_list])
    vectorizer = CountVectorizer(ngram_range=(1,1), binary=False, token_pattern=r"(?u)\b[a-z]{3,}\b").fit(doc_list)
    tf_vectors = vectorizer.transform(doc_list)
    target_cols = [vectorizer.vocabulary_[x] for x in uncertainty_word_list if x in list(vectorizer.vocabulary_.keys())]
    feat = pd.Series([tf_vectors[i,target_cols].sum() / doc_len[i] for i in range(len(doc_list))]).diff()
    feat = feat * -1
    return feat.rename('feat_lm_uncertainty')

# uncertainty words change in proportion
def gen_feat_lm_litigious(docs, litigious_word_list):
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
def gen_feat_sen_enc(docs, st_model):
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
def gen_feat_item_sentiment(docs, fb_tokenizer, fb_model):
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
def gen_feat_fls_sentiment(docs, fb_tokenizer, fb_model):
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
def gen_feat_word2vec(docs, global_tfidf_1g, tfidf_1g_wv_idx, wv_subset):
    doc_list = [doc_dict['full'] for doc_dict in docs.values()]
    tf_vectors = global_tfidf_1g.transform(doc_list)[:,tfidf_1g_wv_idx]
    tf_vectors = (tf_vectors.toarray() > 0).astype(int)
#     tf_vectors = tf_vectors / tf_vectors.sum(axis=1)
    avg_vecs = tf_vectors @ wv_subset
    feat = pd.Series([cosine_similarity(avg_vecs[i-1:i+1,:])[0][1] if i > 0 else np.NaN for i in range(len(doc_list))])
    return feat.rename('feat_word2vec')
