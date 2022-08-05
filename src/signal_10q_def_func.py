# import project modules
from src.util import *
import constants as const
from src.text_processing import *

# import libraries
import numpy as np
import pandas as pd
import re
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score



'''
Change in length
'''
# full doc
def gen_feat_ch_full_len_10q(docs, doc_pairs):
    feat = []
    for doc_id, prev_doc_id in doc_pairs.to_records(index=False):
        l1, l2 = len(docs[prev_doc_id]['full']), len(docs[doc_id]['full'])
        if l1 > 0 and l2 > 0:
            feat.append(np.log(l2) - np.log(l1))
        else:
            feat.append(np.NaN)
    feat = pd.Series(feat) * -1
    return feat.rename('feat_ch_full_len')


'''
Document Similarity
'''
# full doc, cosine similarity, 1 gram
def gen_feat_full_cos_1gram_10q(docs, doc_pairs):
    doc_list = [doc_dict['full'] for doc_dict in docs.values()]
    vectorizer = CountVectorizer(ngram_range=(1,1), binary=False, token_pattern=r"(?u)\b[a-z]{3,}\b")
    tf_vectors = vectorizer.fit_transform(doc_list)
    feat = []
    for doc_id, prev_doc_id in doc_pairs.to_records(index=False):
        i, j = list(docs).index(prev_doc_id), list(docs).index(doc_id)
        feat.append(cosine_similarity(tf_vectors[[i,j],:])[0][1])
    feat = pd.Series(feat)
    return feat.rename('feat_full_cos_1gram')

# full doc, jaccard similarity, 1 gram
def gen_feat_full_jac_1gram_10q(docs, doc_pairs):
    doc_list = [doc_dict['full'] for doc_dict in docs.values()]
    vectorizer = CountVectorizer(ngram_range=(1,1), binary=True, token_pattern=r"(?u)\b[a-z]{3,}\b")
    tf_vectors = vectorizer.fit_transform(doc_list)
    feat = []
    for doc_id, prev_doc_id in doc_pairs.to_records(index=False):
        i, j = list(docs).index(prev_doc_id), list(docs).index(doc_id)
        feat.append(jaccard_score(tf_vectors[i,:].toarray().flatten(), tf_vectors[j,:].toarray().flatten()))
    feat = pd.Series(feat)
    return feat.rename('feat_full_jac_1gram')


'''
Dictionary based sentiment (Loughran and McDonald)
'''
# net postive words change in proportion
def gen_feat_lm_postive_10q(docs, doc_pairs, positive_word_list, negative_word_list):
    doc_list = [doc_dict['full'] for doc_dict in docs.values()]
    doc_len = pd.Series([len(x) for x in doc_list])
    vectorizer = CountVectorizer(ngram_range=(1,1), binary=False, token_pattern=r"(?u)\b[a-z]{3,}\b").fit(doc_list)
    tf_vectors = vectorizer.transform(doc_list)
    pos_target_cols = [vectorizer.vocabulary_[x] for x in positive_word_list if x in list(vectorizer.vocabulary_.keys())]
    neg_target_cols = [vectorizer.vocabulary_[x] for x in negative_word_list if x in list(vectorizer.vocabulary_.keys())]
    sen = [(tf_vectors[i,pos_target_cols].sum() - tf_vectors[i,neg_target_cols].sum()) / doc_len[i] for i in range(len(doc_list))]
    feat = []
    for doc_id, prev_doc_id in doc_pairs.to_records(index=False):
        i, j = list(docs).index(prev_doc_id), list(docs).index(doc_id)
        feat.append(sen[j] - sen[i])
    feat = pd.Series(feat)
    return feat.rename('feat_lm_postive')


'''
Word2Vec
'''
def gen_feat_word2vec_10q(docs, doc_pairs, global_tfidf_1g, tfidf_1g_wv_idx, wv_subset):
    doc_list = [doc_dict['full'] for doc_dict in docs.values()]
    tf_vectors = global_tfidf_1g.transform(doc_list)[:,tfidf_1g_wv_idx]
    tf_vectors = (tf_vectors.toarray() > 0).astype(int)
    avg_vecs = tf_vectors @ wv_subset
    feat = []
    for doc_id, prev_doc_id in doc_pairs.to_records(index=False):
        i, j = list(docs).index(prev_doc_id), list(docs).index(doc_id)
        feat.append(cosine_similarity(avg_vecs[[i,j],:])[0][1])
    feat = pd.Series(feat)
    return feat.rename('feat_word2vec')



