# import libraries
import os
from os.path import isfile, isdir, join
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import time
from bs4 import BeautifulSoup
import re
from IPython.display import display
from zipfile import ZipFile
import pickle
import unicodedata
import pytz
from joblib import Parallel, delayed
import shutil
import random
import requests
import gc
import math
from top2vec import Top2Vec





# load documents (Item 1 - Business)
docs = load_pkl('../input/hkml-signal-extraction-pre/docs')
documents = [docs[cik]['item_1'] for cik in docs]
document_ids = list(docs)

# train model
model = Top2Vec(documents = documents, 
                embedding_model = 'universal-sentence-encoder',
                document_ids = document_ids,
                workers = -1)


# CIK to topic mapping
doc_topics_ = model.get_documents_topics(doc_ids=document_ids)
doc_topics = pd.DataFrame({'cik':document_ids, 'topic':doc_topics_[0], 'topic_words':[', '.join(list(x)) for x in list(doc_topics_[2][:,:10])]})
doc_topics


# number of topics
n_topic = model.get_num_topics()

# topic description
size, topic_num = model.get_topic_sizes()
topics = pd.DataFrame({'topic_num':topic_num, 'size':size})
topic_words, word_scores, topic_nums = model.get_topics(n_topic)
df = pd.DataFrame({'topic_num':topic_nums, 'topic_words':[', '.join(list(x)) for x in list(topic_words[:,:10])]})
topics = topics.merge(df, how='inner', on='topic_num')
display(topics)


# topic words visualised
for i in topics.topic_num:
    model.generate_topic_wordcloud(i)


# export
# save_pkl(model, 'model')
save_pkl(doc_topics, 'doc_topics')
save_pkl(topics, 'topics')


