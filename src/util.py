import constants as const

import yaml 
import signal as signal_
import logging
import os
import warnings
from os.path import isfile, isdir, join
import numpy as np
import pandas as pd
from datetime import datetime, date
import pickle
import pytz
from matplotlib import pyplot as plt
from polyleven import levenshtein


# # signal for timing out an execution
# class TimeoutException(Exception):   # Custom exception class
#     pass
# def timeout_handler(signum, frame):   # Custom signal handler
#     raise TimeoutException

# function to initialize notebook settings
def init_notebook():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.options.mode.chained_assignment = None
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action='ignore', category=FutureWarning)

# initialize logger file
def init_logger():
    timestamp = datetime.strftime(datetime.now(tz=pytz.timezone('Hongkong')), '%Y%m%d_%H%M%S')
    f = f'./logs/debug_{timestamp}.log'
    if os.path.isfile(f):
        os.remove(f)
    logging.basicConfig(filename=f, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')

# log - to replace the print statement by adding the timestamp
def log(msg):
    now = datetime.strftime(datetime.now(tz=pytz.timezone('Hongkong')), '%Y-%m-%d %H:%M:%S')
    print(f'[{now}] {msg}')
    logging.info(msg)
    
# pickle save and load quick functions
def save_pkl(obj, filename):
    pickle.dump(obj, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    return
def load_pkl(filename):
    return pickle.load(open(filename, 'rb'))

# calculate total file size given a folder path
def get_size(path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

# function to clear all contents in the current directory
def clear_output():
    for file in os.listdir():
        os.remove(file)
    return

# function to indicate creating a new plot within the same output cell
fig_num = 0
def new_plot():
    global fig_num
    fig_num += 1
    plt.figure(fig_num)


# function to remove any rows/columns with all NaN
def df_drop_na(df):
    df = df.loc[lambda x: x.notnull().sum(axis=1) > 0]
    df = df[df.notnull().sum(axis=0).loc[lambda x: x>0].index.tolist()]
    return df

def s_drop_na(s):
    return s.loc[lambda x: x.notnull()]

def align_index(dfs):
    for i in range(len(dfs)):
        if i==0:
            idx, col = dfs[i].index, dfs[i].columns
        else:
            idx, col = idx & dfs[i].index, col & dfs[i].columns
    idx, col = idx.sort_values().tolist(), col.sort_values().tolist()
    new_dfs = tuple([df.reindex(index=idx, columns=col) for df in list(dfs)])
    return new_dfs

def align_row_index(dfs):
    for i in range(len(dfs)):
        if i==0:
            idx = dfs[i].index
        else:
            idx = idx.intersection(dfs[i].index)
    idx = idx.sort_values().tolist()
    new_dfs = tuple([df.reindex(index=idx) for df in list(dfs)])
    return new_dfs

def align_col_index(dfs):
    for i in range(len(dfs)):
        if i==0:
            col = dfs[i].columns
        else:
            col = col.intersection(dfs[i].columns)
    col = col.sort_values().tolist()
    new_dfs = tuple([df.reindex(columns=col) for df in list(dfs)])
    return new_dfs

def fast_lev_ratio(s1,s2):
    total_len = len(s1) + len(s2)
    if total_len > 0:
        return 1 - levenshtein(s1, s2) / total_len
    else:
        return np.NaN