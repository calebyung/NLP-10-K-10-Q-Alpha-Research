# load main config
with open('config.yml') as file:
    config = yaml.safe_load(file)   
    file.close()

from src.util import *
import src.constants as c



# import libraries
import os
from os.path import isfile, isdir, join
import numpy as np
from datetime import datetime, date
import time
from dateutil.relativedelta import relativedelta
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from IPython.display import display
from zipfile import ZipFile
import pickle
import unicodedata
import pytz
from joblib import Parallel, delayed
import shutil
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
import yaml

import warnings
warnings.filterwarnings("ignore")

# quandl
import quandl
quandl.ApiConfig.api_key = config['QUANDL_KEY']

# Alpha Vantage
import alpha_vantage
from alpha_vantage.timeseries import TimeSeries
av_token = config['AV_KEY']

import yfinance as yf

# pandas
import pandas as pd
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None


signal_.signal(signal_.SIGALRM, timeout_handler)



