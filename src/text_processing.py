# import project modules
from src.util import *
import constants as const

# import libraries
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import re
from IPython.display import display
import unicodedata



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
    item_ptrn1 = get_item_ptrn1()
    for item in item_ptrn1:
        
        # pattern 1 (normal + upper)
        pos = [(m.start(), m.end()) for m in re.finditer(item_ptrn1[item], doc)] + [(m.start(), m.end()) for m in re.finditer(item_ptrn1[item].upper(), doc)]
        pos = dedup_pos(pos)
        log(f'[{item}] After attempt 1 yielded {len(pos)} matches') if log_mode==True else None

        # pattern 2 ("Item" as optional, normal + upper)
        item_ptrn2 = get_item_ptrn2()
        if len(pos) == 0 or (len(pos) == 1 and len(re.findall(rf"{w('table')}{s()}{w('of')}{s()}{w('content')}", doc.lower())) > 0 and pos[0][0] < 7000):
            pos = pos + [(m.start(), m.end()) for m in re.finditer(item_ptrn2[item], doc)] + [(m.start(), m.end()) for m in re.finditer(item_ptrn2[item].upper(), doc)]
            pos = dedup_pos(pos)
            log(f'[{item}] After attempt 2 yielded {len(pos)} matches') if log_mode==True else None

        # pattern 3
        item_ptrn3 = get_item_ptrn3()
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
    item_ptrn1 = get_item_ptrn1()
    for item in item_ptrn1:
        print(f'{item}: {doc_dict[item][:n]}........{doc_dict[item][-n:]}')
    return