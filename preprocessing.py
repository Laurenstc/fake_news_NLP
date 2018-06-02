# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import string
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk import word_tokenize, WordPunctTokenizer, regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import tensorflow as tf
%matplotlib inline

pd.set_option("max_columns", None)

# Load data
train = pd.read_csv("fake_or_real_news_training_V2.csv")
test = pd.read_csv("fake_or_real_news_test_V2.csv")

def preprocess_data(train, test):


    # Store IDs and drop
    train_IDs = train['ID']
    test_IDs = test['ID']
    train.drop(['ID'], inplace = True, axis = 1)
    test.drop(['ID'], inplace = True, axis = 1)

    # Fix 2182 - Planned Parenthood
    train.loc[2182, 'title'] = train.loc[2182, 'title'] + "; " + train.loc[2182, "text"] + "; " + train.loc[2182, "label"]
    train.loc[2182, 'text'] = train.loc[2182, 'X1']
    train.loc[2182, 'label'] = 'REAL'

    # Drop 3537 - Chart of the Day
    train.drop(3535, axis=0, inplace = True)

    # Reset index
    train.reset_index(drop=True, inplace=True)

    # Fix labels
    for i in range(len(train)):
        if train.loc[i,'X1'] is not np.nan and train.loc[i, 'X2'] is np.nan:
            train.loc[i, 'title'] = train.loc[i, 'title'] + "; " + train.loc[i, 'text']
            train.loc[i, 'text'] = train.loc[i, 'label']
            train.loc[i, 'label'] = train.loc[i, 'X1']


    # Drop X1 and X2
    train = train.drop(["X1", "X2"], axis = 1)


    # Store labels
    labels = train['label']


    # Combine train and test into new dataset called Combined and drop Label
    combined = pd.concat((train, test)).reset_index(drop=True)
    combined.drop(['label'], axis=1, inplace=True)
    print("combined size is : {}".format(combined.shape))

    # Reset index
    combined.reset_index(drop=True, inplace=True)

    return combined, labels

combined, labels = preprocess_data(train, test)
