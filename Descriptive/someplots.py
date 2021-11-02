# Import modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

def make_tar_bin(df, border):
    """ Transform target variable to the binary variable (0 or 1)
        due to the border value 
        Input : df - dataframe
                border - some number (if var <= border -> var = 0,
                                      if var > border -> var = 1)
        Output : df - modified dataframe """
        
    df.loc[df['G3'] <= border, ['G3']] = 0
    df.loc[df['G3'] > border, ['G3']] = 1
    return df

df_com = pd.read_csv('common_clear.csv')
df_mat = pd.read_csv('mat_new.csv')
df_por = pd.read_csv('por_clear.csv')

# Dependancy between G3 and alco for maths
df_mat = make_tar_bin(df_mat, 9.0)
df_mat['G3'] = df_mat['G3'].map({1: 'good', 0: 'bad'})
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 9))
sns.countplot(x = 'G3', hue = 'Dalc', data = df_mat, ax = ax1)
sns.countplot(x = 'G3', hue = 'Walc', data = df_mat, ax = ax2)
for ax in (ax1, ax2):
    ax.set_title('Final grade on consumption dependancy for Maths ')
    ax.set_xlabel('Final grade')
    ax.set_ylabel('Counts')
    
# Dependancy between G3 and alco for por
df_por = make_tar_bin(df_por, 9.0)
df_por['G3'] = df_por['G3'].map({1: 'good', 0: 'bad'})
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 9))
sns.countplot(x = 'G3', hue = 'Dalc', data = df_por, ax = ax1)
sns.countplot(x = 'G3', hue = 'Walc', data = df_por, ax = ax2)
for ax in (ax1, ax2):
    ax.set_title('Final grade on consumption dependancy for Ports ')
    ax.set_xlabel('Final grade')
    ax.set_ylabel('Counts')
    
# Dependancy between G3 and alco for both
df_com = make_tar_bin(df_com, 9.0)
df_com['G3'] = df_com['G3'].map({1: 'good', 0: 'bad'})
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 9))
sns.countplot(x = 'G3', hue = 'Dalc', data = df_com, ax = ax1)
sns.countplot(x = 'G3', hue = 'Walc', data = df_com, ax = ax2)
for ax in (ax1, ax2):
    ax.set_title('Final grade on consumption dependancy for Both ')
    ax.set_xlabel('Final grade')
    ax.set_ylabel('Counts')