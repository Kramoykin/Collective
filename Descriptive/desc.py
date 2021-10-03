# Import modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Loading the datasets
df_mat = pd.read_csv('student-mat.csv')
df_por = pd.read_csv('student-por.csv')

#print(df.head())

## Statistical info

print(df_por.describe())

print(df_mat.describe())

# Data type of attributes
#print(df.info())

# Check unique data in dataset
#print(df.apply(lambda x: len(x.unique())))

## Preprocessing data

# Checking for null values
#print(df.isnull().sum())

# Check for categorical attributes
cat_col = []
for x in df_mat.dtypes.index:
    if df_mat.dtypes[x] == 'object':
        cat_col.append(x)
#print(cat_col)

# Print the categorical columns
#for col in cat_col:
    #print(col)
    #print(df[col].value_counts())
    #print()
    
## Exploratory Analysis

# Categorical Attributes Plots
for col in cat_col:
    name_mat = "./pic/" + str(col) + "_mat" + ".png"
    name_por = "./pic/" + str(col) + "_mat" + ".png"
    name_com = "./pic/" + str(col) + "_com" + ".png"
    # Plot mat
    plt.figure(figsize = (5, 5))
    sns.countplot(df_mat[col])
    plt.savefig(name_mat)
    # Plot por
    plt.figure(figsize = (5, 5))
    sns.countplot(df_por[col])
    plt.savefig(name_por)
    # Plot both
    plt.figure(figsize = (5, 5))
    fig, ax = plt.subplots(1,2, sharey = True)
    sns.countplot(df_mat[col], ax = ax[0]).set(title = "Mat")
    sns.countplot(df_por[col], ax = ax[1]).set(title = "Por")
    plt.savefig(name_com)

# Not categorical columns    
nocat_col = [x for x in df_mat if x not in cat_col]

# Attributes with small amount of possible values
some_val_col = []
some_val_col.append('age')
for col in nocat_col:
    if (len(df_mat[col].unique()) == 5
        or len(df_mat[col].unique()) == 4):
        some_val_col.append(col)
#print(five_val_col)
    
# Small Values Attributes Plots
for col in some_val_col:
    name_mat = "./pic/" + str(col) + "_mat" + ".png"
    name_por = "./pic/" + str(col) + "_mat" + ".png"
    name_com = "./pic/" + str(col) + "_com" + ".png"
    # Plot mat
    plt.figure(figsize = (5, 5))
    sns.countplot(df_mat[col])
    plt.savefig(name_mat)
    # Plot por
    plt.figure(figsize = (5, 5))
    sns.countplot(df_por[col])
    plt.savefig(name_por)
    # Plot both
    plt.figure(figsize = (5, 5))
    fig, ax =plt.subplots(1,2, sharey = True)
    sns.countplot(df_mat[col], ax = ax[0]).set(title = "Mat")
    sns.countplot(df_por[col], ax = ax[1]).set(title = "Por")
    plt.savefig(name_com)

# Attributes with multiple values
mult_val_col = [x for x in nocat_col 
                if x not in some_val_col]
#print(mult_val_col)

# Multiple Values Attributes Plots
for col in mult_val_col:
    name_mat = "./pic/" + str(col) + "_mat" + ".png"
    name_por = "./pic/" + str(col) + "_mat" + ".png"
    name_com = "./pic/" + str(col) + "_com" + ".png"
    # Plot mat
    plt.figure()
    sns.histplot(df_mat[col], bins = 20)
    plt.savefig(name_mat)
    # Plot por
    plt.figure()
    sns.histplot(df_por[col], bins = 20)
    plt.savefig(name_por)
    # Plot both
    plt.figure()
    sns.histplot(df_mat[col], bins = 20)
    sns.histplot(df_por[col], bins = 20)
    plt.savefig(name_com)
    
# Correlation Matrix
# for mat
corr_mat = df_mat.corr()
plt.figure(figsize = (20, 5))
sns.heatmap(corr_mat, annot = True, cmap = 'coolwarm' )
plt.savefig("./pic/corr_mat.png")
# for por
corr_por = df_por.corr()
plt.figure(figsize = (20, 5))
sns.heatmap(corr_por, annot = True, cmap = 'coolwarm' )
plt.savefig("./pic/corr_por.png")
