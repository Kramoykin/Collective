# Import modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import dexplot as dxp
#import warnings

# Loading the datasets
df_mat = pd.read_csv('student-mat.csv')
df_por = pd.read_csv('student-por.csv')

#print(df.head())

## Statistical info

#print(df_por.describe())

#print(df_mat.describe())

# Data type of attributes
#print(df_mat.info())

# Check unique data in dataset
#print(df_mat.apply(lambda x: len(x.unique())))

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
    
# Checking for duplicated values
for item in df_mat.duplicated():
    if (item == True):
        print(item)
        
for item in df_por.duplicated():
    if (item == True):
        print(item)
   
## Dividing Attributes

# Not categorical columns    
nocat_col = [x for x in df_mat if x not in cat_col]

# Attributes with small amount of possible values
some_val_col = []
some_val_col.append('age')
for col in nocat_col:
    if (len(df_mat[col].unique()) == 5
        or len(df_mat[col].unique()) == 4):
        some_val_col.append(col)

# Attributes with multiple values
mult_val_col = [x for x in nocat_col 
                if x not in some_val_col]        
   
## Exploratory Analysis (without normalize)

# Categorical Attributes Plots
for col in cat_col:
    name_mat = "./pic/" + str(col) + "_mat" + ".png"
    name_por = "./pic/" + str(col) + "_por" + ".png"
    name_com = "./pic/" + str(col) + "_com" + ".png"
    # Plot mat
    plt.figure(figsize = (5, 5))
    sns.countplot(df_mat[col]).set(title = "Mat")
    plt.savefig(name_mat)
    # Plot por
    plt.figure(figsize = (5, 5))
    sns.countplot(df_por[col]).set(title = "Por")
    plt.savefig(name_por)
    # Plot both
    plt.figure(figsize = (5, 5))
    fig, ax = plt.subplots(1,2, sharey = True)
    sns.countplot(df_mat[col], ax = ax[0]).set(title = "Mat")
    sns.countplot(df_por[col], ax = ax[1]).set(title = "Por")
    plt.savefig(name_com)

# Small Values Attributes Plots
for col in some_val_col:
    name_mat = "./pic/" + str(col) + "_mat" + ".png"
    name_por = "./pic/" + str(col) + "_mat" + ".png"
    name_com = "./pic/" + str(col) + "_com" + ".png"
    # Plot mat
    plt.figure(figsize = (5, 5))
    sns.countplot(df_mat[col]).set(title = "Mat")
    plt.savefig(name_mat)
    # Plot por
    plt.figure(figsize = (5, 5))
    sns.countplot(df_por[col]).set(title = "Por")
    plt.savefig(name_por)
    # Plot both
    plt.figure(figsize = (5, 5))
    fig, ax =plt.subplots(1,2, sharey = True)
    sns.countplot(df_mat[col], ax = ax[0]).set(title = "Mat")
    sns.countplot(df_por[col], ax = ax[1]).set(title = "Por")
    plt.savefig(name_com)

# Multiple Values Attributes Plots
for col in mult_val_col:
    name_mat = "./pic/" + str(col) + "_mat" + ".png"
    name_por = "./pic/" + str(col) + "_por" + ".png"
    name_com = "./pic/" + str(col) + "_com" + ".png"
    # Plot mat
    plt.figure()
    sns.histplot(df_mat[col], bins = 20, color = 'deeppink', alpha = 0.5\
                 ).set(ylabel = "counts", title = "Mat")
    plt.savefig(name_mat)
    # Plot por
    plt.figure()
    sns.histplot(df_por[col], bins = 20, color = 'lightseagreen', alpha = 0.5\
                 ).set(ylabel = "counts", title = "Por")
    plt.savefig(name_por)
    # Plot both
    fig, axes = plt.subplots(1, 1)
    axes.set_title("Mat + Port")
    sns.histplot(df_mat[col], bins = 20, color = 'deeppink', alpha = 0.5\
                 ).set(ylabel = "counts", title = "Mat")
    sns.histplot(df_por[col], bins = 20, color = 'lightseagreen', alpha = 0.5\
                 ).set(ylabel = "counts", title = "Mat+Por")
    axes.legend(['Mat', 'Port'])
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

## Explanatory analysis (Normalized)

## Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cat_col:
    df_mat[col] = le.fit_transform(df_mat[col])
    df_por[col] = le.fit_transform(df_por[col])

# Small Values Attributes Normalized Plots
for col in some_val_col:
    # Normalization stuff
    x_mat = df_mat[col]
    x_por = df_por[col]
    per_mat = lambda i: len(i) / (len(x_mat)) 
    per_por = lambda j: len(j) / (len(x_por))
    # Plots names
    name_mat = "./pic/" + str(col) + "_mat_norm" + ".png"
    name_por = "./pic/" + str(col) + "_por_norm" + ".png"
    name_com = "./pic/" + str(col) + "_com_norm" + ".png"
    # Plot mat
    plt.figure(figsize = (6, 6))
    sns.barplot(df_por[col], x = x_por, y = x_por\
                , estimator = per_por).set(ylabel = "percent", title = "Mat")
    plt.savefig(name_mat)
    # Plot por
    plt.figure(figsize = (6, 6))
    sns.barplot(df_mat[col], x = x_mat, y = x_mat\
                , estimator = per_mat).set(ylabel = "percent", title = "Por")
    plt.savefig(name_por)
    # Plot both
    plt.figure(figsize = (12, 6))
    fig, ax = plt.subplots(1,2, sharey = True)
    sns.barplot(df_mat[col], x = x_mat, y = x_mat, estimator = per_mat\
                , ax = ax[0]).set(ylabel = "percent", title = "Mat")
    sns.barplot(df_por[col], x = x_por, y = x_por, estimator = per_por\
                , ax = ax[1]).set(ylabel = "percent", title = "Por")
    plt.savefig(name_com)

# Categorical Attributes Normalized Plots
for col in cat_col:
    # Normalization stuff
    x_mat = df_mat[col]
    x_por = df_por[col]
    per_mat = lambda i: len(i) / (len(x_mat)) 
    per_por = lambda j: len(j) / (len(x_por))
    # Plots names
    name_mat = "./pic/" + str(col) + "_mat_norm" + ".png"
    name_por = "./pic/" + str(col) + "_por_norm" + ".png"
    name_com = "./pic/" + str(col) + "_com_norm" + ".png"
    # Plot mat
    plt.figure(figsize = (6, 6))
    sns.barplot(df_por[col], x = x_por, y = x_por\
                , estimator = per_por).set(ylabel = "percent", title = "Mat")
    plt.savefig(name_mat)
    # Plot por
    plt.figure(figsize = (6, 6))
    sns.barplot(df_mat[col], x = x_mat, y = x_mat\
                , estimator = per_mat).set(ylabel = "percent", title = "Por")
    plt.savefig(name_por)
    # Plot both
    plt.figure(figsize = (12, 6))
    fig, ax = plt.subplots(1,2, sharey = True)
    sns.barplot(df_mat[col], x = x_mat, y = x_mat, estimator = per_mat\
                , ax = ax[0]).set(ylabel = "percent", title = "Mat")
    sns.barplot(df_por[col], x = x_por, y = x_por, estimator = per_por\
                , ax = ax[1]).set(ylabel = "percent", title = "Por")
    plt.savefig(name_com)

# Multiple Values Attributes Normalized Plots
for col in mult_val_col:
    name_mat = "./pic/" + str(col) + "_mat_norm" + ".png"
    name_por = "./pic/" + str(col) + "_por_norm" + ".png"
    name_com = "./pic/" + str(col) + "_com_norm" + ".png"
    # Plot mat
    plt.figure()
    sns.histplot(df_mat[col], bins = 20, color = 'deeppink', alpha = 0.5\
                 , stat = 'density').set(ylabel = "frequency", title = "Mat")
    plt.savefig(name_mat)
    # Plot por
    plt.figure()
    sns.histplot(df_por[col], bins = 20, color = 'lightseagreen', alpha = 0.5\
                 , stat = 'density').set(ylabel = "frequency", title = "Por")
    plt.savefig(name_por)
    # Plot both
    fig, axes = plt.subplots(1, 1)
    axes.set_title("Mat + Port")
    sns.histplot(df_mat[col], bins = 20, color = 'deeppink', label = "Mat"\
                 , stat = 'density', alpha = 0.5).set(ylabel = "frequency")
    sns.histplot(df_por[col], bins = 20, color = 'lightseagreen', label = "Por"\
                 , stat = 'density', alpha = 0.5).set(ylabel = "frequency")
    axes.legend(['Mat', 'Port'])
    plt.savefig(name_com)
   
## Merging Dataset

add_por = [int(1)] * len(df_por)
add_mat = [int(0)] * len(df_mat)

df_por["target"] = add_por
df_mat["target"] = add_mat

df_common = df_por.append(df_mat)
print(df_common)

# Outlet analysis
outlet_por = df_por.loc[df_por["G3"] <= 5.0]
outlet_mat = df_mat.loc[df_mat["G3"] <= 5.0]
#print(outlet_por)
#print(outlet_mat)
# Output the outlet rows to xls files 
outlet_por.to_csv('outlet_por.xls', index = None)
outlet_mat.to_csv('outlet_mat.xls', index = None)
#print(df_por.compare(df_mat, keep_shape=True))

# Searching for equal rows

df_mat_copy = df_mat
df_por_copy = df_por

for df in [df_mat_copy, df_por_copy]:
    del df["G1"]
    del df["G2"]
    del df["G3"]
    del df["target"]
    del df["paid"]

merged_df = df_mat_copy.merge(df_por_copy, how = 'inner')
print(merged_df)



