# Import modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import warnings

def print_stat(df):
    """
    Prints dataframe description and some statistics
    , duplicate output to a file.
        Input: df - dataframe
    """
    f = open("stat.txt", "w")
    
    print("\nHead of the {} dataframe: \n{}".format(df.name, df.head()))
    f.write("\nHead of the {} dataframe: \n{}\n".format(df.name, df.head()))
    
    print("\nDescription of the {} dataframe: \n{}".format(df.name
                                                         , df.describe()))
    f.write("\nDescription of the {} dataframe: \n{}\n".format(df.name
                                                         , df.describe()))

    print("\nData types of the {} dataframe:".format(df.name))
    f.write("\nData types of the {} dataframe:\n".format(df.name))
    
    print(df.info())
    df.info(buf = f)

    print("\nUnique values in the {} dataframe: \n{}".format(df.name
                                , df.apply(lambda x: len(x.unique()))))
    f.write("\nUnique values in the {} dataframe: \n{}\n".format(df.name
                                , df.apply(lambda x: len(x.unique()))))
    
    print("\nChecking null values in the {} dataframe: \n{}".format(df.name
                                                        , df.isnull().sum()))
    f.write("\nChecking null values in the {} dataframe: \n{}\n".format(df.name
                                                        , df.isnull().sum()))
    
    print("\nNumber of duplicated values in the {} dataframe: {}".format(df.name
                                                , df_mat.duplicated().sum()))
    f.write("\nNumber of duplicated values in the {} dataframe: {}\n".format(df.name
                                                , df_mat.duplicated().sum()))
    f.close()

def get_cat(df):
    """ 
    Get from the df the list of categorical variables.
        Input : df - dataframe
        Output : cat_col - list of categorical variables names 
    """
    cat_col = []
    for x in df.dtypes.index:
        if df.dtypes[x] == 'object':
            cat_col.append(x)
    return cat_col

import matplotlib.pylab as pylab
def plot_hist(df_mat, df_por):
    """ 
    Plots unnormalized countplots for each parameter of a dataframes separately
    and common plot for both dataframes.
        Input: df_mat - math dataframe
               df_por - por dataframe
    """
    # Setting lebels font for prevent sticking
    params = {
         'axes.labelsize': 'small',
         'axes.titlesize':'small'
         }
    pylab.rcParams.update(params)
    
    # Categorical Attributes Plots
    for col in df_mat:
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
        plt.figure(figsize = (12, 6))
        fig, ax = plt.subplots(1,2, sharey = True)
        sns.countplot(df_mat[col], ax = ax[0]).set(title = "Mat")
        sns.countplot(df_por[col], ax = ax[1]).set(title = "Por")
        plt.savefig(name_com)

from sklearn.preprocessing import LabelEncoder
def encode_cat(df_mat,df_por, cat_col):
    """
    Encodes categorical attributes in dataframe to a numeric format.
        Input: df_mat - math dataframe
               df_por - por dataframe 
               cat_col - list of categorical attributes columns name
        Output: df_mat, df_por
    """
    le = LabelEncoder()
    for col in cat_col:
        df_mat[col] = le.fit_transform(df_mat[col])
        df_por[col] = le.fit_transform(df_por[col])
    return df_mat, df_por
        
def plot_hist_norm(df_mat, df_por):
    """ 
    Plots normalized countplots for each parameter of a dataframes separately
    and common plot for both dataframes Can be used only after label encoding.
        Input: df_mat - math dataframe
               df_por - por dataframe
    """
    # Setting lebels font for prevent sticking
    params = {
         'axes.labelsize': 'small',
         'axes.titlesize':'small'
         }
    pylab.rcParams.update(params)
    # With normalize
    for col in df_mat:
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
        plt.figure(figsize = (6, 6))
        fig, ax = plt.subplots(1,2, sharey = True)
        sns.barplot(df_mat[col], x = x_mat, y = x_mat, estimator = per_mat\
                    , ax = ax[0]).set(ylabel = "percent", title = "Mat")
        sns.barplot(df_por[col], x = x_por, y = x_por, estimator = per_por\
                    , ax = ax[1]).set(ylabel = "percent", title = "Por")
        plt.savefig(name_com)
        
def plot_cor_mat(df):
    """ 
    Plots correlation matrix for all features of dataframe
        Input: df - input dataframe
    """
    name = './pic/corr_' + df.name + '.png'
    corr_mat = df.corr()
    plt.figure(figsize = (20, 5))
    sns.heatmap(corr_mat, annot = True, cmap = 'coolwarm' )
    plt.savefig(name)
    
def merge_df(df_mat, df_por):
    """
    Merges two dataframes excluding duplicating rows from the biggest one.
    Input: df_mat - math dataframe
               df_por - por dataframe 
               cat_col - list of categorical attributes columns name
        Output: df_mat, df_por
    """
    # Add indexes
    df_por['ID'] = np.arange(df_por.shape[0])
    
    # Add course attribute (if port - 1, if math  - 0)
    add_por = [int(1)] * len(df_por)
    add_mat = [int(0)] * len(df_mat)
    df_por["is_por"] = add_por
    df_mat["is_por"] = add_mat
    
    # Create copies for dropping course individual columns
    df_mat_copy = df_mat.copy()
    df_por_copy = df_por.copy()
    
    # Delete course individual columns
    for df in [df_mat_copy, df_por_copy]:
        del df["G1"]
        del df["G2"]
        del df["G3"]
        del df["is_por"]
        del df["paid"]
    
    # Get ID of duplicated rows
    merged_df = df_mat_copy.merge(df_por_copy, how = 'inner')
    del_rows = merged_df['ID']
    
    # Drop duplicated rows from Portuguese
    df_por.drop(del_rows, inplace = True)
    
    # Combine two datasets and output clear common df 
    df_por.drop(columns = ['ID'], inplace = True)
    df_common_clear = df_por.append(df_mat)
    df_common_clear.to_csv('common_clear.csv', index = None)

"""----------------LOAD THE DATA-----------------------------"""
df_mat = pd.read_csv('student-mat.csv')
df_mat.name = 'mat'
df_por = pd.read_csv('student-por.csv')
df_por.name = 'por'

"""----------------PRINT STATISTICS--------------------------"""
print('-----------------DESCRIPTION AND STATISTICS-----------------')

print_stat(df_mat)

print_stat(df_por)

"""----------------FEATURE ENGINEERING----------------------"""
# Check for categorical attributes
cat_col = get_cat(df_mat)

"""----------------EXPLARATORY ANALYSIS---------------------"""
print('----------------EXPLARATORY ANALYSIS---------------------')
# Plot features distributions (without normalize)
#plot_hist(df_mat, df_por)
# Correlation matrix
plot_cor_mat(df_mat)
plot_cor_mat(df_por)
# Encode categorical features
df_mat, df_por = encode_cat(df_mat, df_por, cat_col)
# Plot features distributions (with normalize)
#plot_hist_norm(df_mat, df_por)

"""----------------DATAFRAMES MERGE-------------------------"""
merge_df(df_mat, df_por)



