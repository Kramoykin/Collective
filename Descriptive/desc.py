# Import modules
import pandas as pd
import numpy as np
import pathlib
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
#import warnings

def main(pathm, pathp):
    """----------------LOAD THE DATA-----------------------------"""
    df_mat = pd.read_csv(pathm)
    df_mat.name = 'mat'
    df_por = pd.read_csv(pathp)
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



