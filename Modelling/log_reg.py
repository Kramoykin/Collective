# Import modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier

def get_cat(df):
    """ Get from the df the list of categorical variables
        Input : df - dataframe
        Output : cat_col - list of categorical variables names """
    cat_col = []
    for x in df.dtypes.index:
        if df.dtypes[x] == 'object':
            cat_col.append(x)
    return cat_col

def get_fi(df, X, y, target):
    """ Get the feature importance dataset from dataframe 
        using tree-based estimator.
        Input : df - dataframe
                target - name of the target column
                X - dataframe without target variable
                y - target variable column
        Output : di_df - features dataframe """
    # Classify variables
    dt = DecisionTreeClassifier(random_state = 15
                              , criterion = 'entropy'
                              , max_depth = 10)
    dt.fit(X,y)
    # Get features
    fi_col = []
    fi = []
    for i, col in enumerate(df.drop(target, axis = 1)):
        fi_col.append(col)
        fi.append(dt.feature_importances_[i])
    # Combine features to dataframe
    fi_df = zip(fi_col, fi)
    fi_df = pd.DataFrame(fi_df, columns = ['Feature', 'Feature_Importance'])
    return fi_df

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

"""-----------LOAD DATASET-----------"""

df_com = pd.read_csv('common_clear.csv')

"""-----------DATA PREPARATION-----------"""

# Get the list of categorical variables
cat_rows = get_cat(df_com)

# Transform categorical variables to the numeric format
new_df_com = pd.get_dummies(df_com, columns = cat_rows)


# Check where the border between good and bad students placed 
#(<=9 - bad, >9 - good)
plt.figure()
sns.countplot(new_df_com.loc[new_df_com['G3'] <= 9.0]['G3'])

# Transform target variable to binary variable
new_df_com = make_tar_bin(new_df_com, 9)
print(new_df_com['G3'])

# Split the data into X & y
X = new_df_com.drop('G3', axis = 1).values
y = new_df_com['G3']
# Get the feature importance dataframe
df_fi = get_fi(new_df_com, X, y, 'G3')
print(df_fi)










