# Import modules
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from itertools import product
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

def get_Test_data(df_com):
    
    """-----------DATA PREPARATION-----------"""
    
    # Get the list of categorical variables
    cat_rows = get_cat(df_com)
    
    # Transform categorical variables to the numeric format
    new_df_com = pd.get_dummies(df_com, columns = cat_rows)
    
    
    # Check where the border between good and bad students placed 
    #(<=9 - bad, >9 - good)
    fig, ax = plt.subplots(1,2, sharey = True)
    sns.countplot(new_df_com.loc[new_df_com['G3'] <= 9.0]['G3'], ax = ax[0])
    sns.countplot(new_df_com.loc[new_df_com['G3'] > 9.0]['G3'], ax = ax[1])
    fig.suptitle("The final grade distribution", fontsize=14)
    
    # Transform target variable to binary variable
    new_df_com = make_tar_bin(new_df_com, 9)
    print(new_df_com)
    
    """--------------FEATURES EXPLORATION-------------"""
    
    # Get the feature importance
    df_fi = get_fi(new_df_com, 'G3')
    features = df_fi['Feature'].values
    print(features)
    
    """--------------SPLITTING THE DATA----------------"""
    
    # Split the data into X & y
    X = new_df_com[features].values
    y = new_df_com['G3']
    y = y.astype(int)
    
    # Hold-out validation
    # the first one (for training the model)
    X_train, X_test, y_train, y_test = train_test_split(X
                                                           , y
                                                           , train_size = 0.8
                                                           , test_size = 0.2
                                                           , random_state = 13
                                                           )
    # the second one (will be used after hyperparameter tuning)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train
                                                           , y_train
                                                           , train_size = 0.9
                                                           , test_size = 0.1
                                                           , random_state = 13
                                                           )
    return X_train, X_valid, y_train, y_valid, X_test, y_test

def tune_forest(X_train, y_train, X_test, y_test):
    """ 
    Tuning max_features and max_depth hyperparameters using accuracy score
    and confusion matrix for each combination of parameter values.
        Input: X_train, X_test - predictors dataframes from train and test samples
               y_train, y_test - target arrays from train and test samples
        Output: Nothing. Just prints accuracy scores and plots confusion matrices.
    """
    # Tunning Random Forest
    n_estimators = 100
    max_features = [1, 'sqrt', 'log2']
    max_depths = [None, 2, 3, 4, 5]
    # with product we can iterate through all possible combinations
    for f, d in product(max_features, max_depths): 
        # Create model with parameters combination
        rf = RandomForestClassifier(n_estimators=n_estimators, 
                                    criterion='entropy', 
                                    max_features=f, 
                                    max_depth=d, 
                                    n_jobs=2,
                                    random_state=1337)
        # Fit the model
        rf.fit(X_train, y_train)
        # Predict values 
        prediction_test = rf.predict(X=X_test)
        #Print accuracy score
        print('Classification accuracy on test set with max features = {} and max_depth = {}: {:.3f}'.format(f, d, accuracy_score(y_test,prediction_test)))
        # Plot confusion matrix
        cm = confusion_matrix(y_test, prediction_test)
        cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
        plt.figure()
        plot_confusion_matrix(cm_norm, classes=rf.classes_
                              ,title='Confusion matrix accuracy on test set with max features = {} and max_depth = {}: {:.3f}'.format(f, d, accuracy_score(y_test,prediction_test)))


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

def get_fi(df, target):
    """ 
    Get the feature importance dataset from dataframe
        using tree-based estimator. Returns only non-zero features.
        Input : df - dataframe
                target - name of the target column

        Output : di_df - features dataframe 
    """
    # Split the data  
    X = df.drop(target, axis = 1).values
    y = df[target]
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
    # Sort dataframe and delete features with 0 importance
    fi_df.sort_values('Feature_Importance'
                      , ascending = False
                      , inplace = True
                      , ignore_index = True)
    fi_df = fi_df.loc[fi_df['Feature_Importance'] > 0]
    return fi_df

def make_tar_bin(df, border):
    """ 
    Transform target variable to the binary variable (0 or 1)
        due to the border value.
        Input : df - dataframe
                border - some number (if var <= border -> var = 0,
                                      if var > border -> var = 1)
        Output : df - modified dataframe 
    """
        
    df.loc[df['G3'] <= border, ['G3']] = 0
    df.loc[df['G3'] > border, ['G3']] = 1
    return df

def plot_confusion_matrix(cm, classes = None, title = 'Confusion matrix'):
    """ 
    Plots a confusion matrix.
        Input : cm - confussion matrix
                classes - possible result of classification (0 or 1)
                title - string name of the title
    """
    plt.figure()
    if classes is not None:
        sns.heatmap(cm, xticklabels = classes, yticklabels = classes
                    , vmin = 0., vmax = 1., annot = True
                    , annot_kws = {'size' : 50})
    else:
        sns.heatmap(cm, vmin = 0., vmax = 1.)
    plt.title(title)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

def get_t_f(cm):
    """ 
    Calculates TP, FP, TN, FN values.
        Input : cm - confusion matrix
        Output : (TP, FP, TN, FN) 
    """
    TP = cm[1, 1]
    FP = cm[0, 1]
    TN = cm[0, 0]
    FN = cm[1, 0]
    return (TP, FP, TN, FN)

def calc_t_f_rate(cm):
    """ 
    Calculates TPR, FPR, TNR, FNR rates.
        Input : cm - confusion matrix
        Output : (TPR, FPR, TNR, FNR) 
    """
    TP = cm[1, 1]
    FP = cm[0, 1]
    TN = cm[0, 0]
    FN = cm[1, 0]
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = TN / (TN + FP)
    FNR = FN / (FN + TP)
    return(TPR, FPR, TNR, FNR)

def get_best_C(X_train, y_train, X_test, y_test):
    """ 
    Estimates logistic regression models with different C-parameter values
    to find the best one.
        Input : X_train - training df
                y_train - training target variable values
                X_test - testing df
                y_test - testing target variable values
        Output : df_outcomes - dataframe, containing C-values, Log loss values
                and accuracy values, sorted by Log loss values 
    """
    # Uniformely distributed in log space C-parameter values
    C_list = np.geomspace(1e-5, 1e5, 20)
    # Classification accuracy
    CA = []
    # Logarithmic loss
    Log_loss = []
    # Get all needed values
    for c in C_list:
        log_reg_tune = LogisticRegression(random_state = 13, solver = 'lbfgs', C = c)
        log_reg_tune.fit(X_train, y_train)
        score = log_reg_tune.score(X_test, y_test)
        CA.append(score)
        pred_proba_test = log_reg_tune.predict_proba(X_test)
        log_loss_tune = log_loss(y_test, pred_proba_test)
        Log_loss.append(log_loss_tune)
    # Merge values to a dataframe    
    CA2 = np.array(CA).reshape(20)
    Log_loss2 = np.array(Log_loss).reshape(20)
    outcomes = zip(C_list, CA2, Log_loss2)
    df_outcomes = pd.DataFrame(outcomes, columns = ["C_value", "Accuracy", "Log_loss"])
    # Sort dataframe
    df_outcomes.sort_values("Log_loss", ascending = True
                            , inplace = True, ignore_index = True)
    return df_outcomes

def roc_auc_explore(proba_valid, y_valid):
    """ 
    Calculates points for model ROC and random (diagonal) ROC;
    calculates AUC.
        Input: proba_valid - array of probability values predicted on the 
                             validation samples.
        Output: model_auc - AUC value of the model validation result
                tresholds - array of tresholds for future treshold adjustment
    """
    # ROC and AUC
    ## Probabilities for the positive outcome are kept
    proba_valid = proba_valid[:, 1]
    # Values for the bound roc curve and auc value
    random_proba = [0 for _ in range(len(proba_valid))]
    ## Calculate AUC values
    random_auc = roc_auc_score(y_valid, random_proba)
    model_auc = roc_auc_score(y_valid, proba_valid)
    ## Calculate ROC curves
    r_fpr, r_tpr, _ = roc_curve(y_valid, random_proba)
    lr_fpr, lr_tpr, tresholds = roc_curve(y_valid, proba_valid)
    ## Plot ROC curves
    plt.figure()
    plt.plot(r_fpr, r_tpr, linestyle='--'
           , label='Random prediction (AUROC = %0.3f)' % random_auc)
    plt.plot(lr_fpr, lr_tpr, marker='.'
           , label='Logistic Regression (AUROC = %0.3f)' % model_auc)
    # Title
    plt.title('ROC Plot')
    # Axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # Show legend
    plt.legend() 
    return model_auc, tresholds

def adjust_treshold(tresholds, y_valid, proba_valid):
    """ 
    Creating a sorted dataframe with tresholds and accuracy values to 
    choose the best one.
        Input: tresholds - array of treshold values calculated by the
               roc_auc_explore function
               y_valid - array of the target variable actual values
               from validation samples
        Output: df - threshold/accuracy values dataframe
    """
    acc_Vals = [] # Accuracy scores values for different tresholds
    for t in tresholds:
        y_pred = np.where(proba_valid[:, 1] > t, 1, 0)
        # Accuracy on Valid
        acc_val = accuracy_score(y_valid, y_pred)
        acc_Vals.append(acc_val)
    # Dictionary for dataframe
    d = {'treshold': tresholds, 'accuracy': acc_Vals}
    # Create tresholds dataframe
    df = pd.DataFrame(data = d)
    df = df.sort_values('accuracy', ascending = False)
    
    return df


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

 

