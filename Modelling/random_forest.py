# Import modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from math import sqrt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
import warnings
import shap
warnings.filterwarnings('ignore')
# Import hand-made functional
import Modelling.general_func as gf
from itertools import product

def Main(path): 

    """-----------LOAD DATASET-----------"""
    
    df_com = path
    
    X_train, X_valid, y_train, y_valid, X_test, y_test = gf.get_Test_data(df_com)
    
    
    """---------------Fitting the model without tuning----------------------"""
    rf = RandomForestClassifier(n_estimators=100, criterion='entropy')
    rf.fit(X_train, y_train)
    prediction_test = rf.predict(X=X_test)
    
    # Accuracy on Train
    print("Training Accuracy is: ", rf.score(X_train, y_train))
    # Accuracy on Test
    print("Testing Accuracy is: ", rf.score(X_test, y_test))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, prediction_test)
    cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    gf.plot_confusion_matrix(cm_norm, classes=rf.classes_)
    
    """------------------HYPERPARAMETER TUNING--------------------------------"""
    # Does not returns anything because the best hyperparameters are always
    # the same as default hyperparameters
    gf.tune_forest(X_train, y_train, X_test, y_test)
    
    """------------------TEST TUNED MODEL ON UNSEEN DATA-----------------------"""
    y_pred = rf.predict(X=X_valid)
    y_pred_proba = rf.predict_proba(X_valid)
    
    """-------------------EXPLORING RESULTS------------------------------------"""
    # Accuracy on Valid
    print("\nValidation Accuracy is: ", accuracy_score(y_valid, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_valid, y_pred)
    cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    gf.plot_confusion_matrix(cm_norm, classes=rf.classes_)
    
    # ROC plot and AUC score
    rf_auc, tresholds = gf.roc_auc_explore(y_pred_proba, y_valid)
    print('\nThe AUC value for random forest model is: ', rf_auc)
    
    # Treshold adjustment
    df_tresh = gf.adjust_treshold(tresholds, y_valid, y_pred_proba)
    print('\nTreshold Adjustment: \n', df_tresh)
    best_tresh = df_tresh.iloc[0][0]
    print('\nThe best treshold value is : \n', best_tresh)
    y_pred = np.where(y_pred_proba[:, 1] > best_tresh, 1, 0)
    print('\nAccuracy after adjustment : ', accuracy_score(y_valid, y_pred))
    
    # CM after adjustment
    cm = confusion_matrix(y_valid, y_pred)
    cm_norm = cm / cm.sum(axis = 1).reshape(-1, 1)
    print("\nConfusion Matrix after treshold adjustment: \n", cm)
    gf.plot_confusion_matrix(cm_norm, classes = rf.classes_)


