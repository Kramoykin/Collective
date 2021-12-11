import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import Modelling.general_func as fnc
import shap
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:16:09 2021

@author: deutero
"""

def main(X_train, X_valid, y_train, y_valid, X_test, y_test, features):
    
    """--------------HYPER PARAMETER TUNING----------------"""
    # Get the sorted C-values dataframe
    c_df = fnc.get_best_C(X_train, y_train, X_test, y_test)
    # Choose the best C-value
    c = c_df['C_value'][0]
    
    print(c)
    
    """--------------TEST TUNED MODEL ON UNSEEN DATA----------------"""
    # Initialaze the model
    log_reg = fnc.LogisticRegression(random_state = 13, solver = 'lbfgs', C = c)
    # Fit the regression curve
    log_reg.fit(X_train, y_train)
    # Predict class labels for samples in X
    y_pred_train = log_reg.predict(X_train)
    y_pred_valid = log_reg.predict(X_valid)
    # Probability estimates (the first is for 0 the second is for 1)
    proba_valid = log_reg.predict_proba(X_valid)
    # Coefficients of the features in the decision function
    feature_coeffs = log_reg.coef_
    print("\nFeature coefficients : \n", feature_coeffs)
    
    # Classification Report
    print(classification_report(y_valid, y_pred_valid))
    # Accuracy on Valid
    print("\nThe Model validation accuracy is : ", fnc.accuracy_score(y_valid, y_pred_valid))
    # Logarithmic Loss on Valid
    print("\nThe Model Log Loss of Validating :", fnc.log_loss(y_valid, proba_valid))
    
    
    """-----------------------SHAP-----------------------------"""
    explainer = shap.LinearExplainer(log_reg, X_train, feature_dependence="independent")
    shap_values = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(shap_values, X_test, features)
    
    """--------------EXPLORING RESULTS----------------"""
    # Confusion Matrix
    cm = confusion_matrix(y_valid, y_pred_valid)
    print("\nConfusion Matrix : \n", cm)
    # Normalize the matrix
    cm_norm = cm / cm.sum(axis = 1).reshape(-1, 1)
    print("\n Normalized Confusion Matrix : \n", cm_norm)
    # Plot CM
    fnc.plot_confusion_matrix(cm_norm, classes = log_reg.classes_)
    
    # Calculate True Positive (TP), False Positive (FP), 
    # True Negative (TN), False Negative (FN) and rates (TPR, FPR, TNR, FNR)
    TP, FP, TN, FN = fnc.get_t_f(cm)
    TPR, FPR, TNR, FNR = fnc.calc_t_f_rate(cm)
    
    # ROC plot and AUC score
    logreg_auc, tresholds = fnc.roc_auc_explore(proba_valid, y_valid)
    
    print('\nThe AUC value for logistic regression model is: ', logreg_auc)
    
    # Treshold adjustment
    df_tresh = fnc.adjust_treshold(tresholds, y_valid, proba_valid)
    print('\nTreshold Adjustment: \n', df_tresh)
    best_tresh = df_tresh.iloc[0][0]
    print('\nThe best treshold value is : \n', best_tresh)
    y_pred = np.where(proba_valid[:, 1] > best_tresh, 1, 0)
    print('\nAccuracy after adjustment : ', fnc.accuracy_score(y_valid, y_pred))
    
    # CM after adjustment
    cm = confusion_matrix(y_valid, y_pred)
    cm_norm = cm / cm.sum(axis = 1).reshape(-1, 1)
    print("\nConfusion Matrix after treshold adjustment: \n", cm)
    fnc.plot_confusion_matrix(cm_norm, classes = log_reg.classes_)
    
    """---------------------DUMMY CLASSIFIER-----------------------"""
    
    from sklearn.dummy import DummyClassifier
    
    dummy_clf = DummyClassifier(strategy = "most_frequent")
    dummy_clf.fit(X_train, y_train)
    score_dummy = dummy_clf.score(X_valid, y_valid)
    
    pred_proba_dummy = dummy_clf.predict_proba(X_valid)
    log_loss_dummy = fnc.log_loss(y_valid, pred_proba_dummy) 
    
    print("\nValidating accuracy of DCLF : \n", score_dummy)
    print("\nLog loss of DCLF : \n", log_loss_dummy)