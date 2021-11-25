# Import modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
import warnings
import shap
warnings.filterwarnings('ignore')

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

def roc_auc_explore(proba_valid):
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

def adjust_treshold(tresholds, y_valid):
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

"""-----------LOAD DATASET-----------"""

df_com = pd.read_csv('common_clear.csv')

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

# G3 distribution in validation sample plot
# plt.figure()
# sns.histplot(y_valid, bins = 20, color = 'deeppink'\
#               , label = "G3 distr for both groups"\
#               , alpha = 0.5).set(ylabel = "frequency")

"""--------------HYPER PARAMETER TUNING----------------"""
# Get the sorted C-values dataframe
c_df = get_best_C(X_train, y_train, X_test, y_test)
# Choose the best C-value
c = c_df['C_value'][0]

"""--------------TEST TUNED MODEL ON UNSEEN DATA----------------"""
# Initialaze the model
log_reg = LogisticRegression(random_state = 13, solver = 'lbfgs', C = c)
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
print("\nThe Model validation accuracy is : ", log_reg.score(X_valid, y_valid))
# Logarithmic Loss on Valid
print("\nThe Model Log Loss of Validating :", log_loss(y_valid, proba_valid))


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
plot_confusion_matrix(cm_norm, classes = log_reg.classes_)

# Calculate True Positive (TP), False Positive (FP), 
# True Negative (TN), False Negative (FN) and rates (TPR, FPR, TNR, FNR)
TP, FP, TN, FN = get_t_f(cm)
TPR, FPR, TNR, FNR = calc_t_f_rate(cm)

# ROC plot and AUC score
logreg_auc, tresholds = roc_auc_explore(proba_valid)

print('\nThe AUC value for logistic regression model is: ', logreg_auc)

# Treshold adjustment
df_tresh = adjust_treshold(tresholds, y_valid)
print('\nTreshold Adjustment: \n', df_tresh)
best_tresh = df_tresh.iloc[0][0]
print('\nThe best treshold value is : \n', best_tresh)
y_pred = np.where(proba_valid[:, 1] > best_tresh, 1, 0)
cm = confusion_matrix(y_valid, y_pred)
print("\nConfusion Matrix after treshold adjustment: \n", cm)
plot_confusion_matrix(cm_norm, classes = log_reg.classes_)

"""---------------------DUMMY CLASSIFIER-----------------------"""

from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy = "most_frequent")
dummy_clf.fit(X_train, y_train)
score_dummy = dummy_clf.score(X_valid, y_valid)

pred_proba_dummy = dummy_clf.predict_proba(X_valid)
log_loss_dummy = log_loss(y_valid, pred_proba_dummy) 

print("\nValidating accuracy of DCLF : \n", score_dummy)
print("\nLog loss of DCLF : \n", log_loss_dummy)

 

