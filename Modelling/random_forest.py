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
from log_reg import get_cat
from log_reg import get_fi

# Import hand-made functional
from log_reg import get_cat
from log_reg import get_fi
from log_reg import make_tar_bin
from log_reg import plot_confusion_matrix
from log_reg import get_fi
from log_reg import roc_auc_explore
from log_reg import adjust_treshold


from itertools import product

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
plot_confusion_matrix(cm_norm, classes=rf.classes_)

"""------------------HYPERPARAMETER TUNING--------------------------------"""
# Does not returns anything because the best hyperparameters are always
# the same as default hyperparameters
tune_forest(X_train, y_train, X_test, y_test)

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
plot_confusion_matrix(cm_norm, classes=rf.classes_)

# ROC plot and AUC score
rf_auc, tresholds = roc_auc_explore(y_pred_proba)
print('\nThe AUC value for random forest model is: ', rf_auc)

# Treshold adjustment
df_tresh = adjust_treshold(tresholds, y_valid)
print('\nTreshold Adjustment: \n', df_tresh)
best_tresh = df_tresh.iloc[0][0]
print('\nThe best treshold value is : \n', best_tresh)
y_pred = np.where(y_pred_proba[:, 1] > best_tresh, 1, 0)
print('\nAccuracy after adjustment : ', accuracy_score(y_valid, y_pred))

# CM after adjustment
cm = confusion_matrix(y_valid, y_pred)
cm_norm = cm / cm.sum(axis = 1).reshape(-1, 1)
print("\nConfusion Matrix after treshold adjustment: \n", cm)
plot_confusion_matrix(cm_norm, classes = rf.classes_)


