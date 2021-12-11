# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:48:51 2021

@author: deutero
"""
import pathlib
from pathlib import Path
import pandas as pd
import Modelling.log_reg_main as log_reg_main
import Modelling.random_forest as random_forest
import Descriptive.desc as desc
import Modelling.general_func as gf

pathm = Path(pathlib.Path.cwd(),'Descriptive','student-mat.csv')
pathp = Path(pathlib.Path.cwd(),'Descriptive','student-por.csv')
desc.main(pathm, pathp)


path = Path(pathlib.Path.cwd(),'Descriptive','common_clear.csv')
df_com = pd.read_csv(path)
X_train, X_valid, y_train, y_valid, X_test, y_test, features = gf.get_Test_data(df_com)
"""----------------Logistic Regression-------------------------"""

log_reg_main.main(X_train, X_valid, y_train, y_valid, X_test, y_test, features)

"""----------------Random Forest-------------------------"""
random_forest.main(X_train, X_valid, y_train, y_valid, X_test, y_test)