# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:48:51 2021

@author: deutero
"""
import pathlib
from pathlib import Path
import Modelling.log_reg_main as log_reg_main
import Modelling.random_forest as random_forest
import Descriptive.desc as desc

pathm = Path(pathlib.Path.cwd(),'Descriptive','student-mat.csv')
pathp = Path(pathlib.Path.cwd(),'Descriptive','studen-por.csv')
desc.main(pathm, pathp)


pathCC = Path(pathlib.Path.cwd(),'Descriptive','common_clear.csv')
"""----------------Logistic Regression-------------------------"""

log_reg_main.main(pathCC)

"""----------------Random Forest-------------------------"""
random_forest.main(pathCC)