# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:48:51 2021

@author: deutero
"""

import pathlib
from pathlib import Path
import Modelling.log_reg_main as log_reg_main

path = Path(pathlib.Path.cwd(),'Descriptive','common_clear.csv')
log_reg_main.main(path)

