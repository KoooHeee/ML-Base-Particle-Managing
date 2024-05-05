import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import math
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

'''
bare_rf_importances = bare_rf_clf(smote_x_train, smote_y_train, x_test, y_test)
bare_xtree_importances = bare_xtree_clf(smote_x_train, smote_y_train, x_test, y_test)
tunned_xgb_importances = tunned_xgb_clf(smote_x_train, smote_y_train, x_test, y_test)
'''

bare_rf_importances_dict = dict(bare_rf_importances)
bare_xtree_importances_dict = dict(bare_xtree_importances)
tunned_xgb_importances_dict = dict(tunned_xgb_importances)

def update_and_combine(original_dict, new_dict):
    for key, value in new_dict.items():
        if key in original_dict:
            original_dict[key] += value
        else:
            original_dict[key] = value

treebase_imp_dict = {}
update_and_combine(treebase_imp_dict, bare_rf_importances_dict)
update_and_combine(treebase_imp_dict, bare_xtree_importances_dict)
update_and_combine(treebase_imp_dict, tunned_xgb_importances_dict)

# Aggregate Step
treebase_imp_step_dict = {1100 : 0, 1200 : 0, 1300 :0, 1400 : 0, 1500 : 0, 1600 : 0, 1700 : 0, 1800 : 0, 1900 : 0, 2000 : 0}

for key, value in treebase_imp_dict.items():
    step = int(key.split('_')[0])
    if step in treebase_imp_step_dict:
        treebase_imp_step_dict[step] = value + treebase_imp_step_dict[step]

sorted_treebase_imp_step_dict = dict(sorted(treebase_imp_step_dict.items(), key=lambda item: item[1], reverse=True))

step_key = list(sorted_treebase_imp_step_dict.keys())
step_imp = list(sorted_treebase_imp_step_dict.values())

sorted_steps_in_order = sorted(treebase_imp_step_dict.keys())
step_to_index = {step: index + 1 for index, step in enumerate(sorted_steps_in_order)}


# Aggregate Sensor
treebase_imp_para_list = []

for key, value in treebase_imp_dict.items():
    treebase_imp_para_list.append(key.split('_')[1])

treebase_imp_para_dict = {}
treebase_imp_para_dict_keys = list(set(treebase_imp_para_list))

treebase_imp_para_dict = {key: 0 for key in treebase_imp_para_dict_keys}

for key, value in treebase_imp_dict.items():
    para = key.split('_')[1]
    treebase_imp_para_dict[para] += value

sorted_treebase_imp_para_dict = dict(sorted(treebase_imp_para_dict.items(), key=lambda item: item[1], reverse=True))

key_list = list()
val_list = list()
for key,val in sorted_treebase_imp_para_dict.items() :
    key_list.append(key)
    val_list.append(val)

    
