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

def bare_rf_clf(x_train, y_train, x_test, y_test):
    start_time = time.time()

    rf_clf = RandomForestClassifier(random_state = 107)
    rf_clf.fit(x_train, y_train)
    pred_rf_clf = rf_clf.predict(x_test)

    end_time = time.time()
    print("Classification Report")
    print(classification_report(y_test, pred_rf_clf, digits=4))
    print("Accuracy : {:.4f}".format(accuracy_score(y_test, pred_rf_clf)))
    print("Time : {:.2f} sec".format(end_time - start_time))

    # Feature importance
    importances = rf_clf.feature_importances_
    if isinstance(x_train, pd.DataFrame):
        feature_names = x_train.columns
    else:
        feature_names = [f'feature {i}' for i in range(x_train.shape[1])]

    features_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print("Top 10 Feature Importances:")
    for i in range(len(features_importances)):
        print(f"{i + 1}. {features_importances[i][0]} (importance: {features_importances[i][1]})")

    return features_importances


def bare_xtree_clf(x_train, y_train, x_test, y_test):
    start_time = time.time()

    xtree_clf = ExtraTreesClassifier(random_state=136)
    xtree_clf.fit(x_train, y_train)
    pred_xtree_clf = xtree_clf.predict(x_test)

    end_time = time.time()

    print("Classification Report")
    print(classification_report(y_test, pred_xtree_clf, digits=4))
    print("Accuracy : {:.4f}".format(accuracy_score(y_test, pred_xtree_clf)))
    print("Time : {:.2f} sec".format(end_time - start_time))

     # Feature importance
    importances = xtree_clf.feature_importances_
    if isinstance(x_train, pd.DataFrame):
        feature_names = x_train.columns
    else:
        feature_names = [f'feature {i}' for i in range(x_train.shape[1])]

    features_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print("Top 10 Feature Importances:")
    for i in range(len(features_importances)):
        print(f"{i + 1}. {features_importances[i][0]} (importance: {features_importances[i][1]})")

    return features_importances

def tunned_xgb_clf(x_train, y_train, x_test, y_test):
    start_time = time.time()

    xgb_clf = XGBClassifier(n_estimators = 1100, max_depth = 6, subsample = 1.0, learning_rate = 0.01,
                            n_jobs = -1, random_state = 100)
    xgb_clf.fit(x_train, y_train)
    pred_xgb_clf = xgb_clf.predict(x_test)

    end_time = time.time()

    print("Classification Report")
    print(classification_report(y_test, pred_xgb_clf, digits=4))
    print("Accuracy : {:.4f}".format(accuracy_score(y_test, pred_xgb_clf)))
    print("Time : {:.2f} sec".format(end_time - start_time))

    importances = xgb_clf.feature_importances_
    if isinstance(x_train, pd.DataFrame):
        feature_names = x_train.columns
    else:
        feature_names = [f'feature {i}' for i in range(x_train.shape[1])]

    features_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print("Top 10 Feature Importances:")
    for i in range(len(features_importances)):
        print(f"{i + 1}. {features_importances[i][0]} (importance: {features_importances[i][1]})")

    return features_importances

bare_rf_importances = bare_rf_clf(smote_x_train, smote_y_train, x_test, y_test)
bare_xtree_importances = bare_xtree_clf(smote_x_train, smote_y_train, x_test, y_test)
tunned_xgb_importances = tunned_xgb_clf(smote_x_train, smote_y_train, x_test, y_test)

