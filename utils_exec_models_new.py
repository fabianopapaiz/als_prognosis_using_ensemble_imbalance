
import ast
import json

import utils
import pandas as pd
import numpy as np


import sklearn as sk
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, make_scorer, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
import imblearn.under_sampling as resus
import imblearn.ensemble as resemb
import imblearn.combine as reshyb
from imblearn.ensemble import BalancedBaggingClassifier


import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

import seaborn as sns
plt.style.use('seaborn-whitegrid')

# import plotly as ply
# import plotly.express as px




# CONSTANT to store the random_state stated for reproducibility issues
RANDOM_STATE = 42

N_JOBS = 7

CV_N_SPLITS = 5


def get_kfold_splits(n_splits=CV_N_SPLITS, random_state=RANDOM_STATE, shuffle_kfold=True, ):
    kfold = StratifiedKFold(
        n_splits=n_splits, 
        random_state = random_state if shuffle_kfold else None, 
        shuffle=shuffle_kfold,
    )
    return kfold



def get_default_scoring():
    # metrics to evaluate the model performance
    scoring = {
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'sensitivity': make_scorer(recall_score),
        'specificity': make_scorer(recall_score, pos_label=0),
        'f1': make_scorer(f1_score, zero_division=0.0),
        'AUC': make_scorer(roc_auc_score),
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, zero_division=0),
    }
    return scoring


def print_dataframe(filtered_cv_results):
    """Pretty print for filtered dataframe"""
    for mean_precision, std_precision, mean_recall, std_recall, params in zip(
        filtered_cv_results["mean_test_precision"],
        filtered_cv_results["std_test_precision"],
        filtered_cv_results["mean_test_recall"],
        filtered_cv_results["std_test_recall"],
        filtered_cv_results["params"],
    ):
        print(
            f"precision: {mean_precision:0.3f} (±{std_precision:0.03f}),"
            f" recall: {mean_recall:0.3f} (±{std_recall:0.03f}),"
            f" for {params}"
        )
    print()


 ## 
 # Define the strategy to select the best estimator.
 ##
def refit_strategy(cv_results):

    """The strategy defined here is to filter-out all results below a precision threshold
    of 0.98, rank the remaining by recall and keep all models with one standard
    deviation of the best by recall. Once these models are selected, we can select the
    fastest model to predict.

    Parameters
    ----------
    cv_results : dict of numpy (masked) ndarrays
        CV results as returned by the `GridSearchCV`.

    Returns
    -------
    best_index : int
        The index of the best estimator as it appears in `cv_results`.
    """
    
    # print the info about the grid-search for the different scores

    precision_threshold = 0.98

    cv_results_ = pd.DataFrame(cv_results)
    print("All grid-search results:")
    print_dataframe(cv_results_)

    # Filter-out all results below the threshold
    high_precision_cv_results = cv_results_[
        cv_results_["mean_test_precision"] > precision_threshold
    ]

    print(f"Models with a precision higher than {precision_threshold}:")
    print_dataframe(high_precision_cv_results)

    high_precision_cv_results = high_precision_cv_results[
        [
            "mean_score_time",
            "mean_test_recall",
            "std_test_recall",
            "mean_test_precision",
            "std_test_precision",
            "rank_test_recall",
            "rank_test_precision",
            "params",
        ]
    ]

    # Select the most performant models in terms of recall
    # (within 1 sigma from the best)
    best_recall_std = high_precision_cv_results["mean_test_recall"].std()
    best_recall = high_precision_cv_results["mean_test_recall"].max()
    best_recall_threshold = best_recall - best_recall_std

    high_recall_cv_results = high_precision_cv_results[
        high_precision_cv_results["mean_test_recall"] > best_recall_threshold
    ]
    print(
        "Out of the previously selected high precision models, we keep all the\n"
        "the models within one standard deviation of the highest recall model:"
    )
    print_dataframe(high_recall_cv_results)

    # From the best candidates, select the fastest model to predict
    fastest_top_recall_high_precision_index = high_recall_cv_results[
        "mean_score_time"
    ].idxmin()

    print(
        "\nThe selected final model is the fastest to predict out of the previously\n"
        "selected subset of best models based on precision and recall.\n"
        "Its scoring time is:\n\n"
        f"{high_recall_cv_results.loc[fastest_top_recall_high_precision_index]}"
    )

    return fastest_top_recall_high_precision_index


def get_classifier_class_name(classifier):
    return classifier.__class__.__name__


def create_models_RF_grid(param_grid=None, testing=False):
    # hyperparams
    max_depths = [5, 7, 10, 15] #, 25, 50]
    num_estimators = [11, 15, 21, 51] #, 75, 100, 200] 
    criterions = ['gini', 'entropy'] 
    # class_weights = [None, 'balanced', 'balanced_subsample']
    class_weights = ['balanced']


    if testing:
        max_depths = [5, 7]
        num_estimators = [5, 9] 
        criterions = ['gini']
        class_weights = ['balanced']


    if param_grid is None:
        param_grid = []

    param_grid.append(
        {
            "max_depth": max_depths,
            "n_estimators": num_estimators,
            "criterion": criterions,
            "class_weight": class_weights,
            "random_state": [RANDOM_STATE],
        }
    )    

    classifier = RandomForestClassifier()

    return classifier, param_grid



def create_models_SVM_grid(param_grid=None, testing=False):
    # hyperparams
    kernels = ['rbf', 'linear'] #, 'poly', 'sigmoid',]
    gammas = ['scale', 'auto',]
    
    # class_weights = [None, 'balanced',]
    class_weights = ['balanced',]

    Cs = [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 10] #, 100, 200, 1000, 1500, 1700, 2000]

    if testing:
        kernels = ['rbf', 'linear'] #, 'poly', 'sigmoid',]
        gammas = ['auto',]
        class_weights = ['balanced']
        Cs = [0.1, 0.3, ]


    if param_grid is None:
        param_grid = []

    param_grid.append(
        {
            "C": Cs,
            "kernel": kernels,
            "gamma": gammas,
            "class_weight": class_weights,
            "probability": [True],
            "random_state": [RANDOM_STATE],
        }
    )    

    classifier = svm.SVC()

    return classifier, param_grid




DEFAULT_SCORE = 'balanced_accuracy'

def exec_grid_search(classifier, param_grid, X, y, cv=None, 
                     n_jobs=N_JOBS, verbose=1, scoring=None, 
                     refit=None, return_train_score=False,
                     sort_results=True, dataset_info='', 
                     features_info='', 
                     X_valid=None, y_valid=None, plot_roc_curve=False):


    # get only array of output y, if it was a dataFrame
    if type(y) is pd.DataFrame:
        y = y[utils.CLASS_COLUMN].ravel()  


    if scoring is None:
        scoring = get_default_scoring()

    # define the default REFIT, if not informed
    if refit is None:
        refit = DEFAULT_SCORE

    # define the default kFold object, if not informed
    if cv is None:
        cv = get_kfold_splits()


    # create object GridSearch
    grid = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid, 
        scoring=scoring,
        cv=cv, 
        verbose=verbose,
        n_jobs=n_jobs,
        return_train_score=return_train_score,
        refit=refit,
    )

    # train the gridSearch models
    grid.fit(X, y)    

    return grid


    # df_results = get_grid_search_performances(
    #     grid_search=grid,
    #     dataset_info=dataset_info,
    #     features_info=features_info,
    #     sort_results=sort_results,
    # )


    # det_curve_data = []
    # precision_recall_curve_data = []
    # y_pred = None
    # y_pred_proba = None

    # if plot_roc_curve:
    #     # get the best classifier
    #     clf = grid.best_estimator_    

    #     # fit using all training set
    #     clf.fit(X, y)

    #     # extract classifier name, hyperparams and a model "friendly name"
    #     clf_instance = str(clf).replace('\n', '').replace(' ','').strip()
    #     estimator_name = clf_instance.split('(')[0]
    #     hyperparams = clf_instance.split('(')[1][:-1]
    #     model_desc = get_model_description(estimator_name)

    #     # make predictions using the best classifier
    #     y_pred = clf.predict(X_valid)

    #     # make predictions using probabilities, returning the class 
    #     # probabilities for sample.
    #     # The first column represents the probability of the 
    #     # negative class (Non-Short) and the second column represents 
    #     # the probability of the positive class (Short).
    #     y_pred_proba = clf.predict_proba(X_valid)
    #     y_pred_proba = y_pred_proba[:,1] # get short-survival probabilities


    #     # get performance metrics based on predict and predict_proba
    #     bal_acc, sens, spec, auc, acc, precision, f1 = get_scores_from_predict(
    #         y_validation=y_valid,
    #         y_pred=y_pred,
    #         y_pred_proba=y_pred_proba,
    #         print_info=False,
    #     )

    #     # get performances from the best classifier of the grid serach
    #     df_results = get_grid_search_performances(
    #         # grid_search=grid,
    #         performances=[model_desc, estimator_name, hyperparams, bal_acc, sens, spec, auc, acc, precision, f1],
    #         dataset_info=dataset_info,
    #         features_info=features_info,
    #         sort_results=sort_results,
    #     )



    #     # =======================================================
    #     # Detection Error Tradeoff (DET) curve
    #     # =======================================================
    #     fpr, fnr, thresholds = sk.metrics.det_curve(y_valid, y_pred_proba)
    #     det_curve_data = [estimator_name, fpr, fnr, thresholds]
        

    #     # =======================================================
    #     # Precision-Recall curve
    #     # =======================================================
    #     precision, recall, thresholds = sk.metrics.precision_recall_curve(y_valid, y_pred_proba)
    #     au_prec_recall_curve = sk.metrics.auc(recall, precision)
    #     precision_recall_curve_data = [estimator_name, precision, recall, thresholds]


    #     # =======================================================
    #     # ROC curve
    #     # =======================================================
    #     fpr, tpr, thresholds = sk.metrics.roc_curve(y_valid, y_pred_proba)
    #     roc_auc = sk.metrics.auc(fpr, tpr)
    #     roc_curve_data = [estimator_name, fpr, tpr, thresholds]


    #     # =======================================================
    #     # Predictions data (using predict and predict_proba)
    #     # =======================================================
    #     predictions_data = [estimator_name, y_pred, y_pred_proba]


    #     # print some info
    #     print(f'Classifier: {estimator_name}')
    #     print(f'  Area under ROC: {roc_auc:.2f}; Area under Prec-Recall curve: {au_prec_recall_curve:.2f}')
    #     print()



    # # return all information
    # return grid, df_results, det_curve_data, precision_recall_curve_data, roc_curve_data, predictions_data

