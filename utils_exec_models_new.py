
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


def sort_performances_results(df, cols_order_to_sort=['balanced_accuracy', 'sensitivity', 'specificity', 'fit_time'], cols_to_return=None):
    df_bests = df.sort_values(
        cols_order_to_sort, 
        ascending=[False, False, False, True]
    ).copy()
    if cols_to_return is not None:
        return df_bests[cols_to_return]
    else:
        return df_bests


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
        'auc': 'roc_auc', #make_scorer(roc_auc_score),
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
    class_weights = [None, 'balanced', 'balanced_subsample']
    # class_weights = ['balanced']


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


def create_models_DT_grid(param_grid=None, testing=False):
    # hyperparams
    max_depths = [3, 4, 5, 7, 9, 10, 15, 25] #, 50]
    criterions = ['gini', 'entropy'] #, 'log_loss'] LOG-LOSS DOESN'T WORK
    class_weights = [None, 'balanced']


    if testing:
        max_depths = [5, 7]
        criterions = ['gini', 'entropy']
        class_weights = ['balanced']


    if param_grid is None:
        param_grid = []

    param_grid.append(
        {
            "max_depth": max_depths,
            "criterion": criterions,
            "class_weight": class_weights,
            "random_state": [RANDOM_STATE],
        }
    )    

    classifier = DecisionTreeClassifier()

    return classifier, param_grid


def create_models_NB_Complement_grid(param_grid=None, testing=False):
    # hyperparams
    alphas = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 15.0]
    norms = [False, True]
    force_alphas = [False, True]

    if testing:
        alphas = [0.1, 0.5]
        norms = [False]
        force_alphas = [False]


    if param_grid is None:
        param_grid = []

    param_grid.append(
        {
            "alpha": alphas,
            "norm": norms,
            "force_alpha": force_alphas,
        }
    )    

    classifier = ComplementNB()

    return classifier, param_grid


def create_models_kNN_grid(param_grid=None, testing=False):
    # hyperparams
    weights = ['uniform', 'distance']
    distance_metrics = [
        'euclidean',
        'manhattan',
        'chebyshev',
    ]
    #kNN
    ks = [3, 5, 9, 15] 


    if testing:
        weights = ['distance']
        distance_metrics = ['manhattan']
        #kNN
        ks = [5] 


    if param_grid is None:
        param_grid = []

    param_grid.append(
        {
            "n_neighbors": ks,
            "weights": weights,
            "metric": distance_metrics,
        }
    )    

    classifier = KNeighborsClassifier()

    return classifier, param_grid


def create_models_RadiusNN_grid(param_grid=None, testing=False):
    # hyperparams
    weights = ['uniform', 'distance']
    distance_metrics = [
        'euclidean',
        'manhattan',
        'chebyshev',
    ]

    #radius
    radius_set = [0.3, 0.5, 0.7, 1.0] 
    leaf_sizes = [50, 100, 200, 300, 500] 
    outlier_labels = [0, 1]

    if testing:
        weights = ['distance']
        distance_metrics = ['manhattan']
        #kNN
        ks = [5] 
        #radius
        radius_set = [0.3] 
        leaf_sizes = [50] 
        outlier_labels = [1]


    if param_grid is None:
        param_grid = []


    param_grid.append(
        {
            "radius": radius_set,
            "weights": weights,
            "leaf_size": leaf_sizes,
            "outlier_label": outlier_labels,
            "metric": distance_metrics,
        }
    )    

    classifier = RadiusNeighborsClassifier()


    return classifier, param_grid


def create_models_NB_Gaussian_grid(param_grid=None, testing=False):

    param_grid = [{}]    

    classifier = GaussianNB()

    return classifier, param_grid


def create_models_NN_grid(qty_features, param_grid=None, testing=False):
    # hyperparams
    max_iter = [2000]
    layers = [
        (30),
        (30, 30),
        (30, 30, 30),
        (qty_features,),
        (qty_features, qty_features),
        (qty_features, qty_features, qty_features),
        (qty_features, (qty_features*2)),
        (qty_features, (qty_features*2), qty_features),
        (qty_features, (qty_features*2), (qty_features*2), qty_features),
    ]
    alphas = [0.0001, 0.00001, 0.05, 0.1, 0.3, 0.5]
    activations = ['tanh', 'relu']
    solvers = ['sgd', 'adam']
    learning_rates = ['constant','adaptive']
    learning_rate_init = [0.1, 0.01, 0.3, 0.03, 0.5, 0.7]


    if testing:
        max_iter = [2000]
        layers = [(qty_features), (30, 30)]
        alphas = [0.1, 0.3]
        activations = ['relu']
        solvers = ['sgd']
        learning_rates = ['constant']
        learning_rate_init = [0.1, 0.01]


    if param_grid is None:
        param_grid = []

    param_grid.append(
        {
            "max_iter": max_iter,
            "hidden_layer_sizes": layers,
            "alpha": alphas,
            "activation": activations,
            "solver": solvers,
            "learning_rate": learning_rates,
            "learning_rate_init": learning_rate_init,
            "random_state": [RANDOM_STATE],
        }
    )    
   
    classifier = MLPClassifier()

    return classifier, param_grid


def create_models_BalancedBagging_grid(classifiers, param_grid=None, testing=False):
    # hyperparams
    num_estimators = [11, 15, 51, 75, 101, 201, 301]
    sampling_strategies = ['all', 'majority', 'auto']
    warm_starts = [False, True]

    if testing:
        num_estimators = [3] 
        classifiers = [classifiers[0]]
        warm_starts = [False]

    if param_grid is None:
        param_grid = []

    param_grid.append(
        {
            "estimator": classifiers,
            "n_estimators": num_estimators,
            "sampling_strategy": sampling_strategies,
            "warm_start": warm_starts,
            "random_state": [RANDOM_STATE],
        }
    )    

    classifier = BalancedBaggingClassifier()

    return classifier, param_grid


def create_models_SVM_grid(param_grid=None, testing=False):
    # hyperparams
    kernels = ['rbf', 'linear'] #, 'poly', 'sigmoid',]
    gammas = ['scale', 'auto',]
    
    class_weights = [None, 'balanced',]
    # class_weights = ['balanced',]

    Cs = [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 10] #, 100, 200, 1000, 1500, 1700, 2000]

    if testing:
        kernels = ['rbf', 'linear'] #, 'poly', 'sigmoid',]
        # gammas = ['auto',]
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


# PERFORMANCE_THRESHOLD = 0.75 #(80%)
# PERFORMANCE_COLUMN    ='balanced_accuracy'

def get_grid_search_performances(grid, classifier, get_n_best_performances=5):

    df_results = pd.DataFrame(grid.cv_results_)

    df_results['classifier'] = get_classifier_class_name(classifier)

    # reduce the name of the columns by removing the initial string "mean_test_"
    for col in df_results.columns:
        if col.startswith('mean_test_'):
            col_new = col.replace('mean_test_', '')
            df_results.rename(
                columns={col: col_new}, 
                inplace=True,
            )

    df_results.rename(columns={'mean_score_time': 'fit_time'}, inplace=True)

    # get only the columns of interest        
    cols_of_interest = [
        'classifier',
        'balanced_accuracy',
        'sensitivity',
        'specificity',
        'auc',
        'accuracy',
        'precision',
        'f1',
        'params',
        'fit_time'
    ]
    df_results = df_results[cols_of_interest]

    # rank the results by 'balanced_accuracy', 'sensitivity', 'specificity'
    df_results = sort_performances_results(
        df=df_results,
    )

    # get only the "n" best performances (default=5)
    df_results = df_results.head(get_n_best_performances)

    # round the values using 2 decimal places
    df_results = df_results.round(2)        
    
    return df_results


def grid_search_refit_strategy(cv_results):
    """Define the strategy to select the best estimator.

    The strategy defined here is to filter-out all results below a precision threshold
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
    # balanced_accuracy_threshold = 0.73

    df_cv_results = pd.DataFrame(cv_results)
    # # print("All grid-search results:")
    # # print_dataframe(df_cv_results)

    # # Filter-out all results below the threshold
    # high_bal_acc_cv_results = df_cv_results[
    #     df_cv_results[f'mean_test_balanced_accuracy'] >= balanced_accuracy_threshold
    # ]

    # print(f"Models with a precision higher than {balanced_accuracy_threshold}:")
    # print_dataframe(high_precision_cv_results)
    # high_bal_acc_cv_results = df_cv_results

    df_cv_results = df_cv_results[
        [
            "mean_score_time",
            #
            "mean_test_balanced_accuracy",
            "std_test_balanced_accuracy",
            "rank_test_balanced_accuracy",
            #
            "mean_test_sensitivity",
            "std_test_sensitivity",
            "rank_test_sensitivity",
            #
            "mean_test_specificity",
            "std_test_specificity",
            "rank_test_specificity",
            #
            "params",
        ]
    ]

    # Select the most performant models in terms of balanced accuracy
    # (within 1 sigma from the best)
    best_bal_acc_std = df_cv_results["mean_test_balanced_accuracy"].std()
    best_bal_acc = df_cv_results["mean_test_balanced_accuracy"].max()
    best_bal_acc_threshold = best_bal_acc - best_bal_acc_std

    high_bal_acc_cv_results = df_cv_results[
        df_cv_results["mean_test_balanced_accuracy"] > best_bal_acc_threshold
    ]

    # Select the most performant models in terms of sensitivity
    # (within 1 sigma from the best)
    best_sensitivity_std = high_bal_acc_cv_results["mean_test_sensitivity"].std()
    best_sensitivity = high_bal_acc_cv_results["mean_test_sensitivity"].max()
    best_sensitivity_threshold = best_sensitivity - best_sensitivity_std

    high_sensitivity_cv_results = high_bal_acc_cv_results[
        high_bal_acc_cv_results["mean_test_sensitivity"] > best_sensitivity_threshold
    ]
    # print(
    #     "Out of the previously selected high precision models, we keep all the\n"
    #     "the models within one standard deviation of the highest recall model:"
    # )
    # print('ASSSSS')
    # print(high_sensitivity_cv_results)
    # print()

    # From the best candidates, select the fastest model to predict
    fastest_top_sensitivity_high_precision_index = high_sensitivity_cv_results[
        "mean_score_time"
    ].idxmin()

    # print(
    #     "\nThe selected final model is the fastest to predict out of the previously\n"
    #     "selected subset of best models based on precision and recall.\n"
    #     "Its scoring time is:\n\n"
    #     f"{high_recall_cv_results.loc[fastest_top_sensitivity_high_precision_index]}"
    # )

    print(high_sensitivity_cv_results)

    return fastest_top_sensitivity_high_precision_index


def create_classifier_from_string(classifier_as_str, dict_params_as_str):
    # convert params to dict
    dict_params = dict(dict_params_as_str)

    # create an model instance passing the hyperparameters
    klass = globals()[classifier_as_str]
    clf = klass(**dict_params)

    return clf



def get_performances_from_predictions(y_validation, y_pred, y_pred_proba):
    # calculate the scores using y_pred
    bal_acc = np.round(balanced_accuracy_score(y_validation, y_pred), 2)
    sens    = np.round(recall_score(y_validation, y_pred), 2)
    spec    = np.round(recall_score(y_validation, y_pred, pos_label=0), 2)
    f1      = np.round(f1_score(y_validation, y_pred), 2)
    acc     = np.round(accuracy_score(y_validation, y_pred), 2)
    precision    = np.round(precision_score(y_validation, y_pred), 2)

    # calculate AUC using y_pred_proba
    auc     = np.round(roc_auc_score(y_validation, y_pred_proba), 2)

    return bal_acc, sens, spec, auc, acc, precision, f1



DEFAULT_SCORE = 'balanced_accuracy'

def exec_grid_search(classifier, param_grid, X_train, y_train, 
                     X_valid, y_valid, 
                     cv=None, 
                     n_jobs=N_JOBS, verbose=1, scoring=None, 
                     refit=None, return_train_score=False,
                     get_n_best_performances=5,
                    #  sort_results=True, 
                    #  dataset_info='', 
                    #  features_info='', 
                    #  plot_roc_curve=False,
                     ):


    # get only array of output y_train, if it was a dataFrame
    if type(y_train) is pd.DataFrame:
        y_train = y_train[utils.CLASS_COLUMN].ravel()  

    # # get only array of output y_valid, if it was a dataFrame
    # if type(y_valid) is pd.DataFrame:
    #     y_valid = y_valid[utils.CLASS_COLUMN].ravel()  

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
    grid.fit(X_train, y_train)    


    # get performance for each set of hyperparams
    # filtering by the "n" best performances (default: 5)
    df_performances = get_grid_search_performances(
        classifier=classifier,
        grid=grid,
        get_n_best_performances=get_n_best_performances,
    )


    best_models_performances = []
    det_curve_data = []
    precision_recall_curve_data = []
    roc_curve_data = []
    predictions_data = []

    # make predictions using the best classifiers
    for idx, row in df_performances.iterrows():
        clf = create_classifier_from_string(
            classifier_as_str=row.classifier,
            dict_params_as_str=row.params,
        )

        # fit the classifier again using training data
        clf.fit(X_train, y_train)

        # make predictions
        y_pred = clf.predict(X_valid)

        # make predictions using probabilities, returning the class 
        # probabilities for sample.
        # The first column represents the probability of the 
        # negative class (Non-Short) and the second column represents 
        # the probability of the positive class (Short).
        y_pred_proba = clf.predict_proba(X_valid)
        y_pred_proba = y_pred_proba[:,1] # get short-survival probabilities

        bal_acc, sens, spec, auc, acc, prec, f1 = get_performances_from_predictions(
            y_validation=y_valid,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
        )

        best_models_performances.append({
            'Model': utils.get_model_description(row.classifier),
            'balanced_accuracy': bal_acc,
            'sensitivity': sens,
            'specificity': spec,
            'f1_score': f1,
            'AUC': auc,
            'accuracy': acc,
            'precision': prec,
            'Model_Class': row.classifier,
            'Hyperparams': row.params,
            'fit_time': row.fit_time,
        })

    # create a dataFrame with the best performances
    df_best_performances = pd.DataFrame(best_models_performances)

    # sort the performances and get the first row (best model)
    df_best_performances = sort_performances_results(
        df=df_best_performances,
    )

    df_best_performances_detailed = df_best_performances.copy()

    df_best_performances = df_best_performances.head(1)

    # collect additional info about the best model
    for idx, row in df_best_performances.iterrows():

        # =======================================================
        # Detection Error Tradeoff (DET) curve
        # =======================================================
        fpr, fnr, thresholds = sk.metrics.det_curve(y_valid, y_pred_proba)
        det_curve_aux = {
            'Model'      : row.Model, 
            'Hyperparams': row.Hyperparams, 
            'FPR'        : fpr, 
            'FNR'        : fnr, 
            'Thresholds' : thresholds,
        }
        det_curve_data.append(det_curve_aux)

        # =======================================================
        # Precision-Recall curve
        # =======================================================
        precision, recall, thresholds = sk.metrics.precision_recall_curve(y_valid, y_pred_proba)
        au_prec_recall_curve = sk.metrics.auc(recall, precision)
        precision_recall_curve_aux = {
            'Model'      : row.Model, 
            'Hyperparams': row.Hyperparams, 
            'Precision'  : precision,
            'Recall'     : recall, 
            'Thresholds' : thresholds,
        }
        precision_recall_curve_data.append(precision_recall_curve_aux)

        # =======================================================
        # ROC curve
        # =======================================================
        fpr, tpr, thresholds = sk.metrics.roc_curve(y_valid, y_pred_proba)
        roc_auc = sk.metrics.auc(fpr, tpr)
        roc_curve_aux = {
            'Model'      : row.Model, 
            'Hyperparams': row.Hyperparams, 
            'FPR'        : fpr, 
            'TPR'        : tpr, 
            'Thresholds' : thresholds,
        }
        roc_curve_data.append(roc_curve_aux)

        # =======================================================
        # Predictions data (using predict and predict_proba)
        # =======================================================
        predictions_aux = {
            'Model'       : row.Model, 
            'Hyperparams' : row.Hyperparams, 
            'y_pred'      : y_pred, 
            'y_pred_proba': y_pred_proba
        }
        predictions_data.append(predictions_data)



    # create a dict representing additional info (DET, Prec-Recall-curve, ROC, and predictions)
    additional_info = {
        'DET': det_curve_data,
        'Precision-Recall-Curve': precision_recall_curve_data,
        'ROC': roc_curve_data,
        'Predictions': predictions_data,
    }



    return grid, df_best_performances, df_best_performances_detailed, additional_info


    # filter


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

