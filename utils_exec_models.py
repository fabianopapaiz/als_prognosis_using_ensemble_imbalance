
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

import utils
import pandas as pd
import numpy as np

# CONSTANT to store the random_state stated for reproducibility issues
RANDOM_STATE = 42

N_JOBS = 3

CV_N_SPLITS = 5


def sort_performances_results(df, cols_order_to_sort=['BalAcc', 'Sens', 'Spec'],
                      cols_to_return=None):

    df_bests = df.sort_values(cols_order_to_sort, ascending=False).copy()

    if cols_to_return is not None:
        return df_bests[cols_to_return]
    else:
        return df_bests
    


## get all performances for the models in the GridSearch
def get_grid_search_performances(grid_search, 
            dataset_info, features_info,
            sort_results=True):
    
    classifiers = grid_search.cv_results_['params']

    # get the models and hyperparameters
    models = []
    hyperparams = []
    for classif_dict in classifiers:
        dict_params = {}
        for key, value in classif_dict.items():
            if key == 'classifier':
                clf = value    
            else:
                # correct the param name
                new_key = key.replace('classifier__', '')
                dict_params[new_key] = value    

        model = clf.__class__.__name__
        params = str(dict_params)

        models.append(model)
        hyperparams.append(params)

    # get the performances
    dict_results = {}
    for key, value in grid_search.cv_results_.items():
        # get mean_test performances
        if key.startswith('mean_test_'):
            new_key = key.replace('mean_test_', '')
            dict_results[new_key] = list(value)
        
    bal_accs = np.round(dict_results['balanced_accuracy'], 2)
    senss    = np.round(dict_results['sensitivity'], 2)
    specs    = np.round(dict_results['specificity'], 2)
    f1s      = np.round(dict_results['f1'], 2)
    aucs     = np.round(dict_results['AUC'], 2)
    accs     = np.round(dict_results['accuracy'], 2)
    precs    = np.round(dict_results['precision'], 2)

    # create a dict containg all models, params, and performances 
    models_results = [] 
    for classifier, hyperparams, bal_acc, sens, spec, f1, auc, acc, prec in zip(models, hyperparams, bal_accs, senss, specs, f1s, aucs, accs, precs):

        model_desc = get_model_description(classifier)

        models_results.append({
            'Dataset': dataset_info,
            'Features': features_info,
            'Model': model_desc,
            'BalAcc': bal_acc,
            'Sens': sens,
            'Spec': spec,
            'f1': f1,
            'AUC': auc,
            'Acc': acc,
            'Prec': prec,
            'Classifier': classifier,
            'Hyperparams': str(hyperparams),
        })
    
    # create a dataFrame containg the results
    df_results = pd.DataFrame(models_results)

    if sort_results:
        df_results = sort_performances_results(df=df_results)

    return df_results



DEFAULT_SCORE = 'balanced_accuracy'
def get_default_scoring():
    # metrics to evaluate the model performance
    scoring = {
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'sensitivity': make_scorer(recall_score),
        'specificity': make_scorer(recall_score, pos_label=0),
        'f1': make_scorer(f1_score),
        'AUC': 'roc_auc', #make_scorer(roc_auc_score, multi_class='ovr'),
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, zero_division=0),
    }
    return scoring


def exec_grid_search(param_grid, X, y, cv=None, 
                     n_jobs=N_JOBS, verbose=1, scoring=None, 
                     refit=None, return_train_score=False,
                     sort_results=True, dataset_info='', 
                     features_info=''):

    pipeline = Pipeline(steps=[('classifier', GaussianNB() )])

    if type(y) is pd.DataFrame:
        y = y[utils.CLASS_COLUMN].ravel()  


    if scoring is None:
        scoring = get_default_scoring()

    if refit is None:
        refit = DEFAULT_SCORE

    if cv is None:
        cv = get_kfold_splits()

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid, 
        scoring=scoring,
        cv=cv, 
        verbose=verbose,
        n_jobs=n_jobs,
        return_train_score=return_train_score,
        refit=refit,
    )

    grid.fit(X, y)    

    df_results = get_grid_search_performances(
        grid_search=grid,
        dataset_info=dataset_info,
        features_info=features_info,
        sort_results=sort_results,
    )

    return grid, df_results



def create_models_NB_grid(param_grid=None, testing=False):
    # hyperparams
    alphas = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    norms = [False, True]

    if testing:
        alphas = [0.1, 0.5]
        norms = [False]


    if param_grid is None:
        param_grid = []

    param_grid.append(
        {
            "classifier__alpha": alphas,
            "classifier__norm": norms,
            "classifier": [ComplementNB()]
        }
    )    

    param_grid.append(
        {
            "classifier": [GaussianNB()]
        }
    )    

    return param_grid



def create_models_DT_grid(param_grid=None, testing=False):
    # hyperparams
    max_depths = [3, 4, 5, 7, 9, 10, 15, 25, 50]
    criterions = ['gini', 'entropy'] #, 'log_loss'] LOG-LOSS DOESN'T WORK
    class_weights = [None, 'balanced']


    if testing:
        max_depths = [5]
        criterions = ['gini']
        class_weights = ['balanced']


    if param_grid is None:
        param_grid = []

    param_grid.append(
        {
            "classifier__max_depth": max_depths,
            "classifier__criterion": criterions,
            "classifier__class_weight": class_weights,
            "classifier__random_state": [RANDOM_STATE],
            "classifier": [DecisionTreeClassifier()]
        }
    )    

    return param_grid



def create_models_RF_grid(param_grid=None, testing=False):
    # hyperparams
    max_depths = [10, 15, 25, 50]
    num_estimators = [50, 75, 100, 200] 
    criterions = ['gini', 'entropy'] 
    class_weights = [None, 'balanced', 'balanced_subsample']


    if testing:
        max_depths = [5]
        num_estimators = [50] 
        criterions = ['gini']
        class_weights = ['balanced']


    if param_grid is None:
        param_grid = []

    param_grid.append(
        {
            "classifier__max_depth": max_depths,
            "classifier__n_estimators": num_estimators,
            "classifier__criterion": criterions,
            "classifier__class_weight": class_weights,
            "classifier__random_state": [RANDOM_STATE],
            "classifier": [RandomForestClassifier()]
        }
    )    

    return param_grid



def create_models_NN_grid(qty_features, param_grid=None, testing=False):
    # hyperparams
    max_iter = [1000]
    layers = [
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
    learning_rate_init = [0.7]


    if testing:
        max_iter = [300]
        layers = [(qty_features)]
        alphas = [0.1]
        activations = ['relu']
        solvers = ['sgd']
        learning_rates = ['constant']


    if param_grid is None:
        param_grid = []

    param_grid.append(
        {
            "classifier__max_iter": max_iter,
            "classifier__hidden_layer_sizes": layers,
            "classifier__alpha": alphas,
            "classifier__activation": activations,
            "classifier__solver": solvers,
            "classifier__learning_rate": learning_rates,
            "classifier__learning_rate_init": learning_rate_init,
            "classifier__random_state": [RANDOM_STATE],
            "classifier": [MLPClassifier()]
        }
    )    

    return param_grid



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
    #radius
    radius_set = [0.3, 0.5, 0.7, 1.0] 
    leaf_sizes = [50, 100, 200] 
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

    # k-NN
    param_grid.append(
        {
            "classifier__n_neighbors": ks,
            "classifier__weights": weights,
            "classifier__metric": distance_metrics,
            "classifier": [KNeighborsClassifier()]
        }
    )    

    # Radius-NN
    param_grid.append(
        {
            "classifier__radius": radius_set,
            "classifier__weights": weights,
            "classifier__leaf_size": leaf_sizes,
            "classifier__outlier_label": outlier_labels,
            "classifier__metric": distance_metrics,
            "classifier": [RadiusNeighborsClassifier()]
        }
    )    


    return param_grid


def create_models_SVM_grid(param_grid=None, testing=False):
    # hyperparams
    kernels = ['rbf', 'linear'] #, 'poly', 'sigmoid',]
    gammas = ['scale', 'auto',]
    class_weights = [None, 'balanced',]
    Cs = [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 10, 100, 200, 1000, 1500, 1700, 2000]

    if testing:
        kernels = ['rbf', 'linear'] #, 'poly', 'sigmoid',]
        gammas = ['auto',]
        class_weights = [None]
        Cs = [0.1, 0.3, ]


    if param_grid is None:
        param_grid = []

    param_grid.append(
        {
            "classifier__C": Cs,
            "classifier__kernel": kernels,
            "classifier__gamma": gammas,
            "classifier__class_weight": class_weights,
            "classifier__probability": [True],
            "classifier__random_state": [RANDOM_STATE],
            "classifier": [svm.SVC()]
        }
    )    

    return param_grid


def get_kfold_splits(n_splits=CV_N_SPLITS, random_state=RANDOM_STATE, shuffle_kfold=True, ):
    kfold = StratifiedKFold(
        n_splits=n_splits, 
        random_state = random_state if shuffle_kfold else None, 
        shuffle=shuffle_kfold,
    )
    return kfold


def split_training_testing_validation(df, test_size=0.2, random_state=RANDOM_STATE, stratify=None):

    # separate into input and output variables
    input_vars = df.copy()
    input_vars.drop(columns=[utils.CLASS_COLUMN])

    output_var = df[[utils.CLASS_COLUMN]].copy()

    # if informed in format of 20/30 instead of 0.2/0.3
    if test_size > 1.0:
        test_size = test_size/100

    # split data
    X_train, X_valid, y_train, y_valid = train_test_split(
        input_vars,
        output_var,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # join the X and y for each subset 
    df_train = X_train.copy()
    df_train[utils.CLASS_COLUMN] = y_train[utils.CLASS_COLUMN]

    df_valid = X_valid.copy()
    df_valid[utils.CLASS_COLUMN] = y_valid[utils.CLASS_COLUMN]

    #
    return df_train, df_valid




# create a data-frame joining the 2 groups, labelling groups (Group-1 and Group-2)
def create_data_frame_from_two_groups(series_1, series_2, title_group_1='Group-1', title_group_2='Group-2'):
    col_group_name = 'Group'
    #
    df_1 = pd.DataFrame(series_1)
    df_1[col_group_name] = title_group_1
    #
    df_2 = pd.DataFrame(series_2)
    df_2[col_group_name] = title_group_2
    #
    # df = df_1.append(df_2)
    df = pd.concat([df_1, df_2])
    #
    return df, col_group_name



def get_model_description(model_desc):
    
    NB_models = [
        'ComplementNB', 
        'GaussianNB', 
        'CategoricalNB',
        'NB',

    ]

    KNN_models = [
        'RadiusNeighborsClassifier', 
        'KNeighborsClassifier',
        'k-NN', 
    ]

    NN_models = [
        'MLPClassifier',
        'NN',
    ]

    RF_models = [
        'RandomForestClassifier',
        'RF', 
    ]

    DT_models = [
        'DecisionTreeClassifier',
        'DT', 
    ]

    XGB_models = [
        'XGBClassifier',
        # 'XGBoost',
    ]

    CatBoost_models = [
        'CatBoostClassifier',
        # 'CatBoost',
    ]

    SVM_models = [
        'SVC', 
    ]

    BalancedBagging_models = [
        'BalancedBaggingClassifier',
        'Bal. Bagging'
    ]

    if model_desc in NB_models:
        return 'Naïve Bayes'
    elif model_desc in KNN_models:
        return 'k-NN'
    elif model_desc in NN_models:
        return 'Neural Networks'
    elif model_desc in RF_models:
        return 'Random Forest'
    elif model_desc in DT_models:
        return 'Decision Tree'
    elif model_desc in SVM_models:
        return 'SVM'
    elif model_desc in XGB_models:
        return 'XGBoost'
    elif model_desc in CatBoost_models:
        return 'CatBoost'
    elif model_desc in BalancedBagging_models:
        return 'Balanced Bagging'
    else:
        return model_desc



def get_model_short_description(model_desc):
    
    NB_models = [
        'ComplementNB', 
        'GaussianNB', 
        'CategoricalNB',
        'Naïve Bayes',
    ]

    NN_models = [
        'MLPClassifier',
        'Neural Networks',
    ]

    KNN_models = [
        'RadiusNeighborsClassifier', 
        'KNeighborsClassifier',
        'k-NN', 
    ]


    RF_models = [
        'RandomForestClassifier',
        'Random Forest', 
    ]

    DT_models = [
        'DecisionTreeClassifier',
        'Decision Tree', 
    ]

    BalancedBagging_models = [
        'BalancedBaggingClassifier',
    ]

    if model_desc in NB_models:
        return 'NB'
    elif model_desc in KNN_models:
        return 'k-NN'
    elif model_desc in NN_models:
        return 'NN'
    elif model_desc in RF_models:
        return 'RF'
    elif model_desc in DT_models:
        return 'DT'
    elif model_desc in BalancedBagging_models:
        return 'Bal. Bagging'
    else:
        return model_desc



def sort_performances_results(df, cols_order_to_sort=['BalAcc', 'Sens', 'Spec'], cols_to_return=None):

    df_bests = df.sort_values(cols_order_to_sort, ascending=False).copy()

    if cols_to_return is not None:
        return df_bests[cols_to_return]
    else:
        return df_bests