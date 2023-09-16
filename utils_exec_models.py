
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
def get_grid_search_performances(grid_search=None, performances=None, 
            dataset_info='', features_info='',
            sort_results=True):

    models_results = [] 
    
    if grid_search is not None:
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
        for classifier, hyperparam, bal_acc, sens, spec, f1, auc, acc, prec in zip(models, hyperparams, bal_accs, senss, specs, f1s, aucs, accs, precs):

            model_desc = get_model_description(classifier)

            hyperparam = str(hyperparam).replace('\n', '').replace('             ','')

            # special config for the Balanced Bagging Classifier 
            if classifier == 'BalancedBaggingClassifier':
                hyperparam = "{'n_estimators':" + hyperparam.split(", 'n_estimators':")[1]

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
                'Hyperparams': hyperparam,
            })
        
    elif performances is not None:
        model_desc, classifier, hyperparam, bal_acc, sens, spec, auc, acc, prec, f1 = performances
            
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
            'Hyperparams': hyperparam,
        })



    # create a dataFrame containg the results
    df_results = pd.DataFrame(models_results)

    if sort_results:
        df_results = sort_performances_results(df=df_results)

    return df_results



def exec_grid_search(param_grid, X, y, cv=None, 
                     n_jobs=N_JOBS, verbose=1, scoring=None, 
                     refit=None, return_train_score=False,
                     sort_results=True, dataset_info='', 
                     features_info='', 
                     X_valid=None, y_valid=None, plot_roc_curve=False):

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


    if plot_roc_curve:
        clf = grid.best_estimator_  
        # print(clf)

        clf.fit(X, y)

        str_replace = "Pipeline(steps=[('classifier',"
        clf_instance = str(clf).replace(str_replace, '').replace('\n', '').replace(')])','').replace(' ','').strip()
        # print(clf_instance)
        estimator_name = clf_instance.split('(')[0]
        hyperparams = clf_instance.split('(')[1][:-1]

        # the first column represents the probability of the negative class (class 0) and the second column represents the probability of the positive class (class 1).
        y_pred_proba = clf.predict_proba(X_valid)
        y_pred_proba = y_pred_proba[:,1] #short-survival



        precision, recall, _ = sk.metrics.precision_recall_curve(y_valid, y_pred_proba)
        print(f'AUC Precision-Recall: {sk.metrics.auc(recall, precision)}')
        # optimal_proba_cutoff = list(zip(precision, recall))
        # print('-2', optimal_proba_cutoff)

        # optimal_proba_cutoff = np.abs(precision - recall)
        # print('-1', optimal_proba_cutoff)

        # optimal_proba_cutoff = list(zip(np.abs(precision - recall), y_pred_proba))
        # print('0', optimal_proba_cutoff)

        # optimal_proba_cutoff = sorted(list(zip(np.abs(precision - recall), y_pred_proba)), key=lambda i: i[0], reverse=False)
        # print('1', optimal_proba_cutoff)
        # print('2', optimal_proba_cutoff[0])
        
        prc_display = sk.metrics.PrecisionRecallDisplay.from_predictions(
            y_valid, 
            y_pred_proba,
            name=estimator_name,
            plot_chance_level=True,
            # drop_intermediate=True,
        )

        gmeans = np.sqrt(recall * precision)
        ix = np.argmax(gmeans)
        plt.plot(recall[ix], precision[ix], 'o', ms=5, c='red')
        print('gmeans', recall[ix], precision[ix])

        euclidean_dist = np.sqrt(np.power((1-recall),2) + np.power((1-precision), 2))
        euclidean_dist = sk.metrics.pairwise.euclidean_distances(np.column_stack((recall,precision)), [[1., 1.]])
        ix = np.argmin(euclidean_dist)
        plt.plot(recall[ix], precision[ix], 'o', ms=5, c='blue')
        print('euclid', recall[ix], precision[ix])


        # print(optimal_proba_cutoff[1], optimal_proba_cutoff[0])
        # plt.plot(optimal_proba_cutoff[0], optimal_proba_cutoff[1], 'ko', ms=5)
        plt.show()


        fpr, tpr, thresholds = sk.metrics.roc_curve(y_valid, y_pred_proba)
        roc_auc = sk.metrics.auc(fpr, tpr)
        print(f'AUC ROC: {roc_auc}')
        roc_display = sk.metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=estimator_name)
        roc_display.plot()


        gmeans = np.sqrt(tpr * (1-fpr))
        ix = np.argmax(gmeans)
        # print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
        # print(fpr[ix], tpr[ix])
        plt.plot(fpr[ix], tpr[ix], 'ko', ms=5)
        print('gmeans', tpr[ix], fpr[ix])

        euclidean_dist = np.sqrt(np.power((1-tpr),2) + np.power((0-fpr), 2))
        euclidean_dist = sk.metrics.pairwise.euclidean_distances(np.column_stack((tpr,fpr)), [[1., 0.]])
        ix = np.argmin(euclidean_dist)
        plt.plot(fpr[ix], tpr[ix], 'o', ms=5, c='blue')
        print('gmeans', tpr[ix], fpr[ix])

        plt.show()



        y_pred = clf.predict(X_valid)

        # # confusion matrix
        # cm = confusion_matrix(y_valid, y_pred, labels=clf.classes_)
        # disp = sk.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        # disp.plot()
        # plt.show()

        bal_acc, sens, spec, auc, acc, precision, f1 = get_scores_from_predict(
            y_validation=y_valid,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            print_info=False,
        )

        model_desc = get_model_description(estimator_name)

        df_results = get_grid_search_performances(
            # grid_search=grid,

            performances=[model_desc, estimator_name, hyperparams, bal_acc, sens, spec, auc, acc, precision, f1],
            dataset_info=dataset_info,
            features_info=features_info,
            sort_results=sort_results,
        )
        # # 'balanced_accuracy': make_scorer(balanced_accuracy_score),
        # print("BallAcc: {}".format(balanced_accuracy_score(y_valid, y_pred)))
        # print("Sens: {}".format(recall_score(y_valid, y_pred)))
        # print("Spec: {}".format(recall_score(y_valid, y_pred, pos_label=0)))
        # print("f1: {}".format(f1_score(y_valid, y_pred)))
        # print("Acc: {}".format(accuracy_score(y_valid, y_pred)))
        # print("Prec: {}".format(precision_score(y_valid, y_pred)))
        # print("AUC: {}".format(roc_auc_score(y_valid, y_pred_proba)))


    return grid, df_results


DEFAULT_SCORE = 'balanced_accuracy'
def get_default_scoring():
    # metrics to evaluate the model performance
    scoring = {
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'sensitivity': make_scorer(recall_score),
        'specificity': make_scorer(recall_score, pos_label=0),
        'f1': make_scorer(f1_score),
        #
        'AUC': 'roc_auc', #make_scorer(roc_auc_score, multi_class='ovr'),
        'accuracy': make_scorer(roc_auc_score),

        #
        # 'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, zero_division=0),
    }
    return scoring


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
    max_depths = [3, 4, 5, 7, 9, 10, 15, 25] #, 50]
    criterions = ['gini', 'entropy'] #, 'log_loss'] LOG-LOSS DOESN'T WORK
    class_weights = [None, 'balanced']
    class_weights = ['balanced']


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
    max_depths = [5, 7, 10, 15] #, 25, 50]
    num_estimators = [11, 15, 21, 51] #, 75, 100, 200] 
    criterions = ['gini', 'entropy'] 
    # class_weights = [None, 'balanced', 'balanced_subsample']
    class_weights = ['balanced']


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

    SVM_models = [
        'SVC', 
    ]

    BalancedBagging_models = [
        'BalancedBaggingClassifier',
    ]

    if model_desc in NB_models:
        return 'NB'
    elif model_desc in SVM_models:
        return 'SVM'
    elif model_desc in KNN_models:
        return 'k-NN'
    elif model_desc in NN_models:
        return 'NN'
    elif model_desc in RF_models:
        return 'RF'
    elif model_desc in DT_models:
        return 'DT'
    elif model_desc in BalancedBagging_models:
        return 'Balanced-Bagging'
    else:
        return model_desc



def sort_performances_results(df, cols_order_to_sort=['BalAcc', 'Sens', 'Spec'], cols_to_return=None):

    df_bests = df.sort_values(cols_order_to_sort, ascending=False).copy()

    if cols_to_return is not None:
        return df_bests[cols_to_return]
    else:
        return df_bests
    

 
# get a Set of models from the results CSV informed, without repeating
def get_models_set_from_results(results_csv_file):

    df_classifiers = utils.read_csv(results_csv_file)

    cols = ['Classifier','Hyperparams']
    df_classifiers = df_classifiers[cols]

    df_classifiers = df_classifiers.groupby(by=cols).first().reset_index()
    df_classifiers = df_classifiers.sort_values(by=cols)

    classifiers = []

    # create model instance from hyperparameters
    for idx, row in df_classifiers.iterrows():
        # params_dict = ast.literal_eval(row.Hyperparams)
        # klass = globals()[row.Classifier]
        # clf = klass(**params_dict)    

        m = row.Classifier
        h = row.Hyperparams
        clf = create_model_from_string(
                model=m,
                hyperparams=h,
            )

        classifiers.append(clf)

    #
    return classifiers   




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
            "classifier__estimator": classifiers,
            "classifier__n_estimators": num_estimators,
            "classifier__sampling_strategy": sampling_strategies,
            "classifier__warm_start": warm_starts,
            "classifier__random_state": [RANDOM_STATE],
            "classifier": [resemb.BalancedBaggingClassifier()]
        }
    )    

    return param_grid



# convert hyperparams to dict
# Example: 
#        x = "(alpha=0.05, hidden_layer_sizes=(14,), learning_rate_init=0.7,\n max_iter=1000, random_state=42, solver='sgd')"
#   return = {'alpha': 0.05, 'hidden_layer_sizes': (14,), 'learning_rate_init': 0.7, 'max_iter': 1000, 'random_state': 42, 'solver': 'sgd'}
def convert_hyperparams_to_dict(x):
    
    if type(x) is not str:
        # print(type(x))
        x = str(x)

    if (x.strip() == ''):
        # print(x.strip() == '', f':{x}:', type(x))
        return x


    x = x.replace('{', '').replace('}', '')
    
    if (x.strip() == ''):
        # print(x.strip() == '', f':{x}:', type(x))
        return x

    # copy x
    string = x
    
    # remove information about covariance matrix present in 'metric_params', 
    # to not break this function
    if '{\'VI\':' in string:
        aux = string.split("'metric_params':")
        before = aux[0]
        after = aux[1].split("])},")[1]
        string = before + after


    string.replace("\n", "")

    # Replace ' with " 
    # After, replace "None" with ""
    string = string.replace('\'', '\"').replace(' None', ' ""')

    # Replace ; with , 
    # After, replace "None" with ""
    string = string.replace(';', ',')

    # Replace False/True with "False"/"True" 
    string = string.replace(' False', ' "False"').replace(' True', ' "True"')

    aux = string.split(', "')
    # print(aux)
    params = ''
    for s_aux in aux:
        try:
            s = s_aux.split(':')
            # print(s[0][0])
            if s[0][0] == '\"':
                param = f'{s[0].strip()}'
            else:    
                param = f'"{s[0].strip()}'
            value = s[1]
            # if '\"' not in value:
            #     value = f'"{value.strip()}"' 
            params += f'{param}: {value},'
            # print(params)
        except Exception as ex:
            try:
                s = s_aux.split('=')
                if s[0][0] == '\"':
                    param = f'{s[0].strip()}'
                else:    
                    param = f'"{s[0].strip()}'
                # param = f'"{s[0].strip()}"'
                value = s[1]
                # if '\"' not in value:
                #     value = f'"{value.strip()}"' 
                params += f'{param}: {value},'
            except Exception as ex:
                print('<<ERROR>>')
                print(x, string)
                print(f' - s = {s}')
                print(f' - param = {param}')
                print(aux)
                print(s)
                raise Exception(f'ERROR: {ex}')    
        
    # print('aaaa')
    # print(params)
    params = '{' + params[:-1] + '}'
    # print(params)
    

    # init return variable
    ret = "<ERRO>"
    try:
        # convert string data to dict
        # ret = ast.literal_eval(params)
        # ret = json.loads(params)
        ret = eval(params)
    except Exception as ex:
        print(params)
        raise Exception(f'ERROR: {ex}')    

    #    
    return ret



def get_models_object_from_results(df_results, add_params_info=True):

    models = []

    ens_imb_scenario = ('Balanced Bagging' in df_results.Model.unique())
    scenario = ('Ensemble-Imbalance' if ens_imb_scenario else 'Single-Model')   

    # print(ens_imb_scenario)

    for index, row in df_results.iterrows():

        classifier = row.Classifier
        hyperparams = row.Hyperparams
        if ens_imb_scenario:
            estimator = row.Estimator_Class
            estimator_hyperparams = row.Estimator_Hyperparams
        else:
            estimator = None
            estimator_hyperparams = None

        model = create_model_from_string(
            model=classifier,
            hyperparams=hyperparams,
            estimator_model=estimator,
            estimator_hyperparams=estimator_hyperparams
        )


        # add model info to as a dict
        models.append(
            {
            'model_instance': model,
            'Scenario': scenario,
            'Model': get_model_short_description(classifier),
            'Hyperparams': hyperparams,
            'Estimator': get_model_short_description(estimator),
            'Estimator_Hyperparams': estimator_hyperparams,  
            }
        )
    
    #
    return models




def create_model_from_string(model, hyperparams, estimator_model=None, estimator_hyperparams=None):

    if str(hyperparams).strip() == '':
        hyp = dict()
    else:    
        hyp = str(hyperparams).replace(": ''", ": None")
        hyp = hyp.replace(": 'True'", ": True")
        hyp = hyp.replace(": 'False'", ": False")
        try:
            hyp = eval(hyp)
        except Exception as ex:   
            print(hyp)
            # print(f'ERROR: {ex}')
            print()
            raise Exception(ex)


    # create an model instance passing the hyperparameters
    klass = globals()[model]
    classifier = klass(**hyp)

    if estimator_model is not None:
        # get instance of the estimator for use in the main classifier (e.g., BalanceBagging)
        estimator = create_model_from_string(
            model=estimator_model,
            hyperparams=estimator_hyperparams,
        )
        # set the estimator in the main classifier
        classifier.estimator = estimator

    #
    return classifier



def get_scores_from_predict(y_validation, y_pred=None, y_pred_proba=None, fitted_model=None, X_validation=None, print_info=False):
    # call predict method
    if fitted_model is not None:
        y_pred = fitted_model.predict(X_validation)
    # calculate the scores
    bal_acc = np.round(balanced_accuracy_score(y_validation, y_pred), 2)
    sens    = np.round(recall_score(y_validation, y_pred), 2)
    spec    = np.round(recall_score(y_validation, y_pred, pos_label=0), 2)
    f1      = np.round(f1_score(y_validation, y_pred), 2)
    acc     = np.round(accuracy_score(y_validation, y_pred), 2)
    precision    = np.round(precision_score(y_validation, y_pred), 2)

    if y_pred_proba is None:
        auc     = np.round(roc_auc_score(y_validation, y_pred, multi_class='ovr'), 2)
    else:
        auc     = np.round(roc_auc_score(y_validation, y_pred_proba), 2)

    if print_info:
        print(f'BalAcc: {bal_acc:>4.2f}      f1  : {f1:.2f}')
        print(f'Sens  : {sens:>4.2f}      Acc : {acc:.2f}')
        print(f'Spec  : {spec:>4.2f}      Prec: {precision:.2f}')
        print(f'                  AUC : {auc:.2f}')

    #
    if fitted_model is not None:
        return bal_acc, sens, spec, auc, acc, precision, f1, y_pred
    else:
        return bal_acc, sens, spec, auc, acc, precision, f1


def get_scores_from_predict_proba(y_validation, y_pred=None, fitted_model=None, X_validation=None, print_info=False):
    # call predict method
    if fitted_model is not None:
        y_pred = fitted_model.predict_proba(X_validation)
    # calculate the scores
    # bal_acc = np.round(balanced_accuracy_score(y_validation, y_pred), 2)
    # sens    = np.round(recall_score(y_validation, y_pred), 2)
    # spec    = np.round(recall_score(y_validation, y_pred, pos_label=0), 2)
    # f1      = np.round(f1_score(y_validation, y_pred), 2)
    # acc     = np.round(accuracy_score(y_validation, y_pred), 2)
    # precision    = np.round(precision_score(y_validation, y_pred), 2)
    auc     = np.round(roc_auc_score(y_validation, y_pred, multi_class='ovr'), 2)

    if print_info:
        # print(f'BalAcc: {bal_acc:>4.2f}      f1  : {f1:.2f}')
        # print(f'Sens  : {sens:>4.2f}      Acc : {acc:.2f}')
        # print(f'Spec  : {spec:>4.2f}      Prec: {precision:.2f}')
        print(f'                  AUC : {auc:.2f}')

    return auc

    # #
    # if fitted_model is not None:
    #     return bal_acc, sens, spec, auc, acc, precision, f1, y_pred
    # else:
    #     return bal_acc, sens, spec, auc, acc, precision, f1



def sort_performances_results(df, cols_order_to_sort=['BalAcc', 'Sens', 'Spec'],
                      cols_to_return=None):

    df_bests = df.sort_values(cols_order_to_sort, ascending=False).copy()

    if cols_to_return is not None:
        return df_bests[cols_to_return]
    else:
        return df_bests


def plot_barplot_with_performances_by_model_and_scenario(df, hatched_bars=False, 
            annotate=False, figsize=[20,20], sort_columns=None, graph_title=None,
            remove_model_from_y_label=False, col_group_name='Scenario and Model'):

    col_model = 'Model'
    col_scenario = 'Scenario'
    col_estimator = 'Estimator'
    col_bacc = 'Valid_BalAcc'
    col_sens = 'Valid_Sens'
    col_spec = 'Valid_Spec'
#     col_auc  = 'Valid_AUC'
#     col_acc  = 'Valid_Acc'
#     col_prec  = 'Valid_Prec'
#     col_f1   = 'Valid_f1'
    
    if not remove_model_from_y_label:
        col_group = col_group_name
    else:
        col_group = 'Scenario'
    
    df_aux = df.copy()
    
    
    if sort_columns is None:
        sort_columns = [col_bacc, col_sens, col_spec]
    
    df_aux.sort_values(
        by=sort_columns, 
        ascending=False, 
        inplace=True,
    )
   
    
    if remove_model_from_y_label:
        df_aux[col_group] = df_aux[col_model]
    else:
        df_aux[col_group] = df_aux[col_model] + ' (' + df_aux[col_estimator].astype(str) + ')'
    
    # fix the name of the models of the Single-Model Scenario (remove Estimator info)
    df_aux[col_group] = df_aux[col_group].str.replace("\(nan\)", '')
    
    # df_aux.sort_values(by=[col_group], inplace=True)

    
    to_plot  =[
        ['Balanced Acc.', col_bacc],
        ['Sensitivity', col_sens],
        ['Specficity', col_spec],
#         ['AUC', col_auc],
#         ['f1 Score', col_f1],
#         ['Accuracy', col_acc],
#         ['Precision', col_prec],
    ]
    
    concat_dfs = []
    for desc, col in to_plot:
        # separate the each performance metric into dataFrames
        df_temp = df_aux[[col_group, col]].copy()
        df_temp.rename(columns={col: 'Performance'}, inplace=True)
        s_max = f' (max: {df_temp.Performance.max()})' if not annotate else ''
        df_temp['Metric'] = desc
        concat_dfs.append(df_temp)
        
    # concatenate all DFs
    df_graph = pd.concat(concat_dfs)


    # plot the graph
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df_graph, 
        x='Performance', 
        y=col_group, 
        hue='Metric',
        palette= sns.color_palette("colorblind") if not hatched_bars else None,
    )
    
    

    # if was to plot bar with hash instead of colors
    if hatched_bars:
        
#         hatches = [ "+" , "-", ".", "*","x", "o", "O"] #, "|" , "\\" , "/" ,  ]
        hatches = [ "|" , "\\" , "/" , "+" , "-", ".", "*","x", "o", "O" ]
        
        num_bars = 7
        i = 0
        for bar in ax.patches:
            if (i) % num_bars == 0:
                i = 0
            hatch = hatches[i]
            bar.set_hatch(hatch)
            bar.set_color('white')
            bar.set_edgecolor('black')
            i += 1
    
    
    # annotate the bars with their values
    if annotate and not hatched_bars:
        for p in ax.patches:
            ax.annotate(
                "%.2f" % p.get_width(), 
                xy=(
                    p.get_width()+ 0.015, 
                    p.get_y()+p.get_height()/2
                ),
                xytext=(5, 0), 
                textcoords='offset points', 
                ha="left", 
                va="center",
                size=10,
            )    


  
    plt.xlim(0, 1)
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    # red vertical line to highlight the performance threshold
    plt.axvline(x=0.8, color='r', ls=':', label='Performance Threshold')
    
    #place legend outside top right corner of plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    if graph_title is not None:
        plt.title(graph_title)
    
    #
    plt.show()    
    plt.close()

    return df_graph
