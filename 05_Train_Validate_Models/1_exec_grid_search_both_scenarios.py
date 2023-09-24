import sys
import os
import time

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#needed to import utils.py
sys.path.append('../') 

import utils
import utils_preprocessing
import utils_exec_models
import utils_exec_models_new

import numpy as np
import pandas as pd
#to view entire text of the comuns
pd.set_option('display.max_colwidth', None) 

import sklearn as sk

import matplotlib.pyplot as plt
import seaborn as sns


import pickle

from timeit import default_timer as timer



# ======================================================================
# get train/test data and separate by features_config
# ======================================================================

# get train and tezt sets INCLUDING the Diagnosis_Delay feature
X_train_all, y_train_all, X_valid_all, y_valid_all = utils.get_train_and_validation_data(
    scaled=True,
    use_diagnosis_delay=True,
)

# get train and tezt sets REMOVING the Diagnosis_Delay feature
X_train_ndd, y_train_ndd, X_valid_ndd, y_valid_ndd = utils.get_train_and_validation_data(
    scaled=True,
    use_diagnosis_delay=False,
)



datasets = [
    ['All_Features', X_train_all, y_train_all, X_valid_all, y_valid_all],
    ['without_Diagnosis_Delay', X_train_ndd, y_train_ndd, X_valid_ndd, y_valid_ndd],
] 

# ======================================================================
# ======================================================================


# where to save results files
dir_dest = os.path.abspath('0_exec_results_py')


# define the Cross-Validation strategy
CV_N_SPLITS = 4
CV_N_REPEATS = 3
RANDOM_STATE = 42

# CV strategy
cv = sk.model_selection.RepeatedStratifiedKFold(
    n_splits=CV_N_SPLITS, 
    n_repeats=CV_N_REPEATS, 
    random_state=RANDOM_STATE
)


grids_executed = []


# get all param_grid combinations for each classifier
testing = True
# testing = False


grid_configs = [
    # OK 0 utils_exec_models_new.create_models_NB_Gaussian_grid(testing=testing),
    # OK 0 utils_exec_models_new.create_models_kNN_grid(testing=testing),

    # OK 0 utils_exec_models_new.create_models_NB_Complement_grid(testing=testing),
    
    # OK 0 utils_exec_models_new.create_models_RadiusNN_grid(testing=testing),

    # OK 0 utils_exec_models_new.create_models_DT_grid(testing=testing),
    # OK 0 utils_exec_models_new.create_models_SVM_grid(testing=testing),    

    # OK 0 ['NeuralNetworks', None],

    utils_exec_models_new.create_models_RF_grid(testing=testing),
    


]



# ======================================================================
# ======================================================================
# ======================================================================
# ======================================================================

# for each features_config and datasets
for features_config, X_train, y_train, X_valid, y_valid in datasets:
    
    # for each ML algorithm and param_grid
    for classifier, param_grid in grid_configs: 
    
        
        # configure Neural Networks according to the number of features
        if classifier == 'NeuralNetworks':
            classifier, param_grid = utils_exec_models_new.create_models_NN_grid(
                qty_features=X_train.shape[1], 
                testing=testing
            )

    
        model_desc = utils.get_model_short_description(classifier).replace('-', '')
        utils.print_string_with_separators(f'{classifier} - {features_config}')


        # ====================================================
        # execute gridSearch in the Single-Model scenario
        # ====================================================
        scenario = 'Single_Model'
        print(f'   Executing {scenario}')
        
        grid = sk.model_selection.GridSearchCV(
            estimator=classifier, 
            param_grid=param_grid, 
            scoring=utils_exec_models_new.get_default_scoring(), 
            cv=cv,
            verbose=1,
            n_jobs=utils_exec_models_new.N_JOBS, #7
            return_train_score=True,
            refit=utils_exec_models_new.DEFAULT_SCORE # balanced accuracy
        )

        # fit the grid and save the trainning and validation performances
        start = timer()
        grid, df_validation_performances = utils_exec_models_new.exec_grid_search_and_save_performances(
            dir_dest=dir_dest, 
            testing=testing, 
            grid=grid, 
            classifier=classifier, 
            scenario=scenario, 
            features_config=features_config, 
            X_train=X_train, 
            y_train=y_train, 
            X_valid=X_valid, 
            y_valid=y_valid,
        )
        print(timer() - start)
        print(f' FINISHED !!! [{model_desc} - {scenario} - {features_config}]')
        print()

        
        # =======================================================
        # execute gridSearch in the Ensemble_Imbalance scenario
        # =======================================================

        scenario = 'Ensemble_Imbalance'
        print(f'   Executing {scenario}')

        # verify if is executing a Random Forest classifier
        if str(classifier) == 'RandomForestClassifier()':
            es_classifier, es_estimator, es_param_grid = utils_exec_models_new.create_models_BalancedRandomForest_grid(
                testing=testing,
            )
        #
        # for Balanced Bagging classifier
        else:    

            # use the best 10 performances as estimator to Balanced-Bagging    
            models_to_use_as_estimator = utils_exec_models_new.create_model_instances_from_performances(
                df=df_validation_performances.head(10)
            )

            es_classifier, es_estimator, es_param_grid = utils_exec_models_new.create_models_BalancedBagging_grid(
                estimator=models_to_use_as_estimator,
                testing=testing,
            )

        #execute gridSearch
        es_grid = sk.model_selection.GridSearchCV(
            estimator=es_classifier, 
            param_grid=es_param_grid, 
            scoring=utils_exec_models_new.get_default_scoring(), 
            cv=cv,
            verbose=1,
            n_jobs=utils_exec_models_new.N_JOBS, #7
            return_train_score=True,
            refit=utils_exec_models_new.DEFAULT_SCORE # balanced accuracy
        )

        # fit the grid and save the trainning and validation performances
        start = timer()
        es_grid, es_df_validation_performances = utils_exec_models_new.exec_grid_search_and_save_performances(
            dir_dest=dir_dest, 
            testing=testing, 
            grid=es_grid, 
            classifier=es_classifier, 
            scenario=scenario, 
            features_config=features_config, 
            X_train=X_train, 
            y_train=y_train, 
            X_valid=X_valid, 
            y_valid=y_valid,
        )
        print(timer() - start)
        
        
        # clear_output() 

        print(f' FINISHED !!! [{model_desc} - {scenario} - {features_config}]')
        print()
        
#         break
#     break
    

    
print(f' FINISHED ALL !!!')
    
# display(df_validation_performances.head(2))    
# display(es_df_validation_performances)    

# ======================================================================
# ======================================================================
# ======================================================================
# ======================================================================
        
        
