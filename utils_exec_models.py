
import sklearn as sk
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, make_scorer, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


# CONSTANT to store the random_state stated for reproducibility issues
RANDOM_STATE = 42



CLASS_COLUMN = 'Survival_Group'

def split_training_testing_validation(df, test_size=0.2, random_state=RANDOM_STATE, stratify=None):

    # separate into input and output variables
    input_vars = df.copy()
    input_vars.drop(columns=[CLASS_COLUMN])

    output_var = df[[CLASS_COLUMN]].copy()

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
    df_train[CLASS_COLUMN] = y_train[CLASS_COLUMN]

    df_valid = X_valid.copy()
    df_valid[CLASS_COLUMN] = y_valid[CLASS_COLUMN]

    #
    return df_train, df_valid
