import pandas as pd

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
)


class AnnualTimeSeriesSplit():
    """
    Instantiate with number of folds
    split accepts a pandas dataframe indexed by datetime covering multiple years sorted ascending
    Splits to the number of folds, with a single year returned as the validation set
    Walks up the timeseries yielding the indices from each train, test split
    """
    def __init__(self, n_splits):
        self.n_splits = n_splits
        
    def split(self, X, y=None, groups=None):
        years = X.index.year.unique()
        
        for ind, year in enumerate(years[0:self.n_splits]):
            
            final_train_year = years[-1] - self.n_splits + ind
            
            train_final_index = X.index.get_loc(str(final_train_year)).stop
            test_final_index = X.index.get_loc(str(final_train_year + 1)).stop
            
            train_indices = list(range(0, train_final_index))
            test_indices = list(range(train_final_index, test_final_index))
            
            yield train_indices, test_indices

            
class RollingAnnualTimeSeriesSplit():
    """
    Instantiate with number of folds
    split accepts a pandas dataframe indexed by datetime covering multiple years sorted ascending
    Splits to the number of folds, with a single year returned as the validation set
    Walks up the timeseries yielding the indices from each train, test split
    """
    def __init__(self, n_splits, goback_years=5):
        self.n_splits = n_splits
        self.goback_years = goback_years
        
    def split(self, X, y=None, groups=None):
        years = X.index.year.unique()
        
        for ind, year in enumerate(years[0:self.n_splits]):
            
            final_train_year = years[-1] - self.n_splits + ind
            start_train_year = final_train_year - self.goback_years +1
            print(f'{final_train_year+1}')
            
            train_start_index = X.index.get_loc(str(start_train_year)).start
            train_final_index = X.index.get_loc(str(final_train_year)).stop
            test_final_index = X.index.get_loc(str(final_train_year + 1)).stop
            
            train_indices = list(range(train_start_index, train_final_index))
            test_indices = list(range(train_final_index, test_final_index))
            
            yield train_indices, test_indices


def bound_precision(y_actual: pd.Series, y_predicted: pd.Series, n_to_check=5):
    """
    Accepts two pandas series, and an integer n_to_check
    Series are:
    + actual values
    + predicted values
    Sorts each series by value from high to low, and cuts off each series at n_to_check
    Determines how many hits - ie how many of the indices in the actual series are in the predicted series indices
    Returns number of hits divided by n_to_check    
    """
    y_act = y_actual.copy(deep=True)
    y_pred = y_predicted.copy(deep=True)
    y_act.reset_index(drop=True, inplace=True)
    y_pred.reset_index(drop=True, inplace=True)

    act_dates =set( y_act.sort_values(ascending=False).head(n_to_check).index)
    pred_dates = set(y_pred.sort_values(ascending=False).head(n_to_check).index)
    bound_precision =  len(act_dates.intersection(pred_dates))/ n_to_check
    return bound_precision


def run_cross_val(X, y, cv_splitter, pipeline, scoring=["bound_precision", "mae"]):
    scoring_dict = {
        "bound_precision": bound_precision,
        "mae": mean_absolute_error,
        "r2_score": r2_score,
        "median_absolute_error": median_absolute_error,
        "mean_squared_error": mean_squared_error,
        "mean_squared_log_error": mean_squared_log_error,
    }

    scores_dicts = {}
    scores_dicts["train"] = {}
    scores_dicts["test"] = {}

    for metric in scoring:
        scores_dicts["train"][metric] = []
        scores_dicts["test"][metric] = []

    for train_indx, val_indx in cv_splitter.split(X):
        X_train = X.iloc[train_indx]
        y_train = y.iloc[train_indx]
        X_test = X.iloc[val_indx]
        y_test = y.iloc[val_indx]

        pipeline.fit(X_train, y_train)
        y_train_pred = pipeline.predict(X_train)

        for metric in scoring:
            score = scoring_dict[metric](y_train, y_train_pred)
            scores_dicts["train"][metric].append(score)

        y_test_pred = pipeline.predict(X_test)
        for metric in scoring:
            score = scoring_dict[metric](y_test, y_test_pred)
            scores_dicts["test"][metric].append(score)
            #print(f'test: {score}')

    return scores_dicts


# data_split_dict
# traintest_split_dict
# metric_dict
# values
# {0:
#     {'train':
#             {'mae': [1.0, 1.1, 2.1, 3.2, 1.0, 2.1, 3.1, 2.1, 2,2, 2.3],
#              'mse': [1.0, 1.1, 2.1, 3.2, 1.0, 2.1, 3.1, 2.1, 2,2, 2.3]}
#      'test':
#             {'mae': [1.0, 1.1, 2.1, 3.2, 1.0, 2.1, 3.1, 2.1, 2,2, 2.3],
#              'mse': [1.0, 1.1, 2.1, 3.2, 1.0, 2.1, 3.1, 2.1, 2,2, 2.3]}},
#  1:
#     {'train':
#             {'mae': [1.0, 1.1, 2.1, 3.2, 1.0, 2.1, 3.1, 2.1, 2,2, 2.3],
#              'mse': [1.0, 1.1, 2.1, 3.2, 1.0, 2.1, 3.1, 2.1, 2,2, 2.3]}
#      'test':
#             {'mae': [1.0, 1.1, 2.1, 3.2, 1.0, 2.1, 3.1, 2.1, 2,2, 2.3],
#              'mse': [1.0, 1.1, 2.1, 3.2, 1.0, 2.1, 3.1, 2.1, 2,2, 2.3]}},
#  1:
#     {'train':
#             {'mae': [1.0, 1.1, 2.1, 3.2, 1.0, 2.1, 3.1, 2.1, 2,2, 2.3],
#              'mse': [1.0, 1.1, 2.1, 3.2, 1.0, 2.1, 3.1, 2.1, 2,2, 2.3]}
#      'test':
#             {'mae': [1.0, 1.1, 2.1, 3.2, 1.0, 2.1, 3.1, 2.1, 2,2, 2.3],
#              'mse': [1.0, 1.1, 2.1, 3.2, 1.0, 2.1, 3.1, 2.1, 2,2, 2.3]}}
#              }


def temporal_split(X, y, splitter_col="day_of_week"):
    X_splits = []
    y_splits = []
    Xt = X.copy(deep=True)
    yt = y.copy(deep=True)
    split_flags = sorted(Xt[splitter_col].unique())
    for split_flag in split_flags:
        X_split = Xt[Xt[splitter_col] == split_flag]
        X_splits.append(X_split)
        y_splits.append(y.loc[X_split.index])

    return X_splits, y_splits


def run_data_split_cross_val(
    X, y, data_splitter_col, cv_splitter, model, scoring=["mae"]
):

    scores_dict = {}

    X_splits, y_splits = temporal_split(X, y, data_splitter_col)

    for (indx_splitter, X_split, y_split) in zip(
        sorted(X[data_splitter_col].unique()), X_splits, y_splits
    ):
        scores_dict[indx_splitter] = run_cross_val(
            X_split, y_split, cv_splitter, model, scoring=scoring
        )
    return scores_dict

def save_run_results(X, n_splits, model_str, some_dict, save_path):
    """
    Accepts a dataset used for splitting into train and test,
    number of splits in a Rolling Annual Cross Validation Scheme
    A dictionary of validation results
    A save path for a dataframe
    Updates the file and saves
    Returns the updated file as a DataFrame
    
    """
    if save_path.exists():
        orig_df = pd.read_csv(save_path, index_col=0, header=[0,1] )
        if model_str in orig_df.columns:
            return "Model name already used"
    else:
        orig_df = None
    test_dict_name = 'test'
    n_metrics = len(some_dict[test_dict_name].keys())
    
    df = pd.DataFrame.from_dict(some_dict[test_dict_name])
    df.index = list(range(X.index.year.unique()[-n_splits], X.index.year.unique()[-1]+1,1))
    df.columns = pd.MultiIndex([[model_str], df.columns], labels = [[0]*n_metrics,list(range(n_metrics))])
    
    full_df = pd.concat([orig_df, df], axis=1)
    
    full_df.to_csv(save_path)
    
    return full_df   

