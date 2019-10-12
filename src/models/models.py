import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fbprophet import Prophet


class SetTempAsPower:
    """
    Makes a forecast by selecting the hottest days
    """

    def __init__(self, col="temp_max"):
        self.col = col

    def fit(self, X, y):
        self.y_fit = y
        self.X_fit = X
        self.max_power = y.max()
        self.min_power = y.min()
        return self

    def predict(self, X):
        self.X_predict = X
        minmaxscaler = MinMaxScaler(feature_range=(self.min_power, self.max_power))
        minmaxscaler.fit(self.X_fit[[self.col]])
        scaled = minmaxscaler.transform(self.X_predict[[self.col]])
        self.predict_yhat = pd.Series(
            data=scaled.reshape(1, -1)[0], index=self.X_predict.index
        )
        return self.predict_yhat

    def get_pred_values(self):
        # All Data is only available after fit and predict
        # Build a return DataFrame that looks similar to the prophet output
        # date index | y | yhat| yhat_lower | yhat_upper | is_forecast
        X_fit = self.X_fit.copy()
        X_predict = self.X_predict.copy()
        y = self.y_fit
        if self.X_fit.equals(self.X_predict):
            yhat = self.predict_yhat.copy(deep=True)

        else:
            self.fit(X_fit, y)
            y.name = "y"
            fit_yhat = self.predict(X_fit)
            pred_yhat = self.predict(X_predict)
            y_pred = pd.Series(np.NaN, index=X_predict.index)
            y = pd.concat([y, y_pred], axis=0)
            yhat = pd.concat([fit_yhat, pred_yhat], axis=0)

        yhat.name = "yhat"
        y.name = "y"
        y = self.y_fit.copy()
        full_suite = pd.concat([yhat, y], axis=1)
        full_suite["is_forecast"] = 0
        full_suite["is_forecast"] = full_suite["y"].isna().astype(int)
        full_suite["yhat_upper"] = np.NaN
        full_suite["yhat_lower"] = np.NaN

        full_suite = full_suite[
            ["y", "yhat", "yhat_lower", "yhat_upper", "is_forecast"]
        ]

        return full_suite


class SK_SARIMAX(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """

    def __init__(self, order=(2, 0, 1), seasonal_order=(2, 0, 0, 96), trend="c"):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend

    def fit(self, X, y):
        self.fit_X = X
        self.fit_y = y
        self.model = SARIMAX(
            self.fit_y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            exog=self.fit_X,
        )
        self.results = self.model.fit()
        return self.model

    def predict(self, X, y=None):
        self.predict_X = X
        self.forecast_object = self.results.get_forecast(steps=len(X), exog=X)
        self.conf_int = self.forecast_object.conf_int()
        self.ser = pd.Series(
            data=self.forecast_object.predicted_mean.values, index=self.predict_X.index
        )
        return self.ser


    def get_pred_values(self):
        # All Data is only available after fit and predict
        # Build a return DataFrame that looks similar to the prophet output
        # date index y | yhat| yhat_lower | yhat_upper | is_forecast
        fitted = self.fit_y.copy(deep=True)
        fitted.name = "y"
        fitted = pd.DataFrame(fitted)
        fitted["is_forecast"] = 0
        fitted["yhat"] = self.results.predict()

        ser = self.ser.copy(deep=True)
        predict_y = pd.DataFrame(self.ser, columns=["yhat"])
        predict_y["is_forecast"] = 1

        conf_ints = pd.DataFrame(
            self.forecast_object.conf_int().values,
            index=predict_y.index,
            columns=["yhat_lower", "yhat_upper"],
        )

        unknown = pd.concat([predict_y, conf_ints], axis=1, sort=True)

        full_suite = pd.concat([fitted, unknown], axis=0, sort=True)
        full_suite = full_suite[
            ["y", "yhat", "yhat_lower", "yhat_upper", "is_forecast"]
        ]

        return full_suite

    
    
class SK_Prophet(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    
    def __init__(self, regressors={}, pred_periods=96):
#         self.pred_periods=pred_periods
        self.regressors=regressors
       
        
    def prep_X(self, X):
        # Prophet requires a DataFrame with a column of dates labeled 'ds'
        if 'ds' not in X.columns:
            X = X.assign(ds = X.index)
            X.reset_index(drop=True, inplace=True)
        return X
    
    def prep_y(self, y):
        # prohet requires the target to be labeled as y
        if y.name is not 'y':
            y.name = 'y'
        return y
        
    def fit(self, X, y):
        self.X_fit = X.copy(deep=True)
        self.y_fit = y.copy(deep=True)
        
        # Setup the model
        self.model=Prophet(daily_seasonality=True)
        self.model.add_seasonality(name='summer', period=96, fourier_order=5)
        self.model.add_seasonality(name='workweek', period=5, fourier_order=5)
        self.model.seasonality_mode= 'multiplicative'

        # If regressors is not empty
        if (self.regressors.keys()):
            for regressor, params in self.regressors.items():
                if params:
                    self.model.add_regressor(regressor, prior_scale=params[0], mode=params[1])
                else:
                    self.model.add_regressor(regressor)
        
        # Setup the data
        X = self.prep_X(X)
        y = self.prep_y(y)       
        df = X.merge(right=y, left_on='ds', right_on=y.index)
        
        self.model.fit(df)
        return self.model
        
    def predict(self, X):
        self.X_pred = X.copy(deep=True)
        X_ = self.prep_X(self.X_pred)
        forecast_object = self.model.predict(X_)
        # When we return the prediction, we are looking to return a datetime indexed series
        # Therefore, we need to reverse what was done in prep_X prior to returning
        preds = forecast_object.copy()
        preds.set_index('ds', drop=True, inplace=True)
        preds = preds['yhat']
        preds.index.names=['date']
        return preds
        #return forecast_object['yhat'].values
    
    def get_pred_values(self):
        # All Data is only available after fit and predict
        # Build a return DataFrame that looks similar to the prophet output
        # date index y | yhat| yhat_lower | yhat_upper | is_forecast
        full_X = pd.concat([self.prep_X(self.X_fit),
                            self.prep_X(self.X_pred)],
                            axis=0).reset_index(drop=True)
        forecast_obj = self.model.predict(full_X)
        forecast_obj['is_forecast'] = 0
        forecast_obj.set_index('ds',drop=True, inplace=True)
        del forecast_obj.index.name
        forecast_obj.loc[self.y_fit.index, 'y'] = self.y_fit.values
        forecast_obj.loc[self.X_pred.index, 'is_forecast'] = 1
        
        return forecast_obj

class SK_Prophet_1(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    
    def __init__(self, regressors={}, pred_periods=96):
        self.regressors=regressors
       
        
    def prep_X(self, X):
        # Prophet requires a DataFrame with a column of dates labeled 'ds'
        if 'ds' not in X.columns:
            X = X.assign(ds = X.index)
            X.reset_index(drop=True, inplace=True)
        return X
    
    def prep_y(self, y):
        # prohet requires the target to be labeled as y
        if y.name is not 'y':
            y.name = 'y'
        return y
        
    def fit(self, X, y):
        self.X_fit = X.copy(deep=True)
        self.y_fit = y.copy(deep=True)
        
        # Setup the model with No seasonality
        self.model=Prophet(daily_seasonality=False)

        # If regressors is not empty
        if (self.regressors.keys()):
            for regressor, params in self.regressors.items():
                if params:
                    self.model.add_regressor(regressor, prior_scale=params[0], standardize=params[1], mode=params[2])
                else:
                    self.model.add_regressor(regressor)
        
        # Setup the data
        X = self.prep_X(X)
        y = self.prep_y(y)       
        df = X.merge(right=y, left_on='ds', right_on=y.index)
        
        self.model.fit(df)
        return self.model
        
    def predict(self, X):
        self.X_pred = X.copy(deep=True)
        X_ = self.prep_X(self.X_pred)
        forecast_object = self.model.predict(X_)
        # When we return the prediction, we are looking to return a datetime indexed series
        # Therefore, we need to reverse what was done in prep_X prior to returning
        preds = forecast_object.copy()
        preds.set_index('ds', drop=True, inplace=True)
        preds = preds['yhat']
        preds.index.names=['date']
        return preds
    
    def get_pred_values(self):
        # All Data is only available after fit and predict
        # Build a return DataFrame that looks similar to the prophet output
        # date index y | yhat| yhat_lower | yhat_upper | is_forecast
        full_X = pd.concat([self.prep_X(self.X_fit),
                            self.prep_X(self.X_pred)],
                            axis=0).reset_index(drop=True)
        forecast_obj = self.model.predict(full_X)
        forecast_obj['is_forecast'] = 0
        forecast_obj.set_index('ds',drop=True, inplace=True)
        del forecast_obj.index.name
        forecast_obj.loc[self.y_fit.index, 'y'] = self.y_fit.values
        forecast_obj.loc[self.X_pred.index, 'is_forecast'] = 1
        
        return forecast_obj
