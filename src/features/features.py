from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class CyclicalToCycle(BaseEstimator, TransformerMixin):
    """ A transformer that returns the sin and cosine values of a cyclical sequence
    For more information regarding how to build your own transformer, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """

    def __init__(self, cycle_name="time_of_day", periods_per_cycle=24):
        self.cycle_name = cycle_name
        self.periods_per_cycle = periods_per_cycle

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        X = X.copy(deep=True)
        X["sin_" + self.cycle_name] = np.sin(
            2 * np.pi * X[self.cycle_name] / self.periods_per_cycle
        )
        X["cos_" + self.cycle_name] = np.cos(
            2 * np.pi * X[self.cycle_name] / self.periods_per_cycle
        )
        X.drop([self.cycle_name], axis=1, inplace=True)

        return X
