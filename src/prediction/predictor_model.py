import os
import warnings
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from piml.models import XGB2Regressor
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Regressor:
    """A wrapper class for the XGB2Regressor.

    This class provides a consistent interface that can be used with other
    regressor models.
    """

    model_name = "XGB2Regressor"

    def __init__(
        self,
        n_estimators: Optional[int] = 50,
        eta: float = 0.3,
        max_depth: int = 2,
        tree_method: str = "auto",
        max_bin: int = 256,
        reg_lambda: float = 0,
        reg_alpha: float = 0,
        gamma: float = 0,
        random_state: int = 0,
        **kwargs,
    ):
        """Construct a new XGB2Regressor.

        Args:
             n_estimators (int): Number of gradient boosted trees.
            eta (float): Boosting learning rate.
            max_depth (int): Max tree depth.
            tree_method (str): {“exact”, “hist”, “approx”, “gpu_hist”, “auto”} Specify which tree method to use.
            max_bin (int): If using histogram-based algorithm, maximum number of bins per feature.
            reg_alpha (float): L1 regularization term on weights.
            reg_lambda (float): L2 regularization term on weights.
            gamma (float): Minimum loss reduction required to make a further partition on a leaf node of the tree.
            random_state (int): The random seed.
        """
        self.n_estimators = n_estimators
        self.eta = eta
        self.max_depth = max_depth
        self.tree_method = tree_method
        self.max_bin = max_bin
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> XGB2Regressor:
        """Build a new regressor."""
        model = XGB2Regressor(
            n_estimators=self.n_estimators,
            eta=self.eta,
            max_depth=self.max_depth,
            tree_method=self.tree_method,
            max_bin=self.max_bin,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            gamma=self.gamma,
            random_state=self.random_state,
            **self.kwargs,
        )
        return model

    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the regressor to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """
        self.model.fit(train_inputs, train_targets)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict regression targets for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted regression targets.
        """
        return self.model.predict(inputs)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the regressor and return the r-squared score.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The targets of the test data.
        Returns:
            float: The r-squared score of the regressor.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the regressor to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Regressor":
        """Load the regressor from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Regressor: A new instance of the loaded regressor.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return (
            f"Model name: {self.model_name} ("
            f"eta: {self.eta}, "
            f"gamma: {self.gamma}, "
            f"max_depth: {self.max_depth}, "
            f"n_estimators: {self.n_estimators})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Regressor:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data targets.
        hyperparameters (dict): Hyperparameters for the regressor.

    Returns:
        'Regressor': The regressor model
    """
    regressor = Regressor(**hyperparameters)
    regressor.fit(train_inputs=train_inputs, train_targets=train_targets)
    return regressor


def predict_with_model(regressor: Regressor, data: pd.DataFrame) -> np.ndarray:
    """
    Predict regression targets for the given data.

    Args:
        regressor (Regressor): The regressor model.
        data (pd.DataFrame): The input data.

    Returns:
        np.ndarray: The predicted regression targets.
    """
    return regressor.predict(data)


def save_predictor_model(model: Regressor, predictor_dir_path: str) -> None:
    """
    Save the regressor model to disk.

    Args:
        model (Regressor): The regressor model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Regressor:
    """
    Load the regressor model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Regressor: A new instance of the loaded regressor model.
    """
    return Regressor.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Regressor, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the regressor model and return the r-squared value.

    Args:
        model (Regressor): The regressor model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The targets of the test data.

    Returns:
        float: The r-sq value of the regressor model.
    """
    return model.evaluate(x_test, y_test)
