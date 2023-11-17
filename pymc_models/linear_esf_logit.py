from typing import Dict, Union

import numpy as np
import pandas as pd
import pymc as pm
from pymc_experimental.model_builder import ModelBuilder


class LinearLogitESF(ModelBuilder):
    # Give the model a name
    _model_type = "LinearLogitESF"

    # And a version
    version = "0.1"

    def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        build_model creates the PyMC model

        Parameters:
        model_config: dictionary
            it is a dictionary with all the parameters that we need in our model example:  a_loc, a_scale, b_loc
        X : pd.DataFrame
            The input data that is going to be used in the model. This should be a DataFrame
            containing the features (predictors) for the model. For efficiency reasons, it should
            only contain the necessary data columns, not the entire available dataset, as this
            will be encoded into the data used to recreate the model.

        y : pd.Series
            The target data for the model. This should be a Series representing the output
            or dependent variable for the model.

        kwargs : dict
            Additional keyword arguments that may be used for model configuration.
        """
        # Check the type of X and y and adjust access accordingly
        X_values = X.values
        y_values = y.values if isinstance(y, pd.Series) else y
        self._generate_and_preprocess_model_data(X_values, y_values)

        with pm.Model(coords=self.model_coords) as self.model:
            # Create mutable data containers
            x_data = pm.MutableData("x_data", X_values[:, :1])
            Ei_data = pm.MutableData("E_data", X_values[:, 1:])
            y_data = pm.MutableData("y_data", y_values)

            # prior parameters
            a0_mu_prior = self.model_config.get("a0_mu_prior")
            a0_sigma_prior = self.model_config.get("a0_sigma_prior")
            a1_mu_prior = self.model_config.get("a1_mu_prior")
            a1_sigma_prior = self.model_config.get("a1_sigma_prior")
            sigma_y_alpha_prior = self.model_config.get("sigma_y_alpha_prior")
            sigma_y_beta_prior = self.model_config.get("sigma_y_beta_prior")

            tau = self.model_config.get("tau")
            mu_lbda_prio = self.model_config.get("mu_lbda")
            sigma_lbda_prior = self.model_config.get("sigma_lbda")
            bi_mu_prior = self.model_config.get("bi_mu")

            # priors
            a0 = pm.Normal("a0", mu=a0_mu_prior, sigma=a0_sigma_prior)
            a1 = pm.Normal("a1", mu=a1_mu_prior, sigma=a1_sigma_prior)
            sigma_y = pm.InverseGamma(
                "sigma_y", alpha=sigma_y_alpha_prior, beta=sigma_y_beta_prior
            )
            # Logit prior
            D = self.model_config.get("n_eigenvectors")
            lbdai = pm.LogitNormal("lbdai", mu_lbda_prio, sigma_lbda_prior, shape=(D,))
            scalei = tau * lbdai
            bi = pm.Normal("bi", bi_mu_prior, scalei, shape=(1, D)).T

            # model
            mu = a0 + x_data * a1 + pm.math.dot(Ei_data, bi)
            mu = mu[:, 0]
            obs = pm.Normal(
                "sigmaT",
                mu,
                sigma=sigma_y,
                observed=y_data,
                shape=x_data.shape[0],
                dims="d18Oc",
            )

    def _data_setter(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray] = None
    ):
        if isinstance(X, pd.DataFrame):
            x_values = X.values[:, :1]
            Ei_values = X.values[:, 1:]
        else:
            x_values = X[:, :1]
            Ei_values = X[:, 1:]

        with self.model:
            pm.set_data({"x_data": x_values})
            pm.set_data({"E_data": Ei_values})
            if y is not None:
                pm.set_data({"y_data": y.values if isinstance(y, pd.Series) else y})

    @staticmethod
    def get_default_model_config() -> Dict:
        """
        Returns a class default config dict for model builder if no model_config is provided on class initialization.
        The model config dict is generally used to specify the prior values we want to build the model with.
        It supports more complex data structures like lists, dictionaries, etc.
        It will be passed to the class instance on initialization, in case the user doesn't provide any model_config of their own.
        """
        model_config: Dict = {
            # GLM priors
            "a0_mu_prior": 0.0,
            "a1_mu_prior": 0.0,
            "a0_sigma_prior": 1000.0,
            "a1_sigma_prior": 1000.0,
            "sigma_y_alpha_prior": 1.0,
            "sigma_y_beta_prior": 1.0,
            # Logit prios
            "n_eigenvectors": 4,
            "tau": 5.0,
            "mu_lbda": 0.0,
            "sigma_lbda": 10.0,
            "bi_mu": 0.0,
        }
        return model_config

    @staticmethod
    def get_default_sampler_config() -> Dict:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization.
        The sampler config dict is used to send parameters to the sampler .
        It will be used during fitting in case the user doesn't provide any sampler_config of their own.
        """
        sampler_config: Dict = {
            "draws": 1_000,
            "tune": 1_000,
            "chains": 4,
            "target_accept": 0.5,
            "cores": 1,
        }
        return sampler_config

    @property
    def output_var(self):
        return "sigmaT"

    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        """
        _serializable_model_config is a property that returns a dictionary with all the model parameters that we want to save.
        as some of the data structures are not json serializable, we need to convert them to json serializable objects.
        Some models will need them, others can just define them to return the model_config.
        """
        return self.model_config

    def _save_input_params(self, idata) -> None:
        """
        Saves any additional model parameters (other than the dataset) to the idata object.

        These parameters are stored within `idata.attrs` using keys that correspond to the parameter names.
        If you don't need to store any extra parameters, you can leave this method unimplemented.

        Example:
            For saving customer IDs provided as an 'customer_ids' input to the model:
            self.customer_ids = customer_ids.values #this line is done outside of the function, preferably at the initialization of the model object.
            idata.attrs["customer_ids"] = json.dumps(self.customer_ids.tolist())  # Convert numpy array to a JSON-serializable list.
        """
        pass

    def _generate_and_preprocess_model_data(
        self, X: Union[pd.DataFrame, pd.Series], y: Union[pd.Series, np.ndarray]
    ) -> None:
        """
        Depending on the model, we might need to preprocess the data before fitting the model.
        all required preprocessing and conditional assignments should be defined here.
        """
        # Coords works for inference but not for prediction (?)
        # coords = X["d18Oc"].values if isinstance(X, pd.DataFrame) else X[:, 0]
        # self.model_coords = {"d18Oc": coords}
        self.model_coords = None

        self.X = X
        self.y = y
