from typing import Dict, Union

import numpy as np
import pandas as pd
import pymc as pm
import pytensor as pt
from pymc_experimental.model_builder import ModelBuilder


class LinearRHSESF(ModelBuilder):
    # Give the model a name
    _model_type = "LinearRHSESF"

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
            x_data = pm.MutableData("x_data", X_values[:, 0:1])
            x_stdev = pm.MutableData("x_stdev", X_values[:, 1:2])
            Ei_data = pm.MutableData("E_data", X_values[:, 2:])
            y_data = pm.MutableData("y_data", y_values)

            # prior parameters
            a0_mu_prior = self.model_config.get("a0_mu_prior")
            a0_sigma_prior = self.model_config.get("a0_sigma_prior")
            a1_mu_prior = self.model_config.get("a1_mu_prior")
            a1_sigma_prior = self.model_config.get("a1_sigma_prior")
            sigma_y_alpha_prior = self.model_config.get("sigma_y_alpha_prior")
            sigma_y_beta_prior = self.model_config.get("sigma_y_beta_prior")

            c2_alpha_prior = self.model_config.get("c2_alpha")
            c2_beta_prior = self.model_config.get("c2_beta")
            tau_alpha_prior = self.model_config.get("tau_alpha")
            lbda_alpha_prior = self.model_config.get("lbda_alpha")
            b_mu_prior = self.model_config.get("b_mu_prior")

            # priors
            c2 = pm.InverseGamma("c2", alpha=c2_alpha_prior, beta=c2_beta_prior)
            tau = pm.HalfCauchy("tau", tau_alpha_prior)

            # RHS
            # Specify the number of eigenvectors as a fixed constant, not as
            # a variable depending on the (mutable) input data since any random
            # variable depending on mutable data will be ignored when sampling
            # from the posterior predictive distribution.
            D = self.model_config.get("n_eigenvectors")
            lbdai = pm.HalfCauchy("lbdai", lbda_alpha_prior, shape=(D,))
            scalei = tau * np.sqrt((c2 * lbdai**2) / (c2 + tau**2 * lbdai**2))
            bi = pm.Normal("bi", b_mu_prior, scalei, shape=(1, D)).T

            # Regression coefficients
            a0 = pm.Normal("a0", mu=a0_mu_prior, sigma=a0_sigma_prior)
            a1 = pm.Normal("a1", mu=a1_mu_prior, sigma=a1_sigma_prior)
            sigma_y = pm.InverseGamma(
                "sigma_y", alpha=sigma_y_alpha_prior, beta=sigma_y_beta_prior
            )
            # model
            x_est = pm.Normal("x_est", x_data, sigma=x_stdev)
            mu = a0 + x_est * a1 + pm.math.dot(Ei_data, bi)
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
            x_values = X.values[:, 0:1]
            x_stdev = X.values[:, 1:2]
            Ei_values = X.values[:, 2:]
        else:
            x_values = X[:, 0:1]
            x_stdev = X[:, 1:2]
            Ei_values = X[:, 2:]

        with self.model:
            pm.set_data({"x_data": x_values})
            pm.set_data({"x_stdev": x_stdev})
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
            # RHS priors
            "n_eigenvectors": 4,
            "c2_alpha": 7.5,
            "c2_beta": 0.3,
            "tau_alpha": 0.1,
            "lbda_alpha": 1.0,
            "b_mu_prior": 0.0,
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
