import numpy as np
import statsmodels.api as sm
import xarray as xr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


# Polynomial fit
# =============================================================================
def np_polynomial_fit(x, y):
    # Remove NaNs
    is_valid = ~np.isnan(y)
    x = x[is_valid]
    y = y[is_valid]
    
    # Using statsmodels to get confidence intervals
    X = np.column_stack((np.ones(len(x)), x, x**2))
    model = sm.OLS(y, X)
    results = model.fit()
    return results


def np_polynomial_predict(model, xnew):
    # Generate prediction interval
    xnew = np.vstack([[1]*len(xnew), xnew, xnew**2]).T
    predictions = model.get_prediction(xnew)
    frame = predictions.summary_frame(alpha=0.05)
    frame['obs_std'] = (frame['obs_ci_upper'] - frame['mean']) / 2
    y_pred = frame['mean'].values
    y_pred_std = frame['obs_std'].values
    return y_pred, y_pred_std

def polynomial_fit(x, y, dim):
    return xr.apply_ufunc(
        np_polynomial_fit,
        x,
        y,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[]],
        exclude_dims=set((dim,)),
        vectorize=True,
        dask='parallelized',
    )


def polynomial_predict(model, xnew, dim):
    y_pred_mean, y_pred_stdev = xr.apply_ufunc(
        np_polynomial_predict,
        model,
        xnew,
        input_core_dims=[[], [dim]],
        output_core_dims=[[dim], [dim]],
        vectorize=True,
        dask='parallelized',
    )
    y_pred_mean.name = 'density_pred_mean'
    y_pred_stdev.name = 'density_pred_stdev'
    res = xr.merge([y_pred_mean, y_pred_stdev])
    res['d18Oc_mean'] = xnew
    return res


def extract_coefs(obj):
    deg0 = xr.apply_ufunc(lambda x: x.params[0], obj, vectorize=True)
    deg1 = xr.apply_ufunc(lambda x: x.params[1], obj, vectorize=True)
    deg2 = xr.apply_ufunc(lambda x: x.params[2], obj, vectorize=True)
    coefs = xr.concat([deg0, deg1, deg2], dim='degree').assign_coords({'degree': [0, 1, 2]})
    coefs.name = 'regression_coefficients'
    return coefs

def extract_rsquared(obj):
    return xr.apply_ufunc(lambda x: x.rsquared, obj, vectorize=True)

def txt_coefs(coefs, degree, fmt='.2f'):
    return f'$a_{degree} = {coefs.sel(degree=degree, quantile=.5).data:{fmt}}$ $({coefs.sel(degree=degree, quantile=.025).data:{fmt}}\dots{coefs.sel(degree=degree, quantile=.975).data:{fmt}})$'


# %%
# Some helper functions
# -----------------------------------------------------------------------------

def _np_gp_fit(x, y, kernel=None, constant_value=None, length_scale=None):
    # bring in sklearn format
    x_train = x.reshape(-1, 1)
    y_train = y.reshape(-1, 1)
    
    # Remove NaNs
    is_valid = ~np.isnan(y)
    x_train = x_train[is_valid, :]
    y_train = y_train[is_valid, :]

    # Define kernel
    if constant_value is None:
        constant_value = 1e-2
        constant_value_bounds = (1e-5, 1e2)
    else:
        constant_value_bounds = 'fixed'

    if length_scale is None:
        length_scale = 1
        length_scale_bounds = (1e-1, 1000)
    else:
        length_scale_bounds = 'fixed'
    
    kernel_short = ConstantKernel(constant_value, constant_value_bounds) * RBF(
        length_scale=length_scale, length_scale_bounds=length_scale_bounds,
    )
    kernel_noise = WhiteKernel(
        noise_level=1, noise_level_bounds=(1e-20, 1e1)
    )
    if kernel is None:
        kernel = kernel_noise + kernel_short

    # Make fit
    gpr = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=9, normalize_y=True
    )
    gpr.fit(x_train, y_train)
    return gpr

def _np_gp_predict(gpr, x_pred):
    y_pred = np.zeros_like(x_pred) * np.nan
    y_stdev = np.zeros_like(x_pred) * np.nan
    
    # Remove NaNs
    is_valid = ~np.isnan(x_pred)
    x_pred = x_pred[is_valid]

    # bring in sklearn format
    x_pred = x_pred.reshape(-1, 1)
    y_pred[is_valid], y_stdev[is_valid] = gpr.predict(x_pred, return_std=True)
    return y_pred, y_stdev

def gpr_fit(x, y, dim, kernel=None, constant_value=None, length_scale=None):
    gpr = xr.apply_ufunc(
        _np_gp_fit,
        x,
        y,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[]],
        vectorize=True,
        kwargs={'kernel': kernel, 'constant_value': constant_value, 'length_scale': length_scale},
    )
    return gpr

def gpr_predict(gpr, x_pred, dim):
    da_mean, da_stdev = xr.apply_ufunc(
        _np_gp_predict,
        gpr,
        x_pred,
        input_core_dims=[[], [dim]],
        output_core_dims=[[dim], [dim]],
        vectorize=True,
    )
    da_mean.name = 'density_pred_mean'
    da_stdev.name = 'density_pred_stdev'
    res = xr.merge([da_mean, da_stdev])
    res['d18Oc_mean'] = x_pred
    return res

def compute_ci(mean_prediction, stdev):
    # Compute confidence intervals
    da_ci = mean_prediction.expand_dims({'ci': ['std--', 'std-', 'mean', 'std+', 'std++']})
    da_ci = da_ci.copy(deep=True)
    da_ci.loc[{'ci': 'std--'}] = mean_prediction - 2 * stdev
    da_ci.loc[{'ci': 'std-'}] = mean_prediction - stdev
    da_ci.loc[{'ci': 'std+'}] = mean_prediction + stdev
    da_ci.loc[{'ci': 'std++'}] = mean_prediction + 2 * stdev
    da_ci.name = 'abundance'
    return da_ci
