import datetime
import json
import os
import copy

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.stats import iqr
from scipy.integrate import quad

from models import LinearAffineModel
from co_calibration_gibbs_sampler_expressions import (
    posterior_Xa_explicit,
    posterior_a_explicit,
    posterior_b_explicit,
    posterior_sigma_y_explicit,
)

rstate = np.random.get_state()

# actual ground truth
t = np.linspace(0, 20, 201)
x_actual = np.sin(t) + 1

# reference values
Ux = np.diag(np.full_like(x_actual, 0.01))
x_observed = np.random.multivariate_normal(mean=x_actual, cov=Ux)

# indication of DUT
a_true = 2.0
b_true = 2.5
sigma_y_true = 0.2
true_transfer_dut = LinearAffineModel(a=a_true, b=b_true)
y = true_transfer_dut.apply(x_actual, np.zeros_like(x_actual))[0] + np.random.normal(scale=sigma_y_true, size=len(x_actual))

# special cases (activate by setting to True)
sigma_y_is_given = False
no_error_in_variables_model = False
use_robust_statistics = True

# init prior knowledge
prior = {
    "a" : {
        "mu" : 1.0,
        "sigma" : 1.0,
    },
    "b" : {
        "mu" : 0.0,
        "sigma" : 1.0,
    },
    "sigma_y" : {
        "mu" : 0.5,
        "sigma" : 0.3,
    }
}
init_prior = copy.deepcopy(prior)

# gibbs sampler settings
gibbs_runs = 10000
burn_in = gibbs_runs // 10
use_every = 100


# prepare some plotting
fig, ax = plt.subplots(3, 1)

# random splitpoints
n_splits = np.random.randint(1, len(t) // 3)
split_indices = np.sort(np.random.permutation(np.arange(3, len(t)-1))[:n_splits])

# iterate over splitpoints
parameters_history = {}
parameter_uncertainty_history = {}
for current_indices in np.split(np.arange(len(t)), split_indices):
    
    # available measurement information
    tt = t[current_indices]
    print(tt[0])
    xx_actual = x_actual[current_indices]
    xx_observed = x_observed[current_indices]
    Uxx = Ux[current_indices][:,current_indices]
    Uxx_inv = np.linalg.inv(Uxx)
    yy = y[current_indices]

    # plt
    ax[0].scatter(tt, yy)

    # shortcuts for prior
    mu_a = prior["a"]["mu"]
    mu_b = prior["b"]["mu"]
    mu_sigma_y = prior["sigma_y"]["mu"]

    sigma_a = prior["a"]["sigma"]
    sigma_b = prior["b"]["sigma"]
    sigma_sigma_y = prior["sigma_y"]["sigma"]

    # update posteriors using (block-)Gibbs sampling
    samples = []
    # initital sample from best guess (or any other method?)
    initial_sample = {
        "Xa" : xx_observed,
        "a" : mu_a,
        "b" : mu_b,
        "sigma_y" : mu_sigma_y,
    }
    samples.append(initial_sample)

    # init variables
    ps = samples[0]
    Xa_gibbs = ps["Xa"]
    a_gibbs = ps["a"]
    b_gibbs = ps["b"]
    sigma_y_gibbs = ps["sigma_y"]

    for i_run in range(1, gibbs_runs): # gibbs runs

        if no_error_in_variables_model:
            Xa_gibbs = xx_observed
        else:
            # sample from posterior of Xa
            Xa_gibbs = posterior_Xa_explicit(None, a_gibbs, b_gibbs, sigma_y_gibbs, yy, xx_observed, Uxx_inv)

        # sample from posterior of a
        a_gibbs = posterior_a_explicit(None, b_gibbs, Xa_gibbs, sigma_y_gibbs, yy, mu_a, sigma_a)

        ### TESTING marginalization over Xa
        #args = [sigma_y_gibbs, yy, xx_observed, Uxx_inv, b_gibbs, mu_a, sigma_a, 1.0]
        #posterior_pdf_a_without_Xa(1.0, *args)
        ### /TESTING

        # sample from posterior of b
        b_gibbs = posterior_b_explicit(None, a_gibbs, Xa_gibbs, sigma_y_gibbs, yy, mu_b, sigma_b)

        # sample from posterior of sigma_y
        if sigma_y_is_given:
            sigma_y_gibbs = sigma_y_true
        else:
            sigma_y_gibbs = posterior_sigma_y_explicit(None, a_gibbs, b_gibbs, Xa_gibbs, yy, mu_sigma_y, sigma_sigma_y)

        # 
        samples.append({
        "Xa" : Xa_gibbs,
        "a" : a_gibbs,
        "b" : b_gibbs,
        "sigma_y" : sigma_y_gibbs,
    })

    # estimate posterior from (avoid burn-in and take only every nth sample to avoid autocorrelation)

    considered_samples = samples[burn_in::use_every]
    AA = [sample["a"] for sample in considered_samples]
    BB = [sample["b"] for sample in considered_samples]
    SY = [sample["sigma_y"] for sample in considered_samples]

    if use_robust_statistics:
        posterior = {
            "a" : {
                "mu" : np.median(AA),
                "sigma" : iqr(AA),
            },
            "b" : {
                "mu" : np.median(BB),
                "sigma" : iqr(BB),
            },
            "sigma_y" : {
                "mu" : np.median(SY),
                "sigma" : iqr(SY),
            }
        }
    else:
        posterior = {
            "a" : {
                "mu" : np.mean(AA),
                "sigma" : np.std(AA),
            },
            "b" : {
                "mu" : np.mean(BB), 
                "sigma" : np.std(BB),
            },
            "sigma_y" : {
                "mu" : np.mean(SY),
                "sigma" : np.std(SY),
            }
        }

    # log for plot
    parameters_history[tt[-1]] = np.array([posterior["a"]["mu"], posterior["b"]["mu"], posterior["sigma_y"]["mu"]])
    parameter_uncertainty_history[tt[-1]] = np.array([posterior["a"]["sigma"], posterior["b"]["sigma"], posterior["sigma_y"]["sigma"]])

    # prepare next cycle
    prior = posterior


ax[0].plot(t, x_actual, label="ground truth")
ax[0].errorbar(t, x_observed, np.sqrt(np.diag(Ux)), label="reference signal")
ax[0].plot(t, y, label="dut indication")

#transfer_dut_calib_inverse = LinearAffineModel(**transfer_dut_calib.inverse_model_parameters())
#ax[0].errorbar(t, *transfer_dut_calib_inverse.apply(y_dut, np.zeros_like(y_dut)), label="dut compensated")

ax[0].legend()

#ax[1].plot(t, ux_ref)
# plot coefficient history
t_hist = np.array(list(parameters_history.keys()))

a_hist = np.array(list(parameters_history.values()))[:,0]
b_hist = np.array(list(parameters_history.values()))[:,1]
sigma_y_hist = np.array(list(parameters_history.values()))[:,2]

a_unc_hist = np.array(list(parameter_uncertainty_history.values()))[:,0]
b_unc_hist = np.array(list(parameter_uncertainty_history.values()))[:,1]
sigma_y_unc_hist = np.array(list(parameter_uncertainty_history.values()))[:,2]

ax[1].errorbar(t_hist, a_hist, a_unc_hist, label="a", c="b")
ax[1].hlines(a_true, t_hist.min(), t_hist.max(), colors="b", linestyle="dashed")

ax[1].errorbar(t_hist, b_hist, b_unc_hist, label="b", c="k")
ax[1].hlines(b_true, t_hist.min(), t_hist.max(), colors="k", linestyle="dashed")

ax[1].errorbar(t_hist, sigma_y_hist, sigma_y_unc_hist, label="sigma_y", c="g")
ax[1].hlines(sigma_y_true, t_hist.min(), t_hist.max(), colors="g", linestyle="dashed")

ax[2].plot(t_hist, a_unc_hist, "b")
ax[2].plot(t_hist, b_unc_hist, "k")
ax[2].plot(t_hist, sigma_y_unc_hist, "g")

ax[1].legend()

# store output information
if not os.path.exists("results"):
    os.makedirs("results")

now = datetime.datetime.isoformat(datetime.datetime.utcnow()).replace(":", "-")
basename = os.path.join("results", "run_{NOW}.{EXT}")

fig.set_size_inches(18.5, 10.5)
fig.savefig(basename.format(NOW=now, EXT="png"), dpi=100)

results = {
    "parameters_history" : parameters_history,
    "parameter_uncertainty_history": parameter_uncertainty_history,
    "signals" : {
        "t" : t,
        "x_actual" : x_actual,
        "x_observed" : x_observed,
        "y": y,
    },
    "model" : {
        "a_true" : a_true,
        "b_true" : b_true,
        "sigma_y_true" : sigma_y_true,
        "init_prior" : init_prior,
    },
    "gibbs_settings" : {
        "gibbs_runs" : gibbs_runs,
        "burn_in" : burn_in,
        "use_every" : use_every,
        "sigma_y_is_given" : sigma_y_is_given,
        "no_error_in_variables_model" : no_error_in_variables_model,
        "use_robust_statistics" : use_robust_statistics,
        "random_state" : rstate,
    },
}

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

f = open(basename.format(NOW=now, EXT="json"), "w")
json.dump(results, f, cls=NumpyEncoder, indent=4)
f.close()
