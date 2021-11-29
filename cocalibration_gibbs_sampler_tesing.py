import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from scipy.stats import iqr

from models import LinearAffineModel

# actual ground truth
t = np.linspace(0, 10, 101)
x_actual = np.sin(t) + 1

# reference values
Ux = np.diag(np.full_like(x_actual, 0.01))
x_observed = np.random.multivariate_normal(mean=x_actual, cov=Ux)

# indication of DUT
a_true = 2.0
b_true = 2.5
true_transfer_dut = LinearAffineModel(a=a_true, b=b_true)
y = true_transfer_dut.apply(x_actual, np.zeros_like(x_actual))[0] + np.random.normal(scale=0.2, size=len(x_actual))

# special cases (activate by setting to True)
sigma_y_is_given = True
no_error_in_variables_model = False

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

# pdf of sigma_y (up to proportionality constant)
def posterior_pdf_sigma_y(sigma_y, Y, Xa, a, b, mu_sigma_y, sigma_sigma_y, normalizer=1.0):
    div = 2*sigma_sigma_y**2
    A_tilde = 0.5 * np.sum(np.square(Y - a*Xa - b))
    exponent = - sigma_y**2 / div - sigma_y * mu_sigma_y / div - sigma_y ** (-2) * A_tilde
        
    return np.exp(exponent) / normalizer

# gibbs sampler settings
gibbs_runs = 1000
burn_in = gibbs_runs // 10
use_every = 50


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
            F1 = np.diag(np.full_like(xx_observed, a_gibbs**2 / sigma_y_gibbs**2))
            F2 = a_gibbs / sigma_y_gibbs**2 * (b_gibbs - yy)
            V = F1 + Uxx_inv
            V_inv = np.linalg.inv(V)
            M = (Uxx_inv@xx_observed - F2)@V_inv
            Xa_gibbs = np.random.multivariate_normal(M, V_inv)

        # sample from posterior of a
        A_a = - np.sum(np.square(Xa_gibbs) / (2*sigma_y_gibbs**2)) - 1.0/(2*sigma_a**2) 
        B_a = np.sum((b_gibbs-yy)*Xa_gibbs / (2*sigma_y_gibbs**2)) - mu_a/(2*sigma_a**2)
        a_gibbs = np.random.normal(B_a/A_a, np.sqrt(-1/(2*A_a)))

        # sample from posterior of b
        A_b = - Xa_gibbs.size / (2*sigma_y_gibbs**2) - 1.0/(2*sigma_b**2)
        B_b = np.sum((a_gibbs * Xa_gibbs - yy) / (2*sigma_y_gibbs**2)) - mu_b/(2*sigma_b**2) 
        b_gibbs = np.random.normal(B_b/A_b, np.sqrt(-1/(2*A_b)))

        # sample from posterior of sigma_y
        if sigma_y_is_given:
            sigma_y_gibbs = 0.1
        else:
            ### THIS IS CURRENTLY UNDER TESTING ###
            args = [yy, Xa_gibbs, a_gibbs, b_gibbs, mu_sigma_y, sigma_sigma_y, 1.0]
            normalizer = quad(posterior_pdf_sigma_y, -np.inf, np.inf, args=tuple(args))[0]
            target_quantile = np.random.random()
            args[-1] = normalizer
            evaluate = lambda x: norm(target_quantile - quad(posterior_pdf_sigma_y, -np.inf, x, args=tuple(args))[0])
            res = minimize_scalar(evaluate, bracket=(sigma_y_gibbs - sigma_sigma_y, sigma_y_gibbs + sigma_sigma_y))
            print(res.x)
            sigma_y_gibbs = res.x 

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
a_unc_hist = np.array(list(parameter_uncertainty_history.values()))[:,0]
b_unc_hist = np.array(list(parameter_uncertainty_history.values()))[:,1]
ax[1].errorbar(t_hist, a_hist, a_unc_hist, label="a", c="b")
ax[1].hlines(a_true, t_hist.min(), t_hist.max(), colors="b", linestyle="dashed")
ax[1].errorbar(t_hist, b_hist, b_unc_hist, label="b", c="k")
ax[1].hlines(b_true, t_hist.min(), t_hist.max(), colors="k", linestyle="dashed")
ax[2].plot(t_hist, a_unc_hist, "b")
ax[2].plot(t_hist, b_unc_hist, "k")
ax[1].legend()

plt.show()
