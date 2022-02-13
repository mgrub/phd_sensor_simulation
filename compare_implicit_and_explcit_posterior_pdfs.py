import copy

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.stats import invgamma

from co_calibration_gibbs_sampler_expressions import *

# init priors
mu_a = 1.0
sigma_a = 0.5
mu_b = 2.0
sigma_b = 0.5
shape_sigma_y = 2
scale_sigma_y = 1
loc_sigma_y = 0.0

# define init values to enable calculations
N = np.random.randint(3, 10)
a = np.random.normal(mu_a, sigma_a)
b = np.random.normal(mu_b, sigma_b)
Xa = np.random.random(size=N)
sigma_y = invgamma.rvs(a=shape_sigma_y, scale=scale_sigma_y, loc=loc_sigma_y)

UXo = np.diag((1 + np.random.random(N)) * 0.01)
UXo_inv = np.linalg.inv(UXo)
UY = np.diag(np.square(np.full((N,), sigma_y**2)))

Y = np.random.multivariate_normal(a * Xa + b, UY)
Xo = Y = np.random.multivariate_normal(Xa, UXo)


for item in [N, a, b, sigma_y, Xa, Xo, UXo, Y]:
    print(item)
print("="*30)

# compare implicit and explicit posterior of a
args = [b, Xa, sigma_y, Y, mu_a, sigma_a, 1.0]
normalizer = quad(posterior_a_implicit, -np.inf, np.inf, args=tuple(args), epsrel=-1)[0]
args[-1] = normalizer

x_plot = np.linspace(-5, 5, 200)
pdf_a_implicit = np.array([posterior_a_implicit(xx, *args) for xx in x_plot])
pdf_a_explicit = np.array([posterior_a_explicit(xx, *args) for xx in x_plot])

plt.plot(x_plot, pdf_a_implicit, label="implicit")
plt.plot(x_plot, pdf_a_explicit, label="explicit")
plt.legend()
plt.title("a posterior")
plt.show()


# compare implicit and explicit posterior of b
args = [a, Xa, sigma_y, Y, mu_b, sigma_b, 1.0]
normalizer = quad(posterior_b_implicit, -np.inf, np.inf, args=tuple(args), epsrel=-1)[0]
args[-1] = normalizer

x_plot = np.linspace(-5, 5, 200)
pdf_b_implicit = np.array([posterior_b_implicit(xx, *args) for xx in x_plot])
pdf_b_explicit = np.array([posterior_b_explicit(xx, *args) for xx in x_plot])

plt.plot(x_plot, pdf_b_implicit, label="implicit")
plt.plot(x_plot, pdf_b_explicit, label="explicit")
plt.legend()
plt.title("b posterior")
plt.show()


# compare implicit and explicit posterior of sigma_y
args = [a, b, Xa, Y, shape_sigma_y, scale_sigma_y, loc_sigma_y, 1.0]
normalizer_imp = quad(posterior_sigma_y_implicit, loc_sigma_y, np.inf, args=tuple(args), epsrel=-1)[0]
normalizer_exp = quad(posterior_sigma_y_explicit, loc_sigma_y, np.inf, args=tuple(args), epsrel=-1)[0]
args_imp = copy.deepcopy(args)
args_exp = copy.deepcopy(args)
args_imp[-1] = normalizer_imp
args_exp[-1] = normalizer_exp

x_plot = np.linspace(0.2, 2.0, 400)
pdf_sigma_y_implicit = np.array([posterior_sigma_y_implicit(xx, *args_imp) for xx in x_plot])
pdf_sigma_y_explicit = np.array([posterior_sigma_y_explicit(xx, *args_exp) for xx in x_plot])

plt.plot(x_plot, pdf_sigma_y_implicit, label="implicit")
plt.plot(x_plot, pdf_sigma_y_explicit, label="explicit")
plt.legend()
plt.title("sigma_y posterior")
plt.show()
