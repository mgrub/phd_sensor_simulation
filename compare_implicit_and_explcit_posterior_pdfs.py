from co_calibration_gibbs_sampler_expressions import *
import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

import matplotlib.pyplot as plt

# init priors
mu_a = 1.0
sigma_a = 0.5
mu_b = 2.0
sigma_b = 0.5
mu_sigma_y = 0.1
sigma_sigma_y = 0.01

# define init values to enable calculations
N = np.random.randint(3, 10)
a = np.random.normal(mu_a, sigma_a)
b = np.random.normal(mu_b, sigma_b)
Xa = np.random.random(size=N)
sigma_y = np.random.normal(mu_sigma_y, sigma_sigma_y)

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
normalizer = quad(posterior_a_implicit, -np.inf, np.inf, args=tuple(args), epsabs=1e-40)[0]
#target_quantile = np.random.random()
#args[-1] = normalizer
#bracket = (a - sigma_a, a + sigma_a)
#evaluate = lambda x: norm(target_quantile - quad(posterior_a_implicit, -np.inf, x, args=tuple(args))[0])
#res = minimize_scalar(evaluate, bracket=bracket)
#print(res.x)

x_plot = np.linspace(-5, 5, 100)
pdf_implicit = np.array([posterior_a_implicit(xx, *args) for xx in x_plot])
pdf_explicit = np.array([posterior_a_explicit(xx, *args) for xx in x_plot])

plt.plot(x_plot, pdf_implicit, label="implicit")
plt.plot(x_plot, pdf_explicit, label="explicit")
plt.legend()

plt.show()