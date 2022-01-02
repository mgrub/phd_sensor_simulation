import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
import scipy.optimize as sco
from matplotlib import cm

mean_prior = np.array([1,2])
cov_prior = np.array([[2,0.0],[0.0,1]])
theta_dist = scs.multivariate_normal(mean = mean_prior, cov= cov_prior)

# signals
t = np.linspace(0, 10, 101)
x_true = np.sin(t) + 1
x_ref = x_true + 0.1 * np.random.randn(len(t))
y_ind = 1.5 * x_true + 1.8  + 0.2 * np.random.randn(len(t))

# MC
samples = theta_dist.rvs(size=10000)
samples_likelihood = []
for theta in samples:
    y_ind_model = theta[0] * x_ref + theta[1]

    p_X_theta = 1/np.linalg.norm(y_ind_model - y_ind)
    p_theta = theta_dist.pdf(theta)
    
    #samples_likelihood.append(p_X_theta * p_theta)  # prop to posterior
    samples_likelihood.append(p_X_theta)   # prop to likelihood

# estimate p(X|theta) from MC
samples_likelihood = np.array(samples_likelihood)
kernel = scs.gaussian_kde(np.array(samples).T, weights=samples_likelihood)
mean_likelihood = np.average(samples, weights=samples_likelihood, axis=0)
cov_likelihood = kernel.covariance
#cov_likelihood = np.cov(samples, rowvar=False)

# MLE of p(X|theta)
def evaluate(theta, x_ref, y_ind):
    y_ind_model = theta[0] * x_ref + theta[1]
    residual = np.linalg.norm(y_ind_model - y_ind)
    return residual

res = sco.minimize(evaluate, x0 = mean_prior, args=(x_ref, y_ind))
mean_MLE = res.x
cov_MLE = res.hess_inv


# plot time series
fig1, ax1 = plt.subplots(1, 1)
ax1.plot(t, x_true, label="x true")
ax1.plot(t, x_ref, label="x ref")
ax1.plot(t, y_ind, label="y ind")
ax1.set_title("input time series")
ax1.legend()


# plot prior and likelihood
fig2, ax2 = plt.subplots(1, 1)
# sample evaluation
ax2.scatter(samples[:,0], samples[:,1], c=samples_likelihood)
fig2.colorbar(cm.ScalarMappable(), ax=ax2)

# circle
r = np.linspace(0,2*np.pi,100)
circle = np.array([[np.cos(_r), np.sin(_r)] for _r in r]).T

# estimated prior distribution
ell_prior =  (cov_prior@circle).T + mean_prior
ax2.plot(ell_prior[:,0], ell_prior[:,1], ":b", label="p(theta|alpha)")

# estimated posterior distribution
ell_likelihood =  (cov_likelihood@circle).T + mean_likelihood
ax2.plot(ell_likelihood[:,0], ell_likelihood[:,1], ":r", label="p(X|theta) (MC + gaussian kde)")

# maximum likelihood optimiziation distribution
ell_MLE =  (cov_MLE@circle).T + mean_MLE
ax2.plot(ell_MLE[:,0], ell_MLE[:,1], ":g", label="p(X|theta) (MLE via hess_inv of opt)")


ax2.set_aspect("equal")
ax2.legend()
ax2.set_title("visualize prior and likelihood")
ax2.set_xlabel("a = theta[0]")
ax2.set_ylabel("b = theta[1]")


plt.show()