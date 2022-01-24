import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

# default order: a, b, Xa, sigma_y, Y, Xo, UXo_inv, XYZ, normalizer=1.0

### generic (sed for priors)
def gaussian(x, mu_x, sigma_x):
    return norm.pdf(x, loc=mu_x, scale=sigma_x)

def multivariate_gaussian(x, mu_x, cov_x):
    return multivariate_normal.pdf(x, mean=mu_x, cov=cov_x)



### likelihoods
def likelihood_Y(Y, a, b, Xa, sigma_y):
    exponent = - 1.0 / (2*sigma_y**2) * np.sum(np.square(Y - a*Xa - b)) 
    return np.exp(exponent) / (sigma_y**len(Y))

def likelihood_Xa(Xa, Xo, UXo_inv):
    DX = Xa - Xo
    exponent = -0.5 * DX.T @ UXo_inv @ DX
    return np.exp(exponent)


### posteriors implicit
def posterior_Xa_implicit(Xa, a, b, sigma_y, Y, Xo, UXo_inv, normalizer=1.0):
    expr1 = likelihood_Y(Y, a, b, Xa, sigma_y)
    expr2 = likelihood_Xa(Xa, Xo, UXo_inv)
    return expr1 * expr2 / normalizer

def posterior_a_implicit(a, b, Xa, sigma_y, Y, mu_a, sigma_a, normalizer=1.0):
    expr1 = likelihood_Y(Y, a, b, Xa, sigma_y)
    expr2 = gaussian(a, mu_a, sigma_a)
    return expr1 * expr2 / normalizer

def posterior_b_implicit(b, a, Xa, sigma_y, Y, mu_b, sigma_b, normalizer=1.0):
    expr1 = likelihood_Y(Y, a, b, Xa, sigma_y)
    expr2 = gaussian(b, mu_b, sigma_b)
    return expr1 * expr2 / normalizer

def posterior_sigma_y_implicit(sigma_y, a, b, Xa, Y, mu_sigma_y, sigma_sigma_y, normalizer=1.0):
    expr1 = likelihood_Y(Y, a, b, Xa, sigma_y)
    expr2 = gaussian(sigma_y, mu_sigma_y, sigma_sigma_y)
    return expr1 * expr2 / normalizer


### posteriors explicit (from calculations shown in docs)
def posterior_Xa_explicit(Xa, a, b, sigma_y, Y, Xo, UXo_inv, normalizer=1.0):
    # sample from posterior of Xa
    F1 = np.diag(np.full_like(Xo, a**2 / sigma_y**2))
    F2 = a / sigma_y**2 * (b - Y)
    V_inv = F1 + UXo_inv
    V = np.linalg.inv(V_inv)
    M = V@(UXo_inv@Xo - F2)

    if Xa == None:  # return a sample
        return np.random.multivariate_normal(M, V)
    else:  # return pdf at value Xa
        return multivariate_gaussian(Xa, M, V)

def posterior_a_explicit(a, b, Xa, sigma_y, Y, mu_a, sigma_a, normalizer=1.0):
    A_a = - np.sum(np.square(Xa) / (2*sigma_y**2)) - 1.0/(2*sigma_a**2) 
    B_a = np.sum((b-Y)*Xa / (2*sigma_y**2)) - mu_a/(2*sigma_a**2)

    if a == None:  # return a sample
        return np.random.normal(B_a/A_a, np.sqrt(-1/(2*A_a)))
    else:  # return pdf at value a
        return gaussian(a, B_a/A_a, np.sqrt(-1/(2*A_a)))

def posterior_b_explicit(b, a, Xa, sigma_y, Y, mu_b, sigma_b, normalizer=1.0):
    A_b = - Xa.size / (2*sigma_y**2) - 1.0/(2*sigma_b**2)
    B_b = np.sum((a * Xa - Y) / (2*sigma_y**2)) - mu_b/(2*sigma_b**2) 

    if b == None:  # return a sample
        return np.random.normal(B_b/A_b, np.sqrt(-1/(2*A_b)))
    else:  # return pdf at value b
        return gaussian(b, B_b/A_b, np.sqrt(-1/(2*A_b)))

def posterior_sigma_y_explicit(sigma_y, a, b, Xa, Y, mu_sigma_y, sigma_sigma_y, normalizer=1.0):
    div = sigma_sigma_y**2
    A_tilde = 0.5 * np.sum(np.square(Y - a*Xa - b))
    N = Xa.size
    args = [A_tilde, mu_sigma_y, div, N, normalizer]

    if sigma_y == None:  # return a sample
        normalizer = quad(posterior_sigma_y_explicit_faster, -np.inf, np.inf, args=tuple(args))[0]
        args[-1] = normalizer
        target_quantile = np.random.random()
        evaluate = lambda x: np.linalg.norm(target_quantile - quad(posterior_sigma_y_explicit_faster, -np.inf, x, args=tuple(args))[0])
        res = minimize_scalar(evaluate, bracket=(mu_sigma_y - sigma_sigma_y, mu_sigma_y + sigma_sigma_y))
        return res.x
    else:  # return pdf at value sigma_y
        return posterior_sigma_y_explicit_faster(sigma_y, *args)

def posterior_sigma_y_explicit_faster(sigma_y, A_tilde, mu_sigma_y, div, N, normalizer=1.0):
    if sigma_y == 0.0:
        return 0.0
    else:
        exponent = - sigma_y**2 / (2*div) + sigma_y * mu_sigma_y / div - sigma_y ** (-2) * A_tilde - N * np.log(np.abs(sigma_y))
        return np.exp(exponent) / normalizer

### joint distribution with Xa-marginalization
def likelidhood_a_b_sigma_y_without_Xa(a, b, sigma_y, Xo, UXo_inv, Y, normalizer=1.0):

    F1 = np.diag(np.full_like(Xo, a**2 / sigma_y**2))
    F2 = a / sigma_y**2 * (b - Y)

    V_inv = F1 + UXo_inv
    V = np.linalg.inv(V_inv)
    M = V@(UXo_inv@Xo - F2)

    W_inv = F1 + V_inv
    W = np.linalg.inv(W_inv)
    S = W@(V_inv@M - F2)

    exponent = - 0.5 * (M.T@V_inv@M - S.T@W_inv@S)
    det = np.sqrt(np.linalg.det(W))
        
    return det * np.exp(exponent) / normalizer