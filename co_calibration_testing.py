import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco

from models import LinearAffineModel


t = np.linspace(0, 10, 100)
x_true = np.sin(t) + 1


# reference measurement simulation
transfer_ref = LinearAffineModel(a=2.0, b=0.4, ua=0.05, ub=0.05, uab=0.005)
transfer_ref_inverse = LinearAffineModel(**transfer_ref.inverse_model_parameters())

## simulate data acquisition for reference
y_ref, uy_ref = transfer_ref.apply(x_true, np.zeros_like(x_true))
y_ref += uy_ref * np.random.randn(len(y_ref))  # add corresponding noise

## simulate compensation of reference data
x_ref, ux_ref = transfer_ref_inverse.apply(y_ref, uy_ref)


# device-under-test (dut) measurement simulation
transfer_dut = LinearAffineModel(a=1.7, b=0.7, ua=0.2, ub=0.2, uab=0.01)   # this is unknown later
y_dut, uy_dut = transfer_dut.apply(x_true, np.zeros_like(x_true))
y_dut += uy_dut * np.random.randn(len(y_dut))  # add corresponding noise
transfer_dut_calib = LinearAffineModel(ua=10000, ub=10000, uab=0)

# evaluation function for optimization
def evaluate(theta, y_dut, x_ref):
    # init model
    model = LinearAffineModel(a=theta[0], b=theta[1])

    # apply model
    y_dut_model, _ = model.apply(x_ref, np.zeros_like(x_ref))

    # calculate metric
    residual = np.linalg.norm(y_dut - y_dut_model)

    return residual


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
    xx_ref = x_ref[current_indices]
    uxx_ref = ux_ref[current_indices]
    yy_ref = y_ref[current_indices]
    uyy_ref = uy_ref[current_indices]
    yy_dut = y_dut[current_indices]

    # plt
    ax[0].scatter(tt, yy_dut)

    # estimate params from current data
    #res = sco.minimize(evaluate, x0=transfer_dut_calib.parameters, args=(yy_dut, xx_ref))

    # estimate params distribution
    mu1 = transfer_dut_calib.parameters
    S1 = transfer_dut_calib.parameters_uncertainty
    #mu2 = res.x
    #S2 = res.hess_inv / (10 * np.sqrt(len(tt)))  # currently heuristic correction factor

    # estimate with Monte Carlo
    X = []
    X_hess = []
    for i_run in range(100):
        x0_mc = np.random.multivariate_normal(transfer_dut_calib.parameters, transfer_dut_calib.parameters_uncertainty)
        xx_ref_mc = xx_ref + uxx_ref*np.random.randn(len(xx_ref))

        res = sco.minimize(evaluate, x0=x0_mc, args=(yy_dut, xx_ref_mc))
        
        X.append(res.x)
        X_hess.append(res.hess_inv)

    mu2 = np.mean(X, axis=0)
    S2 = np.cov(X, rowvar=False)

    # update prior param knowledge
    # https://math.stackexchange.com/questions/157172/product-of-two-multivariate-gaussians-distributions  -> check this (there is also a hint on numerical stability)
    S_inv = np.linalg.inv(S1 + S2)    
    mu3 = S2@S_inv@mu1 + S1@S_inv@mu2
    S3 = S1@S_inv@S2
    
    # log
    parameters_history[tt[-1]] = mu3
    parameter_uncertainty_history[tt[-1]] = np.sqrt(np.diag(S3))

    transfer_dut_calib.parameters = mu3
    transfer_dut_calib.parameters_uncertainty = S3


ax[0].plot(t, x_true, label="quantity")
ax[0].errorbar(t, y_ref, uy_ref, label="ref measured")
ax[0].errorbar(t, x_ref, ux_ref, label="ref compensated")

ax[0].plot(t, y_dut, label="dut measured")
transfer_dut_calib_inverse = LinearAffineModel(**transfer_dut_calib.inverse_model_parameters())
ax[0].errorbar(t, *transfer_dut_calib_inverse.apply(y_dut, np.zeros_like(y_dut)), label="dut compensated")

ax[0].legend()

#ax[1].plot(t, ux_ref)
# plot coefficient history
t_hist = np.array(list(parameters_history.keys()))
a_hist = np.array(list(parameters_history.values()))[:,0]
b_hist = np.array(list(parameters_history.values()))[:,1]
a_unc_hist = np.array(list(parameter_uncertainty_history.values()))[:,0]
b_unc_hist = np.array(list(parameter_uncertainty_history.values()))[:,1]
ax[1].errorbar(t_hist, a_hist, a_unc_hist, label="a", c="b")
ax[1].errorbar(t_hist, b_hist, b_unc_hist, label="b", c="k")
ax[2].plot(t_hist, a_unc_hist)
ax[2].plot(t_hist, b_unc_hist)

plt.show()
