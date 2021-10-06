import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco

from models import LinearAffineModel


t = np.linspace(0, 10, 101)
y_true = np.sin(t) + 1


# reference measurement simulation
transfer_ref = LinearAffineModel(a=2.0, b=2.4, ua=0.05, ub=0.05, uab=0.005)
transfer_ref_inverse = LinearAffineModel(**transfer_ref.inverse_model_parameters())

## simulate data acquisition for reference
x_ref, ux_ref = transfer_ref.apply(y_true, np.zeros_like(y_true))
x_ref += ux_ref * np.random.randn(len(x_ref))  # add corresponding noise

## simulate compensation of reference data
y_ref, uy_ref = transfer_ref_inverse.apply(x_ref, ux_ref)


# device-under-test (dut) measurement simulation
transfer_dut = LinearAffineModel(a=1.7, b=0.7, ua=0.2, ub=0.2, uab=0.01)   # this is unknown later
x_dut, ux_dut = transfer_dut.apply(y_true, np.zeros_like(y_true))
x_dut += ux_dut * np.random.randn(len(x_dut))  # add corresponding noise

# initial guess for dut-transfer behavior (theta ~ N(theta0, Sigma0)
theta0 = {"a": 1, "b": 0, "ua": 1, "ub" : 1, "uab" : 0}
non_negative_unc = ((None, None), (None, None), (0, None), (0, None), (None, None))
Sigma0 = np.diag((100, 100, 100, 100, 100))
transfer_dut_calib = LinearAffineModel(**theta0)

# evaluation function for optimization
def evaluate(theta, x_dut, y_ref, uy_ref):
    # init model
    model = LinearAffineModel(a=theta[0], b=theta[1], ua=theta[2], ub=theta[3], uab=theta[4])

    # apply model
    x_dut_model, ux_dut_model = model.apply(y_ref, uy_ref)

    # get uncertainty of inverse
    #inverse_model = LinearAffineModel(**transfer_dut_calib.inverse_model_parameters())
    #y_dut_model, uy_dut_model = inverse_model.apply(x_dut_model, ux_dut_model)

    # calculate residual
    residual = np.linalg.norm(x_dut - x_dut_model)  # mismatch of indicated values
    #residual += np.linalg.norm(uy_ref - uy_dut_model)  # mismatch of indicated uncertainties?

    return residual


# prepare some plotting
fig, ax = plt.subplots(3, 1)

# random splitpoints
n_splits = np.random.randint(1, len(t) // 3)
split_indices = np.sort(np.random.permutation(np.arange(3, len(t)-1))[:n_splits])

# iterate over splitpoints
parameters_history = {}
parameter_uncertainty_history = {}
S3 = Sigma0
for current_indices in np.split(np.arange(len(t)), split_indices):
    
    # available measurement information
    tt = t[current_indices]
    print(tt[0])
    yy_ref = y_ref[current_indices]
    uyy_ref = uy_ref[current_indices]
    xx_dut = x_dut[current_indices]

    # plt
    ax[0].scatter(tt, xx_dut)

    # params distribution (given by hyperparameters mu1 and S1 with theta ~ N(mu1, S1) )
    mu1 = list(transfer_dut_calib.get_params().values())   # [a, b, ua, ub, uab]
    S1 = S3

    # estimate with Monte Carlo
    THETA = []
    for i_run in range(100):
        theta0_mc = np.random.multivariate_normal(mu1, S1)  # initial guess for theta
        yy_ref_mc = yy_ref + uyy_ref*np.random.randn(len(yy_ref))  # reference measurand value (noisy)

        res = sco.minimize(evaluate, x0=theta0_mc, args=(xx_dut, yy_ref_mc, uyy_ref), bounds=non_negative_unc)
        #res = sco.shgo(evaluate, args=(xx_dut, yy_ref_mc, uyy_ref), bounds=non_negative_unc)
        #res = sco.dual_annealing(evaluate, x0=theta0_mc, args=(xx_dut, yy_ref_mc, uyy_ref), bounds=non_negative_unc)
        
        THETA.append(res.x)

    # estimate distribution of maximum likelihood based on MC results
    mu2 = np.mean(THETA, axis=0)
    S2 = np.cov(THETA, rowvar=False)
    print(mu2)

    # update prior param knowledge
    # https://math.stackexchange.com/questions/157172/product-of-two-multivariate-gaussians-distributions  -> check this (there is also a hint on numerical stability)
    S_inv = np.linalg.inv(S1 + S2)    
    mu3 = S2@S_inv@mu1 + S1@S_inv@mu2
    S3 = S1@S_inv@S2
    
    # update model parameters
    transfer_dut_calib.parameters = mu3[:2]
    transfer_dut_calib.parameters_uncertainty = np.array(
            [[np.square(mu3[2]), mu3[4]], [mu3[4], np.square(mu3[3])]]
        )

    # log
    parameters_history[tt[-1]] = mu3
    parameter_uncertainty_history[tt[-1]] = np.sqrt(np.diag(S3))


ax[0].plot(t, y_true, label="quantity")
ax[0].errorbar(t, x_ref, ux_ref, label="ref indication")
ax[0].errorbar(t, y_ref, uy_ref, label="ref measurand estimate")

ax[0].plot(t, x_dut, label="dut indication")
transfer_dut_calib_inverse = LinearAffineModel(**transfer_dut_calib.inverse_model_parameters())
ax[0].errorbar(t, *transfer_dut_calib_inverse.apply(x_dut, np.zeros_like(x_dut)), label="dut measurand estimate")
ax[0].legend()

# plot coefficient history
t_hist = np.array(list(parameters_history.keys()))
params_hist = np.array(list(parameters_history.values()))
params_unc_hist = np.array(list(parameter_uncertainty_history.values()))
labels_hist = ["a", "b", "ua", "ub", "uab"]

for p_hist, up_hist, label in zip(params_hist.T, params_unc_hist.T, labels_hist):
    ax[1].errorbar(t_hist, p_hist, up_hist, label=label)
    ax[2].plot(t_hist, up_hist, label=label)

ax[1].legend()

plt.show()
