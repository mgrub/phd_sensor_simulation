import numpy as np
import matplotlib.pyplot as plt

from models import LinearAffineModel


t = np.linspace(0, 10, 100)
x_true = np.sin(t)


# reference measurement simulation
transfer_ref = LinearAffineModel(a=2.0, b=0.4, ua=0.2, ub=0.2, uab=0.01)
transfer_ref_inverse = LinearAffineModel(**transfer_ref.inverse_model_parameters())

## simulate data acquisition for reference
y_ref, uy_ref = transfer_ref.apply(x_true, np.zeros_like(x_true))
y_ref += uy_ref * np.random.randn(len(y_ref))  # add corresponding noise

## simulate compensation of reference data
x_ref, ux_ref = transfer_ref_inverse.apply(y_ref, uy_ref)


# device-under-test (dut) measurement simulation
transfer_dut = LinearAffineModel(a=1.7, b=0.7, ua=0.4, ub=0.4, uab=0.02)   # this is unknown later
y_dut, uy_dut = transfer_dut.apply(x_true, np.zeros_like(x_true))
y_dut += uy_dut * np.random.randn(len(y_dut))  # add corresponding noise


# prepare some plotting
fig, ax = plt.subplots(2, 1)

# random splitpoints
n_splits = np.random.randint(1, len(t) // 3)
split_indices = np.sort(np.random.permutation(np.arange(3, len(t)-1))[:n_splits])

# iterate over splitpoints
for current_indices in np.split(np.arange(len(t)), split_indices):
    
    # available measurement information
    tt = t[current_indices]
    yy_ref = y_ref[current_indices]
    uyy_ref = uy_ref[current_indices]
    yy_dut = y_dut[current_indices]

    # plt
    ax[0].scatter(tt, yy_dut)

    # TODO: estimate params from current data
    


    # TODO: estimate params distribution


    # TODO: update prior param knowledge


ax[0].plot(t, x_true, label="quantity")
ax[0].errorbar(t, y_ref, uy_ref, label="ref measured")
ax[0].errorbar(t, x_ref, ux_ref, label="ref compensated")
ax[0].plot(t, y_dut, label="dut measured")
ax[0].legend()

ax[1].plot(t, ux_ref)
plt.show()
