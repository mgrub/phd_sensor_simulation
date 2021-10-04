import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt

# # evaluation function for optimization
def evaluate(theta, y_dut, x_ref):

    # apply model
    y_dut_model = theta[0]*x_ref + theta[1]

    # calculate metric
    residual = np.linalg.norm(y_dut - y_dut_model)

    return residual

for Nt in np.int64(np.floor(np.logspace(1,5,25))):
    # inits
    t = np.linspace(0, 20, Nt)
    x_true = np.sin(t) + 2

    y_ref = 2.0 * x_true + 0.5
    y_ref += 0.1 * np.random.randn(len(y_ref))

    x_ref = 0.5 * (y_ref - 0.5)

    y_dut = 1.7 * x_true + 0.7
    y_dut += 0.2 * np.random.randn(len(y_ref))


    X = []
    X_hess = []
    for i_run in range(1000):
        x0_mc = np.array([1.0, 0.0]) 
        x0_mc += 0.2*np.random.randn(2)

        x_ref_mc = x_ref + 0.1*np.random.randn(len(x_ref))

        res = sco.minimize(evaluate, x0=x0_mc, args=(y_dut, x_ref_mc))
        #print(res.hess_inv)
        #print(x0_mc, res.x)
        
        X.append(res.x)
        X_hess.append(res.hess_inv)

    x_mc = np.mean(X, axis=0)
    ux_mc = np.cov(X, rowvar=False)

    print(x_mc)

    a = np.mean(X_hess, axis=0)
    plt.scatter(Nt, np.mean(a / ux_mc))
    #print(ux_mc)
    #print(a)
    #print(np.linalg.inv(a))
    print("====")

plt.show()

# visualize
plt.plot(t, x_true, label="x_true")
plt.plot(t, x_ref, label="x_ref")
plt.plot(t, y_dut, label="y_dut")
plt.plot(t, (y_dut - x_mc[1]) / x_mc[0], label="x_dut")
plt.legend()
plt.show()