import numpy as np
import matplotlib.pyplot as plt

# inits
a_true = 3.0
b_true = 1.2
z = np.array([2.9,1.1]).T
P = 0.1**2 * np.eye(z.size)
Qk = 0.1**2 * np.eye(z.size)
Rk = 0.2**2 * np.eye(2)


t = np.linspace(0, 10, 101)
x = np.sin(t) + 0.3 * np.random.randn(len(t))

z_history = np.empty((len(t), 2))
y_history = np.empty((len(t), 1))

for i, ts in enumerate(t):

    # generate reference value
    x = np.sin(ts) + 0.1 * np.random.randn()
    
    # generate value from device under test using the "true" parameters
    yk = a_true * x + b_true + + 0.2 * np.random.randn()

    # what we currently have in our system
    y = z[0] * x + z[1]
    Fk = np.eye(2)
    Gk = np.array([x, 1])

    # time update
    z = z
    P = P + Qk

    # measurement update
    Kk = P@Gk.T@np.linalg.pinv(Gk@P@Gk.T + Rk)
    z = z + Kk*(yk - y)    # TODO
    P = (np.eye(2) - Kk@Gk)@P

    print(ts, z)