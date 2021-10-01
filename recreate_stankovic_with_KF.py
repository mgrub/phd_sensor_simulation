import numpy as np
import matplotlib.pyplot as plt

# inits
a_true = 3.0
b_true = 1.2
z = np.array([0.5,0.5]).T
P = np.diag([1.0, 1.0])
Qk = np.eye(z.size)
Rk = np.eye(2)


t = np.linspace(0, 10, 100)
x = np.sin(t) + 0.3 * np.random.randn(len(t))

z_history = np.empty((len(t), 2))
y_history = np.empty((len(t), 1))

for i, ts in enumerate(t):
    print(ts)

    # generate reference value
    x = np.sin(ts) + 0.1 * np.random.randn()
    
    # generate value from device under test using the "true" parameters
    y = a_true * x + b_true + + 0.2 * np.random.randn()

    # what we currently have in our system
    yk = z[0] * x + z[1]
    Fk = np.eye(2)
    Gk = np.array([x, 1])

    # time update
    z = z
    P = P + Qk

    # measurement update
    Kk = P@Gk.T@np.linalg.inv(Gk@P@Gk.T + Rk)
    z = z + Kk*(y - yk)    # TODO
    P = (np.eye(2) - Kk@Gk)@P