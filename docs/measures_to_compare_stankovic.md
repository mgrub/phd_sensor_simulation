# How to measure performance of the consensus based parameter estimation methods?

It is of interest to develop a metric that allows to compare the performance of the consensus-based calibration with other methods of same purpose.

How well does the algorithm converge to the true model parameters. Scenarios and measures might be:

Input signal scenarios (sorted by robustness requirements):

- I1: stationary, noiseless, equidistant, no dropouts
- I2: non-stationary (sine sweeps, steps, ramps), noiseless, equidistant, no dropouts
- I3: non-stationary (sine sweeps, steps, ramps), noisy, non-equidistant, no dropouts
- I4: non-stationary (sine sweeps, steps, ramps), noisy, non-equidistant, with dropouts

Transfer model behavior scenarios:

- T1: linear model (y = a*x)
- T2: linear affine model (y = a*x + b)
- T3: second order model
- T4: frequency response / IIR?

Comparison measures:

- M1: convergence of estimated parameters after ... iterations/timesteps? (theta_est(n) - theta_est(n-10?))
- M2: consistency of estimated parameters after ... iterations/timesteps? (theta_est - theta_true)
- M3: runtime / computations after ... iterations/timesteps?

The above combinations of scenarios and measures shall be applied to T2 and:

- base Stankovic method (no uncertainty)
- base Stankovic method with Monte-Carlo (with uncertainty)
- extension of Stankovic method with default weights (with uncertainty)
- extension of Stankovic method with weights according to GUM-uncertainty (with uncertainty)
- new method to be developed within PhD (with uncertainty)

Based on the provided measures in different scenarios, statements about the robustness and applicability of each method shall be made. Moreover, the uncertainty quantification of the "base Stankovic with MC" and "extension of Stankovic" shall be compared. Moreover, the "new method to be developed" shall be tested against all transfer model behavior scenarios (T1, T2, ...). 