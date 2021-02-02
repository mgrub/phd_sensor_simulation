# How to measure performance of the Stankovic method?

It is of interest to develop a metric that allows to compare the performance of the consensus-based calibration with other methods of same purpose.

How well does the algorithm converge to the true model parameters. Scenarios and measures might be:

- robustness
  - sensor dropouts
  - noisy input signals
- complex input signals
- simple input signals (e.g. stationary)
- non-equidistant time
- duration / computation-costs until consensus is reached

compare

- uncertainty quantification against Monte Carlo


TODO: separate scenarios from measures