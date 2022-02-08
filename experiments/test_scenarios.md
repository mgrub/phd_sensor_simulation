# Test Scenarios to evaluate

## Static Noisy Signal

introduce static but noisy measurand (enable amplitude + phase + offset setting for SinusoidalMeasurand )


## Known Measurement Noise of DUT

SinusoidalMeasurand, init sigma_y to the correct value for all algorithms supporting it

## Number of reference sensors: 2, 10


## Dropouts (NaN) in sensor readings
--> maybe add a "dropout_rate" switch to sensor-class?

## Outliers (NaN) in sensor readings
--> maybe add a "outlier_rate" switch to sensor-class?


## reference signal performs better than DUT
unc of ref is smaller than unc of DUT, how?

## reference signal performs equally to DUT

## reference signal performs worse than DUT