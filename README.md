# Data Challenge: Wind Power forecasting for the day-ahead energy market
by Compagnie Nationale du Rhône

https://challengedata.ens.fr/challenges/34

**[on going work]**

--> Time-series prediction with 1h time step.

Train set:
* six wind farms (WF) hourly production data from May the 1st of 2018 to January the 15th of 2019 (8 months and 15 days)
* hourly forecasted meteorological variables, provided by several Numerical Weather Prediction (NWP) models

Prediction target:
* day-ahead WF hourly production of the six WF

Constraint:
* day-ahead predictions must be computed with data that are available in real operational conditions on day D at 09h00 UTC

Test set: 
* Data from January the 16th of 2019 to September the 30rd of 2019 (8 months and 15 days)

Complementary observed data (shouldn't be used directly in the model):
* hourly wind speeds and wind directions observed at the height of each wind turbine

NWP models:
* Hours:
* U & V components of the wind : 
    * NWP1, NWP2, NWP3: 100m height
    * NWP4: 10m height
    
NWP Variable | Prediction description | NWP 1 | NWP 2 | NWP 3 | NWP 4 
------ | --- | --- | --- | --- | ---
Wind speed U,V (m/s) | 10min average [H-10min,H] | x (100m) | x (100m) | x (100m) | x (10m)
Temperature of air T (m/s) | 1hour average [H-1,H] | x |  | x | 
Total cloud cover CLCT (%) | instant value at H | | | | x

Tips: 
* Train 1 model per WF
* derive wind speed (and wind direction) from U and V
* reconstruct NWP time series

Steps: 
* Make sequences from data
* Make a model
* Train it with the sequences
* Cross-validation
* Try several models