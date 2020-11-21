# ENS Data Challenge: Wind Power forecasting for the day-ahead energy market
by Compagnie Nationale du Rhône

<p align="center"><img src="https://cap.img.pmdstatic.net/fit/http.3A.2F.2Fprd2-bone-image.2Es3-website-eu-west-1.2Eamazonaws.2Ecom.2Fcap.2F2019.2F10.2F04.2Fea495374-9115-4be7-a91a-e9bc5b305b0b.2Ejpeg/768x432/background-color/ffffff/focus-point/992%2C1086/quality/70/dangereuses-pour-la-sante-peu-ecolo-faut-il-en-finir-avec-les-eoliennes-1352031.jpg" width="600"/></p>

# Problem

Predict the hourly production of 6 independent wind farms for the day ahead, using recent production data and weather predictions of different national models.

https://challengedata.ens.fr/challenges/34

Time-series prediction with 1h time step.

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

NWP Variable | Prediction description | NWP 1 (hourly) | NWP 2 (every 3 hours) | NWP 3 (every 3 hours) | NWP 4 (hourly)
------ | ----- | ----- | ----- | ----- | -----
Wind speed U,V (m/s) | 10min average [H-10min,H] | x (@100m) | x (@100m) | x (@100m) | x (@10m)
Temperature of air T (m/s) | 1hour average [H-1,H] | x |  | x |
Total cloud cover CLCT (%) | instant value at H | | | | x

Tips: 
* Train 1 model per WF
* derive wind speed (and wind direction) from U and V
* reconstruct NWP time series

# My approach
**[ongoing work]**
* Make a train set of sequences based on a "best weather prediction" matrix.

<p align="center"><img src="/schemes/sequence_structure.jpg" width="600"/></p>

* Make a model
* Train it with the sequences
* Cross-validation
* Try several models
