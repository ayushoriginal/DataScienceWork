## [Taxi Trips](https://github.com/ayushoriginal/DataScienceWork/tree/master/TaxiTrip)
### AIM: Predict the total ride duration of taxis

The goal is to construct a model that can predict the cumulative ride duration of taxi trips in a city. I've taken the dataset released by the NYC Taxi and Limousine Commission, which includes 
-pickup time 
-geo-coordinates 
-number of passengers
 and several other variables.

### First part - Data exploration
The first part is to analyze the dataframe and observe correlation between variables.
![Distributions](https://github.com/ayushoriginal/DataScienceWork/blob/master/TaxiTrip/pic/download.png)
![Rush Hour](https://github.com/ayushoriginal/DataScienceWork/blob/master/TaxiTrip/pic/rush_hour.png)

### Second part - Clustering
The goal of this playground is to predict the trip duration of test set. We know that some neighborhoods are more congested. So, I used K-Means to compute geo-clusters for pickup and drop off.
![Cluster](https://github.com/ayushoriginal/DataScienceWork/blob/master/TaxiTrip/pic/nyc_clusters.png)

### Third part - Cleaning and feature selection 
I have found some odd long trips : one day trip with a mean spead < 1km/h.   
![Outliners](https://github.com/ayushoriginal/DataScienceWork/blob/master/TaxiTrip/pic/outliners.png)
I have removed these outliners.  

I also added features from the data available : Haversine distance, Manhattan distance, means for clusters, PCA for rotation.

### Forth part - Prediction
I compared Random Forest and XGBoost.  
Current Root Mean Squared Logarithmic error : 0.391

Feature importance for RF & XGBoost
![Feature importance](https://github.com/ayushoriginal/DataScienceWork/blob/master/TaxiTrip/pic/feat_importance.png)
