# Data-Science-Predictive-Model
Predicting taxi ride durations in New York City.  

The data from Kaggle including the training and test sets used can be found at 
https://www.kaggle.com/c/nyc-taxi-trip-duration/data

I have two layers here, one with a multilayer perceptron, random forest, ada boost, and gradient boost regressors.  The second layer of xgboost takes as input the predictions of the previous layer.  I added some features like x distance, y distance, euclidian distance, direction, starting neighbourhood, ending neighbourhood, and some relating the date and times, including whether the day of week was a weekday or weekend/holiday.  Commented out is a feature that includes the mean and median ride durations on the same day of week type, at the same hour, and from the same starting and ending neighbourhoods.  It's pretty computationally intensive though and ends up hogging a lot of RAM, so it might not be the best idea to bother with it when running for fun.
