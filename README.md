# Data-Science-Predictive-Model
Predicting taxi ride durations in New York City.  

The data from Kaggle including the training and test sets used can be found at 
https://www.kaggle.com/c/nyc-taxi-trip-duration/data

I started with two layers here, one with a multilayer perceptron, random forest, ada boost, and gradient boost regressors.  The second layer of xgboost takes as input the predictions of the previous layer.  I added some features like x distance, y distance, euclidian distance, manhattan distance, direction, starting neighbourhood, ending neighbourhood, and some relating the date and times, including whether the day of week was a weekday or weekend/holiday.  Also incorporated is a feature that includes the mean and median ride durations on the same day of week type, at the same hour, and from the same starting and ending neighbourhoods.  Morevover, I estimated the average velocity of similar rides (in terms of distance travelled and when), and used this estimate to add a feature which is the predicted duration of a trip based on these average velocities.

I noticed that AdaBoost and the MLP had fairly poor performance, so I removed them.  I also found that the feature importance of minute of departure was quite high.  I believe it shouldn't have been very important given that it doesn't provide all that much critical information, and removing that feature improved performance.  I also found that including the predictions of the first layer to all the other features, and passing all features to the xgboost layer was more beneficial than xgboost only having features that were from the first layer predictors.  I could improve performance further by adding more detailed neighbourhood information, as well as adding more complexity to the regressors.
