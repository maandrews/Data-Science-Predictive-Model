# Predictive model where we attempt to predict the time duration of a taxi ride in New York City.

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import mean_squared_error

# machine learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev2\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')


# Load .csv files
combine = [pd.read_csv('taxitrain.csv'), pd.read_csv('taxitest.csv')]


# Cleaning/Engineering etc.

# Dropping cols that are useless
combine[0] = combine[0].drop(['id'], axis=1)

combine[0] = combine[0].drop(['dropoff_datetime'], axis = 1)

# A lot of random outliers in this data.
allLat = np.array(list(combine[0]['pickup_latitude']) + list(combine[0]['dropoff_latitude']))
allLong = np.array(list(combine[0]['pickup_longitude']) + list(combine[0]['dropoff_longitude']))

longLimits = [np.percentile(allLong, 0.3), np.percentile(allLong, 99.7)]
latLimits = [np.percentile(allLat, 0.3), np.percentile(allLat , 99.7)]
tripLimits = [np.percentile(combine[0]['trip_duration'], 0.3), np.percentile(combine[0]['trip_duration'], 99.7)]

combine[0] = combine[0][(combine[0]['pickup_latitude'] >= latLimits[0])
                        & (combine[0]['pickup_latitude'] <= latLimits[1])]
combine[0] = combine[0][(combine[0]['dropoff_latitude'] >= latLimits[0])
                        & (combine[0]['dropoff_latitude'] <= latLimits[1])]
combine[0] = combine[0][(combine[0]['pickup_longitude'] >= longLimits[0])
                        & (combine[0]['pickup_longitude'] <= longLimits[1])]
combine[0] = combine[0][(combine[0]['dropoff_longitude'] >= longLimits[0])
                        & (combine[0]['dropoff_longitude'] <= longLimits[1])]
combine[0] = combine[0][(combine[0]['trip_duration'] >= tripLimits[0])
                        & (combine[0]['trip_duration'] <= tripLimits[1])]


# Change number of passengers around to reduce possibilities.
combine[0] = combine[0][(combine[0]['passenger_count'] >= 1)]
for dataset in combine:
    dataset.loc[dataset['passenger_count'] >= 7, 'passenger_count'] = 4
    dataset.loc[(dataset['passenger_count'] >= 3) & (dataset['passenger_count'] <= 6), 'passenger_count'] = 3
    dataset['passenger_count'] = dataset['passenger_count'] / 4

# Converting the flag to 1 or 0 from Y or N, and then making one hot
for dataset in combine:
    dataset['store_and_fwd_flag'] = dataset['store_and_fwd_flag'].map({'Y': 1, 'N': 0}).astype(int)

for dataset in combine:
    dataset['yesFlag'] = 0
    dataset['noFlag'] = 0
    dataset.loc[(dataset.store_and_fwd_flag == 0),  'noFlag'] = 1
    dataset.loc[(dataset.store_and_fwd_flag == 1), 'yesFlag'] = 1

combine[0] = combine[0].drop(['store_and_fwd_flag'], axis = 1)
combine[1] = combine[1].drop(['store_and_fwd_flag'], axis = 1)

# New Col: x_distance = difference in longitude
# New Col: y_distance = difference in latitude
# New Col: euclid_dist = Euclidian distance pickup to dropoff.
for dataset in combine:
    dataset['x_distance'] = 0
    dataset['y_distance'] = 0
    dataset['euclid_dist'] = 0
    dataset['man_dist'] = 0

    dataset['x_distance'] = abs(dataset.pickup_longitude - dataset.dropoff_longitude)
    dataset['y_distance'] = abs(dataset.pickup_latitude - dataset.dropoff_latitude)
    dataset['euclid_dist'] = np.sqrt( (dataset['x_distance']*dataset['x_distance'])
                                   + (dataset['y_distance']*dataset['y_distance']) )
    dataset['man_dist'] = dataset['x_distance'] + dataset['y_distance']

# Basically right skew normal so log transform
for ds in combine:
    ds['euclid_dist'] = np.log1p(ds['euclid_dist'])
    ds['x_distance'] = np.log1p(ds['x_distance'])
    ds['y_distance'] = np.log1p(ds['y_distance'])
    ds['man_dist'] = np.log1p(ds['man_dist'])


# Create some neighbourhoods
def getNeighbourhood(long, lat):
    if (long >= -74.019871) & (long <= -73.968201) & (lat >= 40.699884) & (lat <= 40.744750):
        return 0.1
    elif (long >= -74.019871) & (long <= -73.958073) & (lat >= 40.744750) & (lat <= 40.774136):
        return 0.2
    elif (long >= -73.990002) & (long <= -73.926659) & (lat >= 40.774136) & (lat <= 40.805718):
        return 0.3
    elif (long >= -73.968544) & (long <= -73.771820) & (lat >= 40.805718) & (lat <= 41.0):
        return 0.4
    elif (long >= -74.047165) & (long <= -73.854904) & (lat >= 40.565315) & (lat <= 40.701840):
        return 0.5
    elif (long >= -73.854904) & (long <= -73.736115) & (lat >= 40.615371) & (lat <= 40.701840):
        return 0.6
    elif (long >= -73.969917) & (long <= -73.748131) & (lat >= 40.701840) & (lat <= 40.751866):
        return 0.7
    elif (long >= -73.951378) & (long <= -73.844948) & (lat >= 40.751866) & (lat <= 40.789568):
        return 0.8
    elif (long >= -73.844948) & (long <= -73.721695) & (lat >= 40.751866) & (lat <= 40.801264):
        return 0.9
    return 0

# Create features where a ride starts and ends in neighbourhoods
for ds in combine:
    ds['From'] = 0
    ds['To'] = 0

    ds['From'] = ds.apply(lambda row: getNeighbourhood(row['pickup_longitude'], row['pickup_latitude']), axis=1)
    ds['To'] = ds.apply(lambda row: getNeighbourhood(row['dropoff_longitude'], row['dropoff_latitude']), axis=1)


combine[0] = combine[0].round({'pickup_longitude':3, 'pickup_latitude':3, 'dropoff_longitude':3, 'dropoff_latitude':3})
combine[1] = combine[1].round({'pickup_longitude':3, 'pickup_latitude':3, 'dropoff_longitude':3, 'dropoff_latitude':3})
for ds in combine:
    ds['pickup_longitude'] = ds['pickup_longitude'] + 74
    ds['pickup_latitude'] = ds['pickup_latitude'] - 40
    ds['dropoff_longitude'] = ds['dropoff_longitude'] + 74
    ds['dropoff_latitude'] = ds['dropoff_latitude'] - 40



#sns.distplot(combine[0]['euclid_dist']);
# Making vendor ID a one hot.
for dataset in combine:
    dataset['vendor_id_1'] = 0
    dataset['vendor_id_2'] = 0
    dataset.loc[(dataset.vendor_id == 1),  'vendor_id_1'] = 1
    dataset.loc[(dataset.vendor_id == 2), 'vendor_id_2'] = 1

combine[0] = combine[0].drop(['vendor_id'], axis = 1)
combine[1] = combine[1].drop(['vendor_id'], axis = 1)

# Working with the date and time

# Function determines if day is a weekend or holiday (specific to 2016)
def isDayOff(day):
    if (day == 1) | (day == 2) | (day == 3) | (day == 18) | (day == 46) | (day == 85) | (day == 151) | (day == 186)\
            | (day == 249) | (day == 329) | (day == 330) | (day == 361):
        return 1
    elif ((day-2)%7 == 0) | ((day-3)%7 == 0):
        return 1
    return 0


for dataset in combine:
    # New cols we will use:
    dataset['dayOfYear'] = 0
    dataset['Hour'] = 0
    dataset['Minute'] = 0
    dataset['isRushHour'] = 0
    dataset['isMidPeak'] = 0
    dataset['isOffDay'] = 0

    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'], errors='coerce')

    dataset['dayOfYear'] = dataset['pickup_datetime'].dt.dayofyear
    dataset['Hour'] = dataset['pickup_datetime'].dt.hour
    dataset['Minute'] = dataset['pickup_datetime'].dt.minute

    # Determining if a trip started around rush hour
    dataset.loc[(((dataset['Hour'] >= 6) & (dataset['Hour'] < 10))
                 | ((dataset['Hour'] >= 16) & (dataset['Hour'] < 20))), 'isRushHour'] = 1

    # Determining if a trip started around mid-day
    dataset.loc[(dataset['Hour'] >= 10) & (dataset['Hour'] < 16), 'isMidPeak'] = 1

    # Determining if a trip was on a weekend/holiday
    dataset['isOffDay'] = dataset['dayOfYear'].apply(isDayOff)
    # dataset.loc[(dataset['isOffDay'] == 1), 'isRushHour'] = 0
    # dataset.loc[(dataset['isOffDay'] == 1), 'isMidPeak'] = 0



combine[0] = combine[0].drop(['pickup_datetime'], axis = 1)
combine[1] = combine[1].drop(['pickup_datetime'], axis = 1)

# Get average trip durations of rides on the same type of day (weekday, weekend) at the same hour.
guess_times_mean = np.zeros((24, 2, 10, 10))
guess_times_median = np.zeros((24, 2, 10, 10))
guess_vel = np.zeros((24, 2, 10, 10))
for i in range(0, 24):
    for j in range(0, 2):
        for k in range(0, 10):
            for l in range(0, 10):
                # This joins those with the same day type and hour departed
                time_df = combine[0][(combine[0]['isOffDay'] == j) & (combine[0]['Hour'] == i) &
                                   (combine[0]['From'] == (l/10)) &
                                   (combine[0]['To'] == (k/10))]['trip_duration'].dropna()
                dist_df = combine[0][(combine[0]['isOffDay'] == j) & (combine[0]['Hour'] == i) &
                                   (combine[0]['From'] == (l/10)) &
                                   (combine[0]['To'] == (k/10))]['euclid_dist'].dropna()

                guess_dist = dist_df.mean()
                guess_mean = time_df.mean()
                guess_med = time_df.median()

                if np.isnan(guess_mean):
                    guess_times_mean[i, j, k, l] = 0
                    guess_vel[i, j, k, l] = 0
                else:
                    if np.isnan(guess_dist):
                        guess_vel[i, j, k, l] = 0
                    elif guess_mean == 0:
                        guess_vel[i, j, k, l] = 0
                    else:
                        guess_vel[i, j, k, l] = guess_dist / guess_mean

                    guess_times_mean[i, j, k , l] = guess_mean

                if np.isnan(guess_med):
                    guess_times_median[i, j, k, l] = 0
                else:
                    guess_times_median[i, j, k, l] = guess_med

for ds in combine:
    ds['TripMean'] = 0
    ds['TripMed'] = 0
    ds['TripVel'] = 0
    for i in range(0, 24):
        for j in range(0, 2):
            for k in range(0, 10):
                for l in range(k, 10):
                    ds.loc[(ds['isOffDay'] == j) & (ds['Hour'] == i) &
                           (ds['From'] == (l/10)) & (ds['To'] == (k/10)),
                           'TripMean'] = guess_times_mean[i, j, k, l]
                    ds.loc[(ds['isOffDay'] == j) & (ds['Hour'] == i) &
                           (ds['From'] == (l / 10)) & (ds['To'] == (k / 10)),
                           'TripMed'] = guess_times_median[i, j, k, l]
                    ds.loc[(ds['isOffDay'] == j) & (ds['Hour'] == i) &
                           (ds['From'] == (l / 10)) & (ds['To'] == (k / 10)),
                           'TripVel'] = guess_vel[i, j, k, l]

combine[0]['Duration_From_Vel'] = 0
combine[1]['Duration_From_Vel'] = 0

combine[0].loc[(combine[0]['TripVel'] != 0),'Duration_From_Vel'] = combine[0]['euclid_dist'] / combine[0]['TripVel']
combine[1].loc[(combine[1]['TripVel'] != 0),'Duration_From_Vel'] = combine[1]['euclid_dist'] / combine[1]['TripVel']

combine[0] = combine[0].drop(['TripVel'], axis = 1)
combine[1] = combine[1].drop(['TripVel'], axis = 1)

combine[0]['TripMean'] = np.log1p(combine[0]['TripMean'])
combine[0]['TripMed'] = np.log1p(combine[0]['TripMed'])
combine[1]['TripMean'] = np.log1p(combine[1]['TripMean'])
combine[1]['TripMed'] = np.log1p(combine[1]['TripMed'])
combine[0]['Duration_From_Vel'] = np.log1p(combine[0]['Duration_From_Vel'])
combine[1]['Duration_From_Vel'] = np.log1p(combine[1]['Duration_From_Vel'])

combine[0]['trip_duration'] = np.log1p(combine[0]['trip_duration'])


combine[0] = combine[0].drop(['Minute'], axis = 1)
combine[1] = combine[1].drop(['Minute'], axis = 1)

combine[0] = combine[0].drop(['passenger_count'], axis = 1)
combine[1] = combine[1].drop(['passenger_count'], axis = 1)

print("Done pre-processing.")

# Training Models

y_all = combine[0]["trip_duration"]
combine[0] = combine[0].drop("trip_duration", axis=1)

frac_test = 0.2
x_train, x_train2, y_train, y_train2 = train_test_split(combine[0], y_all, test_size = frac_test, random_state=156)


# xg_boost = xgb.XGBRegressor(max_depth=20, learning_rate=0.1, n_estimators=200)
# xg_boost.fit(x_train, y_train, eval_metric='rmse')
# #xgb.plot_importance(xg_boost)
# pred = xg_boost.predict(x_train2)
# print("xg: ", np.sqrt( mean_squared_error(np.expm1(y_train2), np.expm1(pred)) ) )

# Neural Net
# mlp = MLPRegressor(hidden_layer_sizes=(15,7,4), learning_rate_init = 0.01, random_state=252)
# mlp.fit(x_train,y_train)
# pred = mlp.predict(x_train2)
# print("Neural Net: ", np.sqrt( mean_squared_error(np.expm1(y_train2), np.expm1(pred)) ) )

# Random Forest
rf = RandomForestRegressor(n_estimators = 25, random_state=147)
rf.fit(x_train,y_train)
pred = rf.predict(x_train2)
print("Random Forest: ", np.sqrt( mean_squared_error(np.expm1(y_train2), np.expm1(pred)) ) )

# Gradient Boosting
gbc = GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.1, random_state=1652)
gbc.fit(x_train,y_train)
pred = gbc.predict(x_train2)
print("Gradient Boost: ", np.sqrt( mean_squared_error(np.expm1(y_train2), np.expm1(pred)) ) )

# Decision Tree
dt = DecisionTreeRegressor(max_depth = 15, random_state=2422)
dt.fit(x_train,y_train)
pred = dt.predict(x_train2)
print("Decision Tree: ", np.sqrt( mean_squared_error(np.expm1(y_train2), np.expm1(pred)) ) )



#predictions_MLP_train = mlp.predict(x_train2)
predictions_RF_train = rf.predict(x_train2)
predictions_GBC_train = gbc.predict(x_train2)
predictions_DT_train = dt.predict(x_train2)

# predictions_AVG = (predictions_MLP_train+predictions_AB_train+predictions_RF_train
#                    +predictions_GBC_train+predictions_DT_train) / 5

# predictions_AVG = predictions_AVG.reshape(-1,1)
#predictions_MLP_train = predictions_MLP_train.reshape(-1,1)
predictions_RF_train = predictions_RF_train.reshape(-1,1)
predictions_GBC_train = predictions_GBC_train.reshape(-1,1)
predictions_DT_train = predictions_DT_train.reshape(-1,1)


# next_x_train = np.concatenate((predictions_RF_train,
#                                predictions_GBC_train,predictions_DT_train), axis=1)
x_train2['pred_RF'] = predictions_RF_train[:]
x_train2['pred_GBC'] = predictions_GBC_train[:]
x_train2['pred_DT'] = predictions_DT_train[:]

print("Training second layer")

xg_boost = xgb.XGBRegressor(max_depth=20, learning_rate=0.1, n_estimators=200)
xg_boost.fit(x_train2, y_train2, eval_metric='rmse')


x_test = combine[1].drop(['id'], axis=1)

# First level predictions of test set:
#predictions_MLP_train = mlp.predict(x_test)
predictions_RF_train = rf.predict(x_test)
predictions_GBC_train = gbc.predict(x_test)
predictions_DT_train = dt.predict(x_test)

# predictions_AVG = (predictions_MLP_train + predictions_AB_train + predictions_RF_train
#                    + predictions_GBC_train + predictions_DT_train) / 5

# predictions_AVG = predictions_AVG.reshape(-1,1)
#predictions_MLP_train = predictions_MLP_train.reshape(-1,1)
predictions_RF_train = predictions_RF_train.reshape(-1,1)
predictions_GBC_train = predictions_GBC_train.reshape(-1,1)
predictions_DT_train = predictions_DT_train.reshape(-1,1)

# next_x_test = np.concatenate((predictions_RF_train,
#                                predictions_GBC_train,predictions_DT_train), axis=1)
x_test['pred_RF'] = predictions_RF_train[:]
x_test['pred_GBC'] = predictions_GBC_train[:]
x_test['pred_DT'] = predictions_DT_train[:]


# Final model prediction
final_pred = xg_boost.predict(x_test)
final_pred = np.expm1(final_pred)
sub = pd.DataFrame({'id': combine[1]['id'], 'trip_duration': final_pred})
sub.to_csv("Taxi_Ride_Submission.csv", index=False)







