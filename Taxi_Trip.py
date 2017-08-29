# Predictive model where we attempt to predict the time duration of a taxi ride in New York City.

import pandas as pd
import numpy as np

# machine learning
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold


import warnings
warnings.filterwarnings('ignore')


# Load .csv files
train_df = pd.read_csv('taxitrain.csv')
test_df = pd.read_csv('taxitest.csv')
combine = [train_df, test_df]


# Cleaning/Engineering etc.

# Dropping cols that are useless
combine[0] = combine[0].drop(['id'], axis = 1)

combine[0] = combine[0].drop(['dropoff_datetime'], axis = 1)

# Some trip times and columns are not correct, discard any trip longer than 2.5 hours.
combine[0] = combine[0][combine[0].trip_duration < 9000]

# Change number of passengers around to reduce possibilities.
for dataset in combine:
    dataset.loc[ dataset['passenger_count'] >= 7, 'passenger_count'] = 5
    dataset.loc[(dataset['passenger_count'] >= 3) & (dataset['passenger_count'] <= 6), 'passenger_count'] = 4


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
    dataset['x_distance'] = abs(dataset.pickup_longitude - dataset.dropoff_longitude)
    dataset['y_distance'] = abs(dataset.pickup_latitude - dataset.dropoff_latitude)
    dataset['euclid_dist'] = np.sqrt( (dataset['x_distance']*dataset['x_distance'])
                                   + (dataset['y_distance']*dataset['y_distance']) )


# Scale the distances
minx = combine[0]['x_distance'].min()
miny = combine[0]['y_distance'].min()
maxx = combine[0]['x_distance'].max()
maxy = combine[0]['y_distance'].max()

combine[0]['x_distance'] = (combine[0]['x_distance']-minx)/(maxx-minx)
combine[0]['y_distance'] = (combine[0]['y_distance']-miny)/(maxy-miny)

minx = combine[1]['x_distance'].min()
miny = combine[1]['y_distance'].min()
maxx = combine[1]['x_distance'].max()
maxy = combine[1]['y_distance'].max()

combine[1]['x_distance'] = (combine[1]['x_distance']-minx)/(maxx-minx)
combine[1]['y_distance'] = (combine[1]['y_distance']-miny)/(maxy-miny)


#Scale regular lat and longs
minLat = combine[0]['pickup_latitude'].min()
minLong = combine[0]['pickup_longitude'].min()
maxLat = combine[0]['pickup_latitude'].max()
maxLong = combine[0]['pickup_longitude'].max()

combine[0]['pickup_latitude'] = (combine[0]['pickup_latitude']-minLat)/(maxLat-minLat)
combine[0]['pickup_longitude'] = (combine[0]['pickup_longitude']-minLong)/(maxLong-minLong)

minLat = combine[1]['pickup_latitude'].min()
minLong = combine[1]['pickup_longitude'].min()
maxLat = combine[1]['pickup_latitude'].max()
maxLong = combine[1]['pickup_longitude'].max()

combine[1]['pickup_latitude'] = (combine[0]['pickup_latitude']-minLat)/(maxLat-minLat)
combine[1]['pickup_longitude'] = (combine[0]['pickup_longitude']-minLong)/(maxLong-minLong)

minLat = combine[0]['dropoff_latitude'].min()
minLong = combine[0]['dropoff_longitude'].min()
maxLat = combine[0]['dropoff_latitude'].max()
maxLong = combine[0]['dropoff_longitude'].max()

combine[0]['dropoff_latitude'] = (combine[0]['dropoff_latitude']-minLat)/(maxLat-minLat)
combine[0]['dropoff_longitude'] = (combine[0]['dropoff_longitude']-minLong)/(maxLong-minLong)

minLat = combine[1]['dropoff_latitude'].min()
minLong = combine[1]['dropoff_longitude'].min()
maxLat = combine[1]['dropoff_latitude'].max()
maxLong = combine[1]['dropoff_longitude'].max()

combine[1]['dropoff_latitude'] = (combine[0]['dropoff_latitude']-minLat)/(maxLat-minLat)
combine[1]['dropoff_longitude'] = (combine[0]['dropoff_longitude']-minLong)/(maxLong-minLong)


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


combine[0] = combine[0].drop(['pickup_datetime'], axis = 1)
combine[1] = combine[1].drop(['pickup_datetime'], axis = 1)

combine[0]['trip_duration'] = np.log1p(combine[0]['trip_duration'])

#print(combine[0].head())

# Training Models

ntrain = combine[0].shape[0]
ntest = combine[1].shape[0]
y_train = combine[0].trip_duration.values
combine[0] = combine[0].drop(['trip_duration'], axis = 1)


#Validation function

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0001, random_state=20))
en = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0001, l1_ratio=.9, random_state=21))
nn = MLPRegressor(hidden_layer_sizes=(14,7,5), random_state=22)
lasso2 = make_pipeline(RobustScaler(), Lasso(alpha =0.0001, random_state=19))

# From Serigne:
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


stacked_averaged_models = StackingAveragedModels(base_models = (lasso, en, nn), meta_model = lasso2)

stacked_averaged_models.fit(combine[0].values, y_train)

test = combine[1].drop(['id'], axis = 1)

final_pred = np.expm1(stacked_averaged_models.predict(test))

sub = pd.DataFrame({'id': combine[1]['id'], 'trip_duration': final_pred})
sub.to_csv("Taxi_Ride_Submission.csv", index=False)
