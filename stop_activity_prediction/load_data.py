import json
import pandas as pd
import random
import os
import pyproj
import numpy as np
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from shapely.geometry import Point
from shapely.ops import transform
from sklearn.preprocessing import OneHotEncoder

# load config file
with open(Path(os.path.dirname(os.path.realpath(__file__)), '../config.json')) as f:
    config = json.load(f)


class DataLoader:
    """
    Loads the combined HVP dataset containing POI data and URA land use data and performs data preparation.
    """
    def __init__(self):
        """
        Initialises the class object by loading the combined HVP dataset containing POI data and URA land use data.
        """
        print('Loading batch data...')
        batch1 = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                            config['processed_data_directory'] + 'batch_stop_data_1.xlsx'))
        batch2 = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                            config['processed_data_directory'] + 'batch_stop_data_2.xlsx'))
        batch3 = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                            config['processed_data_directory'] + 'batch_stop_data_3.xlsx'))
        batch4 = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                            config['processed_data_directory'] + 'batch_stop_data_4.xlsx'))
        batch5 = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                            config['processed_data_directory'] + 'batch_stop_data_5.xlsx'))
        batch6 = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                            config['processed_data_directory'] + 'batch_stop_data_6.xlsx'))
        batch7 = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                            config['processed_data_directory'] + 'batch_stop_data_7.xlsx'))
        batch8 = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                            config['processed_data_directory'] + 'batch_stop_data_8.xlsx'))
        self.data = pd.concat([batch1, batch2, batch3, batch4, batch5, batch6, batch7, batch8], ignore_index=True)

    def check_stop_order(self, data):
        """
        Checks if the stops made by each driver is in chronological order.

        Parameters:
            data: pd.Dataframe
                Contains the combined HVP dataset.
        """
        for driver_id in data['DriverID'].unique():
            driver_data = deepcopy(data[data['DriverID'] == driver_id].reset_index(drop=True))
            unix_time = np.array([datetime.strptime(time_str, '%Y-%m-%d %H-%M-%S').timestamp()
                                  for time_str in driver_data['StartTime'].tolist()])
            time_diff = unix_time[1:] - unix_time[:-1]
            if len(driver_data) > 1:
                assert np.any(time_diff >= 0.0)

    def _buffer_in_meters(self, lng, lat, radius):
        """
        Converts a latitude, longitude coordinate pair into a buffer with user-defined radius.s

        :param lng: float
            Contains the longitude information.
        :param lat: float
            Contains the latitude information.
        :param radius: float
            Contains the buffer radius in metres.
        :return:
        buffer_latlng: Polygon
            Contains the buffer.
        """
        proj_meters = pyproj.CRS('EPSG:3414')  # EPSG for Singapore
        proj_latlng = pyproj.CRS('EPSG:4326')

        project_to_metres = pyproj.Transformer.from_crs(proj_latlng, proj_meters, always_xy=True).transform
        project_to_latlng = pyproj.Transformer.from_crs(proj_meters, proj_latlng, always_xy=True).transform
        pt_meters = transform(project_to_metres, Point(lng, lat))
        buffer_meters = pt_meters.buffer(radius)
        buffer_latlng = transform(project_to_latlng, buffer_meters)
        return buffer_latlng

    def _extract_other_driver_activities(self, driver_data, other_driver_data):
        """
        Extracts the activity information performed by other drivers in the same area.

        Parameters:
            driver_data: pd.Dataframe
                Contains the combined HVP dataset for a particular driver.
            other_driver_data: pd.Dataframe
                Contains the combined HVP dataset for the other drivers.

        Return:
            driver: pd.Dataframe
                Contains the combined HVP dataset for a particular driver + past activities of other drivers
        """
        other_driver_activities = pd.DataFrame()
        driver_data = gpd.GeoDataFrame(driver_data,
                                       geometry=gpd.points_from_xy(driver_data['StopLon'],
                                                                   driver_data['StopLat']))
        other_driver_data = gpd.GeoDataFrame(other_driver_data,
                                             geometry=gpd.points_from_xy(other_driver_data['StopLon'],
                                                                         other_driver_data['StopLat']))
        for i in range(len(driver_data)):
            # create 100m circular buffer around stop
            buffer = self._buffer_in_meters(driver_data.loc[i, 'StopLon'],
                                            driver_data.loc[i, 'StopLat'], 50.0)
            nearby_stops = other_driver_data[other_driver_data.intersects(buffer)].reset_index(drop=True)

            if len(nearby_stops) == 0:
                other_driver_activities = other_driver_activities.append(pd.Series(dtype=object), ignore_index=True)
            else:
                activity_cols = [col for col in nearby_stops.columns
                                 if ('Activity.' in col)
                                 and ('MappedActivity.' not in col)
                                 and ('Other.' not in col)]
                mapped_activity_cols = [col for col in nearby_stops.columns
                                        if ('MappedActivity.' in col) and ('Other.' not in col)]

                # calculate distribution of activities conducted near the stop
                summed_activity = nearby_stops.sum()[activity_cols]
                normalised_activity = (summed_activity * 100) / (summed_activity.sum() + 1e-9)

                # calculate distribution of mapped activities conducted near the stop
                summed_mapped_activity = nearby_stops.sum()[mapped_activity_cols]
                normalised_mapped_activity = (summed_mapped_activity * 100) / (summed_mapped_activity.sum() + 1e-9)

                # merge original and mapped activity types conducted by other drivers
                other_driver_activities = other_driver_activities.append(pd.concat([normalised_activity,
                                                                                    normalised_mapped_activity]).T,
                                                                         ignore_index=True)

        assert len(driver_data) == len(other_driver_activities)
        other_driver_activities_cols = ['Other.{}'.format(column) for column in other_driver_activities.columns]
        other_driver_activities.columns = other_driver_activities_cols
        driver_data = pd.concat([driver_data, other_driver_activities], axis=1)
        driver_data.fillna(0, inplace=True)
        return driver_data

    def _extract_past_activities(self, data):
        """
        Extracts past activities performed by each driver.

        Parameters:
            data: pd.Dataframe
                Contains the combined HVP dataset.

        Return:
            new_data: pd.DataFrame
                Contains the combined HVP dataset with past activities performed by each driver
        """
        assert type(data) == gpd.GeoDataFrame
        new_data = pd.DataFrame()

        # extract unix time of each stop
        data['StopUnixTime'] = [datetime.strptime(time_str, '%Y-%m-%d %H-%M-%S').timestamp()
                                for time_str in data['StartTime'].tolist()]

        for driver_id in data['DriverID'].unique():
            driver_data = deepcopy(data[data['DriverID'] == driver_id].reset_index(drop=True))
            past_activities = pd.DataFrame()

            for i in range(len(driver_data)):
                # create 100m circular buffer around stop
                buffer = self._buffer_in_meters(driver_data.loc[i, 'StopLon'],
                                                driver_data.loc[i, 'StopLat'], 50.0)

                nearby_stops = driver_data[driver_data.intersects(buffer)].reset_index(drop=True)
                nearby_stops = nearby_stops[nearby_stops['StopUnixTime'] <
                                            driver_data.loc[i, 'StopUnixTime']].reset_index(drop=True)

                if len(nearby_stops) == 0:
                    past_activities = past_activities.append(pd.Series({'Activity.Shift': 0}), ignore_index=True)
                else:
                    activity_cols = [col for col in nearby_stops.columns
                                     if ('Activity.' in col) and
                                     ('MappedActivity.' not in col) and
                                     ('Other.' not in col)]
                    mapped_activity_cols = [col for col in nearby_stops.columns
                                            if ('MappedActivity.' in col) and
                                            ('Other.' not in col)]

                    # calculate distribution of activities conducted near the stop
                    summed_activity = nearby_stops.sum()[activity_cols]
                    normalised_activity = (summed_activity * 100) / (summed_activity.sum() + 1e-9)

                    # calculate distribution of mapped activities conducted near the stop
                    summed_mapped_activity = nearby_stops.sum()[mapped_activity_cols]
                    normalised_mapped_activity = (summed_mapped_activity * 100) / (summed_mapped_activity.sum() + 1e-9)

                    past_activities = past_activities.append(pd.concat([normalised_activity,
                                                                        normalised_mapped_activity]).T,
                                                             ignore_index=True)

            assert len(driver_data) == len(past_activities)
            past_activities_cols = ['Past.{}'.format(column) for column in past_activities.columns]
            past_activities.columns = past_activities_cols
            driver_data = pd.concat([driver_data, past_activities], axis=1)
            driver_data.fillna(0, inplace=True)

            new_data = pd.concat([new_data, driver_data], ignore_index=True)
            new_data.fillna(0, inplace=True)

        return new_data

    def _one_hot_encoding(self, train_col, test_col, feature_name):
        """
        Performs one hot encoding of a particular column for both training and test datasets.

        Parameters:
            train_col: pd.Series
                Contains the column to be one-hot-encoded from the training dataset.
            test_col: pd.Series
                Contains the column to be one-hot-encoded from the test dataset.
            feature_name: str
                Contains the name of the feature to be one-hot-encoded.

        Return:
            train_onehot_df: pd.Dataframe
                Contains the one-hot-encoded dataframe of the column in the training dataset.
            test_onehot_df: pd.Dataframe
                Contains the one-hot-encoded dataframe of the column in the test dataset.
        """
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(np.array(pd.concat([train_col, test_col], ignore_index=True)).reshape(-1, 1))
        train_onehot_df = pd.DataFrame(encoder.transform(np.array(train_col).reshape(-1, 1)),
                                       columns=['{}.{}'.format(feature_name, cat.replace('X_', ''))
                                                for cat in encoder.get_feature_names(['X'])])
        test_onehot_df = pd.DataFrame(encoder.transform(np.array(test_col).reshape(-1, 1)),
                                      columns=['{}.{}'.format(feature_name, cat.replace('X_', ''))
                                               for cat in encoder.get_feature_names(['X'])])
        return train_onehot_df, test_onehot_df

    def train_test_split(self, test_ratio=0.25):
        """
        Performs train test split on the combined HVP dataset and performs feature extraction.

        Parameters:
            test_ratio: float
                Contains the ratio for the test dataset.

        Return:
            train_data: pd.Dataframe
                Contains the training dataset after feature extraction.
            test_data: pd.Dataframe
                Contains the test dataset after feature extraction.
        """
        # check local directory and load cache if available
        print('Performing train test split...')
        if (os.path.exists(os.path.join(os.path.dirname(__file__),
                                        config['processed_data_directory'] + 'train_data.xlsx'))) and \
            (os.path.exists(os.path.join(os.path.dirname(__file__),
                                         config['processed_data_directory'] + 'test_data.xlsx'))):
            train_data = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                                    config['processed_data_directory'] + 'train_data.xlsx'))
            test_data = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                                   config['processed_data_directory'] + 'test_data.xlsx'))

            return train_data, test_data

        # perform train test split
        driver_id = self.data['DriverID'].unique()
        random.shuffle(driver_id)
        test_id = driver_id[:int(len(driver_id) * test_ratio)]
        train_id = driver_id[int(len(driver_id) * test_ratio):]
        train_data = self.data[self.data['DriverID'].isin(train_id)].reset_index(drop=True)
        test_data = self.data[self.data['DriverID'].isin(test_id)].reset_index(drop=True)

        # check if stops are in chronological order
        self.check_stop_order(train_data)
        self.check_stop_order(test_data)

        # perform one hot encoding
        print('Performing one hot encoding...')
        train_vehtype, test_vehtype = self._one_hot_encoding(train_data['VehicleType'],
                                                             test_data['VehicleType'], 'VehicleType')
        train_dayofweek, test_dayofweek = self._one_hot_encoding(train_data['DayOfWeekStr'],
                                                                 test_data['DayOfWeekStr'], 'DayOfWeek')
        train_landuse, test_landuse = self._one_hot_encoding(train_data['MappedLandUseType'],
                                                             test_data['MappedLandUseType'], 'LandUse')
        assert len(train_vehtype) == len(train_data)
        assert len(train_dayofweek) == len(train_data)
        assert len(train_landuse) == len(train_data)
        assert len(test_vehtype) == len(test_data)
        assert len(test_dayofweek) == len(test_data)
        assert len(test_landuse) == len(test_data)
        train_data = pd.concat([train_data, train_vehtype, train_dayofweek, train_landuse], axis=1)
        train_data.drop(columns=['VehicleType', 'DayOfWeekStr', 'MappedLandUseType'], inplace=True)
        test_data = pd.concat([test_data, test_vehtype, test_dayofweek, test_landuse], axis=1)
        test_data.drop(columns=['VehicleType', 'DayOfWeekStr', 'MappedLandUseType'], inplace=True)

        # extract additional features based on other drivers' activities
        print('Extracting activity information of other drivers...')
        temp_train_data = pd.DataFrame()
        for driver_id in train_data['DriverID'].unique():
            driver_data = deepcopy(train_data[train_data['DriverID'] == driver_id].reset_index(drop=True))
            other_driver_data = deepcopy(train_data[train_data['DriverID'] != driver_id].reset_index(drop=True))
            temp_train_data = pd.concat([temp_train_data,
                                         self._extract_other_driver_activities(driver_data, other_driver_data)],
                                        ignore_index=True)
        train_data = deepcopy(temp_train_data.reset_index(drop=True))
        test_data = self._extract_other_driver_activities(test_data, train_data)

        # extract additional features based on drivers' past activities
        print('Extracting past activity information...')
        train_data = self._extract_past_activities(train_data)
        test_data = self._extract_past_activities(test_data)

        # once again check if stops are in chronological order
        self.check_stop_order(train_data)
        self.check_stop_order(test_data)

        # drop irrelevant columns
        dropped_cols = ['StopLat', 'StopLon', 'Address', 'StartTime', 'EndTime', 'StopID',
                        'TripID', 'Stops', 'Travels', 'YMD', 'LandUseType', 'geometry',
                        'VehicleType', 'DayOfWeekStr', 'MappedLandUseType', 'LandUse.nan',
                        'StopUnixTime']
        retained_train_cols = [column for column in train_data.columns if column not in dropped_cols]
        retained_test_cols = [column for column in test_data.columns if column not in dropped_cols]
        assert retained_train_cols == retained_test_cols
        train_data = train_data[retained_train_cols]
        test_data = test_data[retained_test_cols]

        # save training and test dataset in local directory
        if not os.path.exists(os.path.join(os.path.dirname(__file__), config['processed_data_directory'])):
            os.makedirs(os.path.join(os.path.dirname(__file__), config['processed_data_directory']))

        train_data.to_excel(os.path.join(os.path.dirname(__file__),
                                         config['processed_data_directory'] + 'train_data.xlsx'.format(train_data)),
                            index=False, encoding='utf-8')
        test_data.to_excel(os.path.join(os.path.dirname(__file__),
                                        config['processed_data_directory'] + 'test_data.xlsx'.format(test_data)),
                           index=False, encoding='utf-8')

        return train_data, test_data


if __name__ == '__main__':
    extractor = DataLoader()
    data = extractor.data
    train_data, test_data = extractor.train_test_split(test_ratio=0.25)
