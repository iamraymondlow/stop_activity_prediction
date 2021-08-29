import json
import pandas as pd
import random
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from copy import deepcopy

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
        self.data = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                               config['processed_data_directory'] + 'combined_stop_data.xlsx'))

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
            assert np.any(time_diff >= 0.0)

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
        return driver_data

    def _extract_past_activities(self, data):
        """
        Extracts past activities performed by each driver.

        Parameters:
            data: pd.Dataframe
                Contains the combined HVP dataset.

        Return:
            data: pd.DataFrame
                Contains the combined HVP dataset with past activities performed by each driver
        """
        return data

    def train_test_split(self, test_ratio=0.3):
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
        #TODO

        # extract additional features based on other drivers' activities
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
        train_data = self._extract_past_activities(train_data)
        test_data = self._extract_past_activities(test_data)

        # once again check if stops are in chronological order
        self.check_stop_order(train_data)
        self.check_stop_order(test_data)

        # drop irrelevant columns
        dropped_cols = ['StopLat', 'StopLon', 'Address', 'StartTime', 'EndTime', 'StopID',
                        'TripID', 'Stops', 'Travels', 'YMD', 'LandUseType', 'geometry']
        retained_cols = [column for column in self.data.columns if column not in dropped_cols]
        train_data = train_data[retained_cols]
        test_data = test_data[retained_cols]

        return train_data, test_data


if __name__ == '__main__':
    extractor = DataLoader()
    data = extractor.data
    train_data, test_data = extractor.train_test_split(test_ratio=0.3)
