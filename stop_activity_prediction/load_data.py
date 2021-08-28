import json
import pandas as pd
import random
import os
from pathlib import Path

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

        # perform one hot encoding

        # extract additional features based on other drivers' activities

        # extract additional features based on drivers' past activities

        # drop irrelevant columns
        dropped_cols = ['StopLat', 'StopLon', 'Address', 'StartTime', 'EndTime', 'StopID',
                        'TripID', 'Stops', 'Travels', 'YMD', 'LandUseType']
        retained_cols = [column for column in self.data.columns if column not in dropped_cols]
        train_data = train_data[retained_cols]
        test_data = test_data[retained_cols]

        return train_data, test_data

    def _extract_other_driver_activities(self):
        return None

    def _extract_past_activities(self):
        return None


if __name__ == '__main__':
    extractor = DataLoader()
    data = extractor.data
    train_data, test_data = extractor.train_test_split(test_ratio=0.3)
