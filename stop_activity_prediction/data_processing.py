import json
import pandas as pd
import os
from pathlib import Path

# load config file
with open(Path(os.path.dirname(os.path.realpath(__file__)), '../config.json')) as f:
    config = json.load(f)


class DataProcessor:
    """
    Perform processing on the HVP dataset by combining verified stop information, data from
    operational survey and vehicle information.
    """
    def __init__(self):
        """
        Initialises the class object with an empty dataframe to store the combined data for each batch.
        """
        self.combined_data = pd.DataFrame()

    def _load_verified_trips(self, batch_num):
        """
        Loads the verified trips data for a particular batch and removes the irrelevant columns.

        Parameters:
            batch_num: int
                Contains the batch number.

        Return:
            verified_trips[retained_columns]: pandas.DataFrame
                Contains the verified trips information for a particular batch.
        """
        with open(config['verified_stop_directory'].format(batch_num=batch_num)) as f:
            verified_trips = json.load(f)
        verified_trips = pd.json_normalize(verified_trips)

        # filter important features
        retained_columns = ['DriverID', 'VehicleType', 'Stops', 'Travels', 'YMD', 'Timeline', 'DayOfWeekStr']

        return verified_trips[retained_columns]

    def load_operation_survey(self, batch_num):
        """
        Loads the operation survey for a particular batch and removes the irrelevant columns.

        Parameters:
            batch_num: int
                Contains the batch number.

        Return:
            operation_data[retained_columns]: pandas.DataFrame
                Contains the operation survey data for a particular batch.
        """
        # load operational survey
        with open(config['operation_survey_directory'].format(batch_num=batch_num)) as f:
            operation_data = json.load(f)
        operation_data = pd.json_normalize(operation_data)

        # filter important features
        important_features = ['FrequentPlaces', 'Commodity', 'SpecialCargo', 'Company.Type', 'Industry',
                              'Driver.ID']
        retained_columns = [column
                            for column in operation_data.columns
                            for feature in important_features
                            if feature in column]

        return operation_data[retained_columns]

    def _extract_verified_stops(self, verified_trips):
        """
        Extracts the verified stop information based on the verified trips.

        Parameters:
            verified_trips: pandas.DataFrame
                Contains the verified trip information for a particular batch
        """
        return None

    def process_data(self, batch_num):
        """
        Performs data fusion and subsequent processing for the verified trips and operation survey data
        for a particular batch. The processed batch data and combined dataset is saved in the local directory.

        Parameters:
            batch_num: int
                Contains the batch number.
        """
        print('Processing Batch {} Data...'.format(batch_num))

        # import verified trip information
        verified_trips = self._load_verified_trips(batch_num)

        # import operation survey data
        operational_data = self._load_operation_survey(batch_num)

        # extract verified stop information
        # verified_stops = self._extract_verified_stops(verified_trips)

        # merge data
        batch_data = verified_trips.merge(operational_data, how='left',
                                          right_on='Driver.ID', left_on='DriverID')
        batch_data.drop(columns=['Driver.ID'], inplace=True)

        # store processed batch data and combined data locally
        if not os.path.exists(config['processed_data_directory']):
            os.makedirs(config['processed_data_directory'])

        batch_data.to_excel(config['processed_data_directory'] + 'batch_data_{}.xlsx'.format(batch_num), index=False)
        self.combined_data = pd.concat([self.combined_data, batch_data], ignore_index=True)
        self.combined_data.to_excel(config['processed_data_directory'] + 'combined_data.xlsx', index=False)


if __name__ == '__main__':
    processor = DataProcessor()
    processor.process_data(batch_num=1)
    processor.process_data(batch_num=2)
    processor.process_data(batch_num=3)
    processor.process_data(batch_num=4)
    processor.process_data(batch_num=5)
    processor.process_data(batch_num=6)
    processor.process_data(batch_num=7)
    processor.process_data(batch_num=8)
