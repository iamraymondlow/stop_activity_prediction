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
        self.combined_data = pd.DataFrame()

    def _load_verified_trips(self, batch_num):
        with open(config['verified_stop_directory'].format(batch_num=batch_num)) as f:
            verified_trips = json.load(f)
        verified_trips = pd.json_normalize(verified_trips)

        # filter important features
        retained_columns = ['DriverID', 'VehicleType', 'Stops', 'Travels', 'YMD', 'Timeline', 'DayOfWeekStr']

        return verified_trips[retained_columns]

    def _load_operational_survey(self, batch_num):
        # load operational survey
        with open(config['operation_survey_directory'].format(batch_num=batch_num)) as f:
            operational_data = json.load(f)
        operational_data = pd.json_normalize(operational_data)

        # filter important features
        important_features = ['FrequentPlaces', 'Commodity', 'SpecialCargo', 'Company.Type', 'Industry',
                              'Driver.ID']
        retained_columns = [column
                            for column in operational_data.columns
                            for feature in important_features
                            if feature in column]

        return operational_data[retained_columns]

    def _extract_verified_stops(self, verified_trips):
        return None

    def process_data(self, batch_num):
        # import verified trip information
        verified_trips = self._load_verified_trips(batch_num)

        # import operational survey data
        operational_data = self._load_operational_survey(batch_num)

        # extract verified stop information
        # verified_stops = self._extract_verified_stops(verified_trips)

        # merge data
        batch_data = verified_trips.merge(operational_data, how='left',
                                          right_on='Driver.ID', left_on='DriverID')

        # store processed data locally
        if not os.path.exists(config['processed_data_directory']):
            os.makedirs(config['processed_data_directory'])

        batch_data.to_excel(config['processed_data_directory'] + 'batch_data_{}.xlsx'.format(batch_num), index=False)
        self.combined_data = pd.concat([self.combined_data, batch_data], ignore_index=True)
        self.combined_data.to_excel(config['processed_data_directory'] + 'combined_data.xlsx', index=False)

        return verified_stops, operational_data, batch_data


if __name__ == '__main__':
    processor = DataProcessor()
    verified_stops, operational_data, batch_data = processor.process_data(batch_num=1)
    # processor.process_data(batch_num=1)
    # processor.process_data(batch_num=2)
    # processor.process_data(batch_num=3)
    # processor.process_data(batch_num=4)
    # processor.process_data(batch_num=5)
    # processor.process_data(batch_num=6)
    # processor.process_data(batch_num=7)
    # processor.process_data(batch_num=8)
