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

    def _load_verified_stops(self, batch_num):
        return None

    def _load_operational_survey(self, batch_num):
        return None

    def _load_vehicle_info(self, batch_num):
        return None

    def process_data(self, batch_num):
        # import verified stop information
        verified_stops = self._load_verified_stops(batch_num)

        # import operational survey data
        operational_data = self._load_operational_survey(batch_num)

        # import vehicle information
        vehicle_information = self._load_vehicle_info(batch_num)

        # merge data
        batch_data = pd.merge(verified_stops, operational_data, on='')
        batch_data = pd.merge(batch_data, vehicle_information, on='')

        # store processed data locally
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
