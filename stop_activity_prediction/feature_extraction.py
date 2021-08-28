import json
import pandas as pd
import os
from pathlib import Path

# load config file
with open(Path(os.path.dirname(os.path.realpath(__file__)), '../config.json')) as f:
    config = json.load(f)


class FeatureExtractor:
    """
    Perform feature extraction on the combined HVP dataset containing POI data and URA land use data.
    """
    def __init__(self):
        """
        Initialises the class object by loading the combined HVP dataset containing POI data and URA land use data.
        """
        self.data = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                               config['processed_data_directory'] + 'combined_stop_data.xlsx'))

    def train_test_split(self, test_ratio=0.3):
        # perform train test split

        # extract additional features based on other drivers' activities

        # extract additional features based on drivers' past activities

        return pd.DataFrame(), pd.DataFrame()

    def _extract_other_driver_activities(self):
        return None

    def _extract_past_activities(self):
        return None


if __name__ == '__main__':
    extractor = FeatureExtractor()
    data = extractor.data
    train_data, test_data = extractor.train_test_split(test_ratio=0.3)
