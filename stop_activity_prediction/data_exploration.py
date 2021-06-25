import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path

# load config file
with open(Path(os.path.dirname(os.path.realpath(__file__)), '../config.json')) as f:
    config = json.load(f)


class DataExplorer:
    """
    Performs data exploration and visualisation of trip and stop activity data.
    """
    def __init__(self):
        """
        Initialises the class object by storing the combined trip data and stop activity data
        as class attributes.
        """
        trip_data = pd.read_excel(config['processed_data_directory'] + 'combined_trip_data.xlsx')
        stop_data = pd.read_excel(config['processed_data_directory'] + 'combined_stop_data.xlsx')
        self.trip_data = trip_data
        self.stop_data = stop_data

    def _plot_bar_graph(self, feature_name, title):
        """
        Plots a bar graph for a user-defined feature.

        Parameters:
            feature_name: str
                Contains the feature substring that can be found in the columns of interest.
            title: str
                Contains the title indicated on the bar graph.
        """
        feature_columns = [column for column in self.trip_data.columns if feature_name in column]
        feature_data = self.trip_data[feature_columns]
        feature_sum = feature_data.sum(axis=0)

        feature_sum.plot(kind='bar')
        feature_types = [column.replace('{}.'.format(feature_name), '') for column in feature_columns]
        plt.xticks(ticks=range(len(feature_types)), labels=feature_types)
        plt.ylabel('Frequency')
        plt.title(title)
        plt.show()

    def calculate_trip_statistics(self):
        """
        Calculates different statistics related to trip data.
        """
        # calculate number of unique trips
        print('Number of unique trips: {}'.format(len(self.trip_data)))

        # calculate number of unique drivers
        print('Number of unique drivers: {}'.format(len(self.trip_data['DriverID'].unique())))

        # calculate average number of stops per trip and visualise stop distribution
        print('Average number of stops per trip: {}'.format(self.trip_data['Stops'].mean()))
        self.trip_data.boxplot(column=['Stops'])
        plt.ylabel('Number of Stops per Trip')
        plt.show()

        # vehicle type breakdown
        vehicle_type = self.trip_data['VehicleType'].value_counts()
        vehicle_type.plot(kind='bar')
        plt.ylabel('Frequency')
        plt.title('Vehicle Type')
        plt.show()

        # commodity type breakdown
        self._plot_bar_graph('Commodity', 'Commodity Type')

        # special cargo type breakdown
        self._plot_bar_graph('SpecialCargo', 'Special Cargo Type')

        # company type breakdown
        self._plot_bar_graph('Company.Type', 'Company Type')

        # industry type breakdown
        self._plot_bar_graph('Industry', 'Industry Type')

    def calculate_stop_statistics(self):
        """
        Calculates different statistics related to stop data.
        """
        return None


if __name__ == '__main__':
    explorer = DataExplorer()
    explorer.calculate_trip_statistics()
    explorer.calculate_stop_statistics()
