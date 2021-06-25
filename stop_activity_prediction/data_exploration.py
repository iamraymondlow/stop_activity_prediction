import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
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
        self.activity_types = ['DeliverCargo', 'PickupCargo', 'Other', 'Shift', 'ProvideService',
                               'OtherWork', 'Meal', 'DropoffTrailer', 'PickupTrailer', 'Fueling',
                               'Personal', 'Passenger', 'Resting', 'Queuing', 'DropoffContainer',
                               'PickupContainer', 'Fail', 'Passenger']

        # create data analysis folder if not found
        if not os.path.exists(config['data_analysis_directory']):
            os.makedirs(config['data_analysis_directory'])

    def _plot_bar_graph_multicol(self, feature_name, title, data='trip'):
        """
        Plots a bar graph for a user-defined feature that is stored over multiple columns.

        Parameters:
            feature_name: str
                Contains the feature substring that can be found in the columns of interest.
            title: str
                Contains the title indicated on the bar graph.
            data: str
                Indicates whether we are interested in trip data or stop data
        """
        if data == 'trip':
            feature_columns = [column for column in self.trip_data.columns if feature_name in column]
            feature_data = self.trip_data[feature_columns]
        elif data == 'stop':
            feature_columns = [column for column in self.stop_data.columns if feature_name in column]
            feature_data = self.stop_data[feature_columns]
        else:
            raise ValueError('Data type {} is not supported. data accepts "stop" or "trip".'.format(data))

        feature_sum = feature_data.sum(axis=0)

        feature_sum.plot(kind='bar')
        feature_types = [column.replace('{}.'.format(feature_name), '') for column in feature_columns]
        plt.xticks(ticks=range(len(feature_types)), labels=feature_types)
        plt.ylabel('Frequency')
        plt.title(title)
        plt.show()

    def _plot_bar_graph_singlecol(self, column_name, title, data='trip'):
        """
        Plots a bar graph for a user-defined feature that is stored in one column.

        Parameters:
            column_name: str
                Contains the column name of the feature of interest.
            title: str
                Contains the title indicated on the bar graph.
            data: str
                Indicates whether we are interested in trip data or stop data
        """
        if data == 'trip':
            feature_type = self.trip_data[column_name].value_counts()
        elif data == 'stop':
            feature_type = self.stop_data[column_name].value_counts()
        else:
            raise ValueError('Data type {} is not supported. data accepts "stop" or "trip".'.format(data))
        feature_type.plot(kind='bar')
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
        self._plot_bar_graph_singlecol('VehicleType', 'Vehicle Type')

        # day of week breakdown
        self._plot_bar_graph_singlecol('DayOfWeekStr', 'Day of Week')

        # commodity type breakdown
        self._plot_bar_graph_multicol('Commodity', 'Commodity Type')

        # special cargo type breakdown
        self._plot_bar_graph_multicol('SpecialCargo', 'Special Cargo Type')

        # company type breakdown
        self._plot_bar_graph_multicol('Company.Type', 'Company Type')

        # industry type breakdown
        self._plot_bar_graph_multicol('Industry', 'Industry Type')

    def _extract_activity_type(self, series_row):
        """
        Extracts and concatenates the activities conducted at a particular stop as a single string

        Parameters:
            series_row: pandas.Series
                Contains the attributes of a stop.

        Return:
            activity_str: str
                Contains the activities conducted at a particular stop in the form of a string.
        """
        activity_list = [key.replace('Activity.', '')
                         for key, value in series_row.items()
                         if ('Activity.' in key) and (value == 1)]
        activity_str = ','.join(activity_list)
        return activity_str

    def calculate_stop_statistics(self):
        """
        Calculates different statistics related to stop data.
        """
        # calculate number of unique stops
        print('Number of unique stops: {}'.format(len(self.stop_data['StopID'].unique())))

        # place type breakdown
        self._plot_bar_graph_multicol('PlaceType', 'Place Type', data='stop')

        # activity type breakdown
        self._plot_bar_graph_multicol('Activity', 'Activity Type', data='stop')

        # activity type subset breakdown
        self.stop_data['ActivityType'] = self.stop_data.apply(self._extract_activity_type, axis=1)
        self.stop_data['ActivityType'].value_counts().to_excel(
            config['data_analysis_directory'] + 'activity_breakdown.xlsx',
            index=True)

        # stop duration distribution based on activity types
        filtered_stop_data = self.stop_data[self.stop_data['ActivityType'].isin(self.activity_types)]
        sns.boxplot(y='Duration', x='ActivityType', data=filtered_stop_data, showfliers=False)
        plt.ylabel('Stop Duration (s)')
        plt.xticks(rotation=90)
        plt.show()


if __name__ == '__main__':
    explorer = DataExplorer()
    # explorer.calculate_trip_statistics()
    explorer.calculate_stop_statistics()
    stop_data = explorer.stop_data
