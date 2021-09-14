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
        # load stop data
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
        self.stop_data = pd.concat([batch1, batch2, batch3, batch4, batch5, batch6, batch7, batch8], ignore_index=True)

        # load trip data
        trip1 = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                            config['processed_data_directory'] + 'batch_stop_data_1.xlsx'))
        trip2 = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                            config['processed_data_directory'] + 'batch_stop_data_2.xlsx'))
        trip3 = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                            config['processed_data_directory'] + 'batch_stop_data_3.xlsx'))
        trip4 = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                            config['processed_data_directory'] + 'batch_stop_data_4.xlsx'))
        trip5 = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                            config['processed_data_directory'] + 'batch_stop_data_5.xlsx'))
        trip6 = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                            config['processed_data_directory'] + 'batch_stop_data_6.xlsx'))
        trip7 = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                            config['processed_data_directory'] + 'batch_stop_data_7.xlsx'))
        trip8 = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                            config['processed_data_directory'] + 'batch_stop_data_8.xlsx'))
        self.trip_data = pd.concat([trip1, trip2, trip3, trip4, trip5, trip6, trip7, trip8], ignore_index=True)

        self.mapped_activity_types = ['DeliverCargo', 'PickupCargo', 'Other', 'Shift', 'Break',
                                      'DropoffTrailerContainer', 'PickupTrailerContainer', 'Maintenance']
        self.activity_types = ['DeliverCargo', 'PickupCargo', 'Other', 'Shift', 'ProvideService',
                                      'OtherWork', 'Meal', 'DropoffTrailer', 'PickupTrailer', 'Fueling',
                                      'Personal', 'Passenger', 'Resting', 'Queuing', 'DropoffContainer',
                                      'PickupContainer', 'Fail', 'Maintenance']
        self.place_types = ['ContainerYard', 'DistributionCenter', 'Natural', 'IntermediateStorage',
                            'Facility', 'Residence', 'Park', 'Headquarter', 'Warehouse', 'Construction',
                            'Unknown', 'Transfer', 'Factory', 'Retail']
        self.poi_types = ['emergency_services', 'government', 'Unknown', 'food', 'lodging', 'recreation',
                          'religion', 'retail', 'parking', 'services', 'healthcare', 'school', 'gas_station',
                          'cemetry', 'transport']
        self.colours = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
                        "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2",
                        "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]

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
        activity_list = [key.replace('MappedActivity.', '')
                         for key, value in series_row.items()
                         if ('MappedActivity.' in key) and (value == 1)]
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
        self._plot_bar_graph_multicol('MappedActivity', 'Activity Type', data='stop')

        # activity type subset breakdown
        self.stop_data['ActivityType'] = self.stop_data.apply(self._extract_activity_type, axis=1)
        self.stop_data['ActivityType'].value_counts().to_excel(
            config['data_analysis_directory'] + 'activity_breakdown.xlsx',
            index=True)

        # stop duration distribution based on activity types
        filtered_stop_data = self.stop_data[self.stop_data['ActivityType'].isin(self.mapped_activity_types)]
        sns.boxplot(y='Duration', x='ActivityType', data=filtered_stop_data, showfliers=False)
        plt.ylabel('Stop Duration (s)')
        plt.xticks(rotation=90)
        plt.show()

    def plot_activity_starttime(self):
        """
        Plots the distribution of each activity type based on start time.
        """
        activity_df = pd.DataFrame()
        for activity_type in self.mapped_activity_types:
            filtered_data = self.stop_data[
                self.stop_data['MappedActivity.{}'.format(activity_type)] == 1
                ].reset_index(drop=True)
            starthour_count = filtered_data['StartHour'].value_counts()
            normalised_starthour_count = (starthour_count * 100) / (starthour_count.sum() + 1e-9)
            activity_df = activity_df.append(normalised_starthour_count.T, ignore_index=True)
        activity_df.index = self.mapped_activity_types
        activity_df = activity_df.fillna(0).reset_index(drop=False)
        activity_df_transp = activity_df.set_index('index').T

        # plot graph
        activity_df_transp.plot(figsize=(12, 10), color=self.colours, lw=3)
        # title, legend, labels
        plt.title('Temporal Distribution of each Activity Type\n')
        plt.legend(self.mapped_activity_types, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6)
        plt.xticks(ticks=range(0, 24), labels=[str(hour)+'00'
                                               if len(str(hour)) == 2 else '0{}00'.format(hour)
                                               for hour in range(0, 24)])
        plt.xlabel('Start Hour')
        plt.ylabel('Percentage')
        plt.show()

    def plot_activity_dayofweek(self):
        """
        Plots the frequency of each activity conducted based on the day of week.
        """
        fields = ['MappedActivity.{}'.format(activity_type) for activity_type in self.mapped_activity_types]
        grouped_activity = self.stop_data.groupby('DayOfWeekStr').sum()[fields]
        grouped_activity = grouped_activity.reindex(['Sunday', 'Saturday', 'Friday', 'Thursday',
                                                     'Wednesday', 'Tuesday', 'Monday'])
        labels = self.mapped_activity_types

        # figure and axis
        fig, ax = plt.subplots(1, figsize=(12, 10))
        # plot bars
        left = len(grouped_activity) * [0]
        for idx, name in enumerate(fields):
            plt.barh(grouped_activity.index, grouped_activity[name], left=left, color=self.colours[idx])
            left = left + grouped_activity[name]
        # title, legend, labels
        plt.title('Activity Frequency vs Day of Week\n')
        plt.legend(labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6)
        plt.xlabel('Frequency')
        # adjust limits and draw grid lines
        plt.ylim(-0.5, ax.get_yticks()[-1] + 0.5)
        ax.set_axisbelow(True)
        ax.xaxis.grid(color='gray', linestyle='dashed')
        plt.show()

    def plot_activity_placetype(self):
        """
        Plots the distribution of each place type based on the activity conducted.
        """
        # remove duplicated stops
        stop_data = self.stop_data.drop_duplicates(subset=['StopID']).reset_index(drop=True)

        fields = ['PlaceType.{}'.format(place_type) for place_type in self.place_types]
        activity_df = pd.DataFrame()
        for activity_type in self.mapped_activity_types:
            filtered_data = stop_data[stop_data['MappedActivity.{}'.format(activity_type)] == 1].reset_index(drop=True)
            summed_placetype = filtered_data.sum()[fields]
            normalised_placetype = (summed_placetype * 100) / (summed_placetype.sum() + 1e-9)
            activity_df = activity_df.append(normalised_placetype.T, ignore_index=True)
        activity_df.index = self.mapped_activity_types

        # figure and axis
        fig, ax = plt.subplots(1, figsize=(12, 10))
        # plot bars
        left = len(activity_df) * [0]
        for idx, name in enumerate(fields):
            plt.barh(activity_df.index, activity_df[name], left=left, color=self.colours[idx])
            left = left + activity_df[name]
        # title, legend, labels
        plt.title('Place Type Distribution vs Activity Type\n')
        plt.legend(self.place_types, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6)
        plt.xlabel('Percentage')
        # adjust limits and draw grid lines
        plt.ylim(-0.5, ax.get_yticks()[-1] + 0.5)
        ax.set_axisbelow(True)
        ax.xaxis.grid(color='gray', linestyle='dashed')
        plt.show()

    def plot_activity_landuse(self):
        """
        Plots the distribution of each land use type based on the activity conducted.
        """
        # remove duplicated stops
        stop_data = self.stop_data.drop_duplicates(subset=['StopID']).reset_index(drop=True)

        activity_df = pd.DataFrame()
        for activity_type in self.mapped_activity_types:
            filtered_data = stop_data[stop_data['MappedActivity.{}'.format(activity_type)] == 1].reset_index(drop=True)
            landuse_count = filtered_data['MappedLandUseType'].value_counts()
            normalised_landuse_count = (landuse_count * 100) / (landuse_count.sum() + 1e-9)
            activity_df = activity_df.append(normalised_landuse_count.T, ignore_index=True)
        activity_df.index = self.mapped_activity_types
        activity_df.fillna(0, inplace=True)

        # figure and axis
        fig, ax = plt.subplots(1, figsize=(12, 10))
        # plot bars
        left = len(activity_df) * [0]
        for idx, name in enumerate(activity_df.columns):
            plt.barh(activity_df.index, activity_df[name], left=left, color=self.colours[idx])
            left = left + activity_df[name]
        # title, legend, labels
        plt.title('Land Use Type Distribution vs Activity Type\n')
        plt.legend(activity_df.columns, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
        plt.xlabel('Percentage')
        # adjust limits and draw grid lines
        plt.ylim(-0.5, ax.get_yticks()[-1] + 0.5)
        ax.set_axisbelow(True)
        ax.xaxis.grid(color='gray', linestyle='dashed')
        plt.show()

    def plot_activity_vehicletype(self):
        """
        Plots the distribution of each activity type based on vehicle type.
        """
        # remove duplicated stops
        stop_data = self.stop_data.drop_duplicates(subset=['StopID']).reset_index(drop=True)

        fields = ['MappedActivity.{}'.format(activity_type) for activity_type in self.mapped_activity_types]
        grouped_activity = stop_data.groupby('VehicleType').sum()[fields]
        grouped_activity = grouped_activity.div(grouped_activity.sum(axis=1), axis=0)
        labels = self.mapped_activity_types

        # figure and axis
        fig, ax = plt.subplots(1, figsize=(12, 10))
        # plot bars
        left = len(grouped_activity) * [0]
        for idx, name in enumerate(fields):
            plt.barh(grouped_activity.index, grouped_activity[name], left=left, color=self.colours[idx])
            left = left + grouped_activity[name]
        # title, legend, labels
        plt.title('Activity Frequency vs Vehicle Type\n')
        plt.legend(labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6)
        plt.xlabel('Percentage')
        # adjust limits and draw grid lines
        plt.ylim(-0.5, ax.get_yticks()[-1] + 0.5)
        ax.set_axisbelow(True)
        ax.xaxis.grid(color='gray', linestyle='dashed')
        plt.show()

    def plot_activity_poitype(self):
        """
        Plots the distribution of each POI place type based on the activity conducted.
        """
        # remove duplicated stops
        stop_data = self.stop_data.drop_duplicates(subset=['StopID']).reset_index(drop=True)

        fields = ['POI.{}'.format(poi_type) for poi_type in self.poi_types]
        activity_df = pd.DataFrame()
        for activity_type in self.mapped_activity_types:
            filtered_data = stop_data[stop_data['MappedActivity.{}'.format(activity_type)] == 1].reset_index(drop=True)
            summed_poitype = filtered_data.sum()[fields]
            normalised_poitype = (summed_poitype * 100) / (summed_poitype.sum() + 1e-9)
            activity_df = activity_df.append(normalised_poitype.T, ignore_index=True)
        activity_df.index = self.mapped_activity_types

        # figure and axis
        fig, ax = plt.subplots(1, figsize=(12, 10))
        # plot bars
        left = len(activity_df) * [0]
        for idx, name in enumerate(fields):
            plt.barh(activity_df.index, activity_df[name], left=left, color=self.colours[idx])
            left = left + activity_df[name]
        # title, legend, labels
        plt.title('POI Type Distribution vs Activity Type\n')
        labels = [poi_type.replace('_', ' ').title() for poi_type in self.poi_types]
        plt.legend(labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6)
        plt.xlabel('Percentage')
        # adjust limits and draw grid lines
        plt.ylim(-0.5, ax.get_yticks()[-1] + 0.5)
        ax.set_axisbelow(True)
        ax.xaxis.grid(color='gray', linestyle='dashed')
        plt.show()


if __name__ == '__main__':
    explorer = DataExplorer()
    stop_data = explorer.stop_data
    explorer.calculate_trip_statistics()
    explorer.calculate_stop_statistics()
    explorer.plot_activity_dayofweek()
    explorer.plot_activity_placetype()
    explorer.plot_activity_landuse()
    explorer.plot_activity_starttime()
    explorer.plot_activity_vehicletype()
    explorer.plot_activity_poitype()

