import json
import pandas as pd
import os
import fiona
import geopandas as gpd
from pathlib import Path
from flatten_dict import flatten

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
        self.combined_trip_data = pd.DataFrame()
        self.combined_stop_data = pd.DataFrame()

    def _vehicle_type_mapping(self, vehicle_type):
        """
        Performs a vehicle type mapping to merge similar vehicle types together.

        Parameters:
            vehicle_type: str
                Contains the original vehicle type.

        Return:
            vehicle_mapping[vehicle_type]: str
                Contains the mapped vehicle type. Returns "Unknown" if vehicle_type is None.
        """
        if vehicle_type is None or vehicle_type == 'Nil':
            return "Unknown"

        vehicle_mapping = {"Truck": "Truck", "Truck-Other": "Truck",
                           "Single unit - Rigid/fenced Walls with open top": "Single Unit",
                           "Single unit - Rigid (>2-Axle)": "Single Unit",
                           "Single unit - Rigid (2-Axle)": "Single Unit",
                           "Single unit - Fully Refrigerated": "Single Unit",
                           "Single unit - No Walls": "Single Unit", "Bus": "Bus",
                           "Detachable Trailer - Flat Bed": "Tractor Trailer",
                           "Detachable Trailer - Tractor Trailer (Multiple)": "Tractor Trailer",
                           "Detachable Trailer - Tractor Trailer (Single)": "Tractor Trailer",
                           "Detachable Trailer - Fully Refrigerated": "Tractor Trailer",
                           "Truck-Detachable Trailer": "Tractor Trailer",
                           "Tractor Trailer (Single)": "Tractor Trailer",
                           "Other - Cement mixer": "Cement Mixer", "Other - Dump/Garbage Truck": "Dump/Garbage Truck",
                           "Other - Fuel/Gas Tank": "Fuel/Gas Tank", "Other": "Other",
                           "Truck-Other-Cement Mixer": "Cement Mixer", "Van, Sport Utility Vehicle (SUV)": "Van"}
        return vehicle_mapping[vehicle_type]

    def _load_verified_trips(self, batch_num):
        """
        Loads the verified trips data for a particular batch and removes the irrelevant columns.

        Parameters:
            batch_num: int
                Contains the batch number.

        Return:
            verified_trips: pandas.DataFrame
                Contains the verified trips information for a particular batch.
        """
        with open(os.path.join(os.path.dirname(__file__),
                               config['verified_stop_directory'].format(batch_num=batch_num))) as f:
            verified_trips = json.load(f)
        verified_trips = pd.json_normalize(verified_trips)

        # filter important features
        retained_columns = ['DriverID', 'VehicleType', 'Stops', 'Travels', 'YMD', 'Timeline', 'DayOfWeekStr']
        verified_trips = verified_trips[retained_columns]

        # perform mapping for vehicle type information
        verified_trips['VehicleType'] = verified_trips['VehicleType'].apply(self._vehicle_type_mapping)

        return verified_trips

    def _load_operation_survey(self, batch_num):
        """
        Loads the operation survey for a particular batch and removes the irrelevant columns.

        Parameters:
            batch_num: int
                Contains the batch number.

        Return:
            operation_data: pandas.DataFrame
                Contains the operation survey data for a particular batch.
        """
        # load operational survey
        with open(os.path.join(os.path.dirname(__file__),
                               config['operation_survey_directory'].format(batch_num=batch_num))) as f:
            operation_data = json.load(f)
        operation_data = pd.json_normalize(operation_data)

        # filter important features
        important_features = ['Commodity', 'SpecialCargo', 'Company.Type', 'Industry',
                              'Driver.ID']
        retained_columns = [column
                            for column in operation_data.columns
                            for feature in important_features
                            if feature in column]
        retained_columns.remove('Commodity.OtherStr')
        operation_data = operation_data[retained_columns]

        return operation_data

    def _generate_trip_id(self, verified_trips, batch_num):
        """
        Assigns a unique ID to each trip that contains the batch number as well.

        Parameters:
            verified_trips: pandas.DataFrame
                Contains the trip information for a particular batch.
            batch_num: int
                Contains the batch number.

        Return:
            verified_trips: pandas.DataFrame
                Contains the trip information for a particular batch with unique ID for each trip.
        """
        verified_trips = verified_trips.rename_axis('TripID').reset_index()
        verified_trips['TripID'] = 'B{}_'.format(batch_num) + verified_trips['TripID'].astype(str)

        return verified_trips

    def _process_timeline(self, timeline):
        """
        Process the timeline information of a particular trip to extract the stop information.

        Parameters:
            timeline: list of dictionaries
                Contains the stops made during a particular trip.

        Return:
            stops_df: pandas.DataFrame
                Contains the stops made during a particular trip, concatenated and formatted as a single Dataframe.
        """
        timeline_list = []
        for i in range(len(timeline)):
            for j in range(len(timeline.loc[i, 'Timeline'])):
                stop_dict = flatten(timeline.loc[i, 'Timeline'][j], reducer='dot')
                stop_dict['TripID'] = timeline.loc[i, 'TripID']
                timeline_list.append(stop_dict)

        # filter out stops and travel
        timeline_df = pd.DataFrame(timeline_list)
        stops_df = timeline_df[timeline_df['Type'] == 'Stop'].reset_index(drop=True)

        # drop redundant columns
        stops_df.rename(columns={'ID': 'StopID'}, inplace=True)
        interested_columns = ['Attribute.PlaceType.', 'Attribute.Address', 'Attribute.StopLon', 'Attribute.StopLat',
                              'Attribute.Activity.', 'StartTime', 'EndTime', 'Duration', 'StopID', 'TripID']
        retained_columns = [column
                            for column in stops_df.columns
                            for interested_column in interested_columns
                            if interested_column in column]
        retained_columns.remove('Attribute.PlaceType.Applicable')
        retained_columns.remove('Attribute.Activity.OtherStr')
        stops_df = stops_df[retained_columns]

        # remove 'Attribute_' from column name
        stops_df.columns = [col_name.replace('Attribute.', '') for col_name in stops_df.columns]

        return stops_df

    def _extract_verified_stops(self, verified_trips, batch_num):
        """
        Extracts the verified stop information based on the verified trips.

        Parameters:
            verified_trips: pandas.DataFrame
                Contains the verified trip information for a particular batch.
            batch_num: int
                Contains the batch number.

        Return:
            verified_stops: pandas.DataFrame
                Contains the verified stop information for a particular batch.
        """
        # extract stop information and frequent places
        verified_trips = self._generate_trip_id(verified_trips, batch_num)
        timeline = verified_trips[['Timeline', 'TripID']]
        other_trip_info = verified_trips.drop(columns=['Timeline'])
        timeline_info = self._process_timeline(timeline)

        # merge with other trip information
        verified_stops = timeline_info.merge(other_trip_info, how='left', on='TripID')

        return verified_stops

    def _remove_bus_data(self, trip_data, stop_data):
        """
        Removes all trip and stop data collected for buses.

        Parameters:
            trip_data: pandas.DataFrame
                Contains the trip data for a particular batch.
            stop_data: pandas.DataFrame
                Contains the stop data for a particular batch.

        Return:
            filtered_trip_data: pandas.DataFrame
                Contains the filtered trip data for a particular batch without any bus-related trips.
            filtered_stop_data: pandas.DataFrame
                Contains the filtered stop data for a particular batch without any bus-related stops.
        """
        filtered_trip_data = trip_data[trip_data['VehicleType'] != 'Bus']
        filtered_stop_data = stop_data[stop_data['VehicleType'] != 'Bus']
        return filtered_trip_data, filtered_stop_data

    def _load_landuse_data(self):
        """"
        Loads the URA 2019 land use data.

        Return:
            landuse_data: pd.DataFrame
                Contains the land use information from URA.
        """
        fiona.drvsupport.supported_drivers['KML'] = 'rw'
        landuse_data = gpd.read_file(os.path.join(os.path.dirname(__file__), config['ura_landuse']),
                                     driver='KML')
        landuse_data['landuse_type'] = landuse_data['Description'].apply(lambda x: pd.read_html(x)[0].loc[0, 'Attributes.1'])
        landuse_data.drop(columns=['Name', 'Description'], inplace=True)

        return landuse_data

    def perform_feature_extraction(self):
        # perform train test split
        #TODO

        # feature extraction for stop activities by other drivers
        # TODO


        # feature extraction for past stop activities
        # TODO
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

        # extract verified stop information
        verified_stops = self._extract_verified_stops(verified_trips, batch_num)

        # import operation survey data
        operation_data = self._load_operation_survey(batch_num)

        # import URA land use data
        landuse_data = self._load_landuse_data()

        # merge trip data
        batch_trip_data = verified_trips.merge(operation_data, how='left',
                                               right_on='Driver.ID', left_on='DriverID')
        batch_trip_data.drop(columns=['Driver.ID'], inplace=True)

        # merge stop data with operation data and land use data
        batch_stop_data = verified_stops.merge(operation_data, how='left',
                                               right_on='Driver.ID', left_on='DriverID')
        batch_stop_data.drop(columns=['Driver.ID'], inplace=True)
        batch_stop_data = gpd.GeoDataFrame(batch_stop_data,
                                           geometry=gpd.points_from_xy(batch_stop_data['StopLon'],
                                                                       batch_stop_data['StopLat']),
                                           crs=4326)
        batch_stop_data = gpd.sjoin(batch_stop_data, landuse_data, how="left", op='intersects')
        batch_stop_data.drop(columns=['index_right'], inplace=True)

        # remove stop and trip data related to buses
        batch_trip_data, batch_stop_data = self._remove_bus_data(batch_trip_data, batch_stop_data)

        # load POI data
        # TODO

        # store processed batch data and combined data locally
        if not os.path.exists(os.path.join(os.path.dirname(__file__), config['processed_data_directory'])):
            os.makedirs(os.path.join(os.path.dirname(__file__), config['processed_data_directory']))

        batch_trip_data.to_excel(os.path.join(os.path.dirname(__file__),
                                              config['processed_data_directory'] +
                                              'batch_trip_data_{}.xlsx'.format(batch_num)),
                                 index=False)
        batch_stop_data.to_excel(os.path.join(os.path.dirname(__file__),
                                              config['processed_data_directory'] +
                                              'batch_stop_data_{}.xlsx'.format(batch_num)),
                                 index=False)
        self.combined_trip_data = pd.concat([self.combined_trip_data, batch_trip_data], ignore_index=True)
        self.combined_trip_data.to_excel(os.path.join(os.path.dirname(__file__),
                                                      config['processed_data_directory'] + 'combined_trip_data.xlsx'),
                                         index=False)
        self.combined_stop_data = pd.concat([self.combined_stop_data, batch_stop_data], ignore_index=True)
        self.combined_stop_data.to_excel(os.path.join(os.path.dirname(__file__),
                                                      config['processed_data_directory'] + 'combined_stop_data.xlsx'),
                                         index=False)
        return batch_stop_data


if __name__ == '__main__':
    processor = DataProcessor()
    batch_stop_data = processor.process_data(batch_num=1)
    # processor.process_data(batch_num=1)
    # processor.process_data(batch_num=2)
    # processor.process_data(batch_num=3)
    # processor.process_data(batch_num=4)
    # processor.process_data(batch_num=5)
    # processor.process_data(batch_num=6)
    # processor.process_data(batch_num=7)
    # processor.process_data(batch_num=8)
    # processor.perform_feature_extraction()
