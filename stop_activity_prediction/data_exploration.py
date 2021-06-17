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
        trip_data = pd.read_excel(config['processed_data_directory'] + 'combined_data.xlsx')
        self.trip_data = trip_data

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
        commodity_columns = [column for column in self.trip_data.columns if 'Commodity' in column]
        commodity_columns.remove('Commodity.OtherStr')
        commodity_data = self.trip_data[commodity_columns]
        commodity_sum = commodity_data.sum(axis=0)

        commodity_sum.plot(kind='bar')
        commodity_types = [column.replace('Commodity.', '') for column in commodity_columns]
        plt.xticks(ticks=range(len(commodity_types)), labels=commodity_types)
        plt.ylabel('Frequency')
        plt.title('Commodity Type')
        plt.show()

        # special cargo type breakdown
        specialcargo_columns = [column for column in self.trip_data.columns if 'SpecialCargo' in column]
        specialcargo_data = self.trip_data[specialcargo_columns]
        specialcargo_sum = specialcargo_data.sum(axis=0)

        specialcargo_sum.plot(kind='bar')
        specialcargo_types = [column.replace('SpecialCargo.', '') for column in specialcargo_columns]
        plt.xticks(ticks=range(len(specialcargo_types)), labels=specialcargo_types)
        plt.ylabel('Frequency')
        plt.title('Special Cargo Type')
        plt.show()

        # company type breakdown
        company_columns = [column for column in self.trip_data.columns if 'Company.Type' in column]
        company_data = self.trip_data[company_columns]
        company_sum = company_data.sum(axis=0)

        company_sum.plot(kind='bar')
        company_types = [column.replace('Company.Type.', '') for column in company_columns]
        plt.xticks(ticks=range(len(company_types)), labels=company_types)
        plt.ylabel('Frequency')
        plt.title('Company Type')
        plt.show()

        # industry type breakdown
        industry_columns = [column for column in self.trip_data.columns if 'Industry' in column]
        industry_data = self.trip_data[industry_columns]
        industry_sum = industry_data.sum(axis=0)

        industry_sum.plot(kind='bar')
        industry_types = [column.replace('Industry.', '') for column in industry_columns]
        plt.xticks(ticks=range(len(industry_types)), labels=industry_types)
        plt.ylabel('Frequency')
        plt.title('Industry Type')
        plt.show()


if __name__ == '__main__':
    explorer = DataExplorer()
    explorer.calculate_trip_statistics()
