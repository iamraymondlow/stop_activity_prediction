import os
import json
import glob
from load_data import DataLoader
from joblib import dump, load


# load config file
with open(os.path.join(os.path.dirname(__file__), '../config.json')) as f:
    config = json.load(f)

class MLModel:
    """
    Trains and evaluates the performance of traditional ML models based on the training and test dataset.
    """
    def __init__(self):
        """
        Initialises the model object by loading the training and test datasets.
        """
        loader = DataLoader()
        self.train_data, self.test_data = loader.train_test_split(test_ratio=0.25)

        # define features of interest
        features = ['DriverID', 'Duration', 'StartHour', 'DayOfWeek.', 'PlaceType.', 'Commodity.',
                    'SpecialCargo.', 'Company.Type.', 'Industry.', 'VehicleType.', 'NumPOIs', 'POI.',
                    'LandUse.', 'Other.MappedActivity.', 'Past.MappedActivity.']
        feature_cols = [col
                        for col in self.train_data.columns
                        for feature in features
                        if feature in col]
        # original activity types
        # activity_cols = ['Activity.PickupTrailer', 'Activity.Passenger', 'Activity.Fueling', 'Activity.OtherWork',
        #                  'Activity.DropoffTrailer', 'Activity.Resting', 'Activity.Personal', 'Activity.Shift',
        #                  'Activity.ProvideService', 'Activity.DropoffContainer', 'Activity.Queuing', 'Activity.Other',
        #                  'Activity.DeliverCargo', 'Activity.Maintenance', 'Activity.Fail', 'Activity.PickupCargo',
        #                  'Activity.Meal', 'Activity.PickupContainer']
        # mapped activity types
        activity_cols = ['MappedActivity.DeliverCargo', 'MappedActivity.PickupCargo', 'MappedActivity.Other',
                         'MappedActivity.Shift', 'MappedActivity.Break', 'MappedActivity.DropoffTrailerContainer',
                         'MappedActivity.PickupTrailerContainer', 'MappedActivity.Maintenance']

        self.train_x = self.train_data[feature_cols]
        self.train_y = self.train_data[activity_cols]
        self.test_x = self.test_data[feature_cols]
        self.test_y = self.test_data[activity_cols]
        self.model = None

    def train(self, algorithm=None):
        """
        Trains a model on the training dataset using a user-defined ML algorithm supported by sklearn.

        Parameters:
            algorithm: str
                Indicates the name of the algorithm used to train the model.
        """
        # train model
        model = None
        # gb_models = self._train(train_datasets, 'GB')
        # rf_models = self._train(train_datasets, 'RF')
        # xgboost_models = self._train(train_datasets, 'XGB')

        # save model
        if not os.path.exists(os.path.join(os.path.dirname(__file__), config['activity_models_directory'])):
            os.makedirs(os.path.join(os.path.dirname(__file__), config['activity_models_directory']))

        dump(model, os.path.join(os.path.dirname(__file__),
                                 config['activity_models_directory'] +
                                 'model_{}.joblib'.format(algorithm)))
        return None

    def evaluate(self, algorithm=None):
        """
        Evaluates the performnace of the trained model based on test dataset.

        Parameters:
            algorithm: str
                Indicates the name of the algorithm used to train the model.
        """
        # load model
        self.model = load(os.path.join(os.path.dirname(__file__),
                                       config['activity_models_directory'] +
                                       'model_{}.joblib'.format(algorithm)))

        # evaluate model based on test set


        # evaluate model performance on hold out set
        # print('Perform model evaluation...')
        # y_pred_gb = self._predict(gb_models, test_data[['address_similarity', 'address_str_similarity',
        #                                                 'name_similarity']])
        # y_pred_rf = self._predict(rf_models, test_data[['address_similarity', 'address_str_similarity',
        #                                                 'name_similarity']])
        # y_pred_xgboost = self._predict(xgboost_models, test_data[['address_similarity', 'address_str_similarity',
        #                                                           'name_similarity']])
        # self._evaluate(test_data['label'], y_pred_gb, 'Gradient Boosting')
        # self._evaluate(test_data['label'], y_pred_rf, 'Random Forest')
        # self._evaluate(test_data['label'], y_pred_xgboost, 'XGBoost')

        return None


if __name__ == '__main__':
    model = MLModel()
    train_data = model.train_data
    test_data = model.test_data
    train_x = model.train_x
    train_y = model.train_y
    test_x = model.test_x
    test_y = model.test_y

    # # gradient boosting
    # model.train(algorithm='GB')
    # model.evaluate(algorithm='GB')
    #
    # # decision tree with adaptive boosting
    # model.train(algorithm='AB')
    # model.evaluate(algorithm='AB')
    #
    # # random forest
    # model.train(algorithm='RF')
    # model.evaluate(algorithm='RF')
    #
    # # nested logit
    # model.train(algorithm='NL')
    # model.evaluate(algorithm='NL')
    #
    # # multinomial logit model
    # model.train(algorithm='ML')
    # model.evaluate(algorithm='ML')
