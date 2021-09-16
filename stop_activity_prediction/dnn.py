import os
import json
from load_data import DataLoader
from joblib import dump, load
from sklearn.metrics import accuracy_score, classification_report, f1_score, hamming_loss, \
    jaccard_score, precision_recall_fscore_support, roc_auc_score, zero_one_loss


# load config file
with open(os.path.join(os.path.dirname(__file__), '../config.json')) as f:
    config = json.load(f)

class DNN:
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

    def train(self, algorithm=None, classifier_chain=True):
        """
        Trains a model on the training dataset using a user-defined ML algorithm supported by sklearn.

        Parameters:
            algorithm: str
                Indicates the name of the algorithm used to train the model.
            classifier_chain: bool
                Indicates whether the problem will be transformed into a classifier chain
        """
        # initialise model

        # fit model on training data

        # save model
        if not os.path.exists(os.path.join(os.path.dirname(__file__), config['activity_models_directory'])):
            os.makedirs(os.path.join(os.path.dirname(__file__), config['activity_models_directory']))

        return None

    def evaluate(self, algorithm=None):
        """
        Evaluates the performance of the trained model based on test dataset.

        Parameters:
            algorithm: str
                Indicates the name of the algorithm used to train the model.
        """
        # load model
        self.model = load(os.path.join(os.path.dirname(__file__),
                                       config['activity_models_directory'] +
                                       'model_{}.joblib'.format(algorithm)))

        # perform inference on test set
        test_pred = self.model.predict(self.test_x)

        # generate evaluation scores
        print('algorithm: {}'.format(algorithm))
        print('classes: {}'.format(self.model.classes_))
        print('accuracy: {}'.format(accuracy_score(self.test_y, test_pred)))
        print('f1 score: {}'.format(f1_score(self.test_y, test_pred, average=None)))
        print('hamming loss: {}'.format(hamming_loss(self.test_y, test_pred)))
        print('jaccard score: {}'.format(jaccard_score(self.test_y, test_pred, average=None)))
        print('roc auc score: {}'.format(roc_auc_score(self.test_y, test_pred)))
        print('zero one loss: {}'.format(zero_one_loss(self.test_y, test_pred)))
        print('precision recall fscore report: {}'.format(precision_recall_fscore_support(self.test_y, test_pred,
                                                                                          average=None)))
        print('classification report: {}'.format(classification_report(self.test_y, test_pred)))
        print()
        return None


if __name__ == '__main__':
    model = DNN()

    # train and evaluate model performance
    model.train()
    model.evaluate()
