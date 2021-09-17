import os
import json
import pandas as pd
import scipy
from load_data import DataLoader
from joblib import dump, load
from sklearn.metrics import accuracy_score, classification_report, hamming_loss, \
    jaccard_score, roc_auc_score, zero_one_loss
from skmultilearn.problem_transform import ClassifierChain
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


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
        feature_cols = [col for col in self.train_data.columns
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

    def _initialise_model(self, algorithm=None):
        """
        Initialise the corresponding class object based on the defined algorithm.

        Parameters:
            algorithm: str
                Indicates the name of the algorithm used to train the model.
        Returns:
            model: sklearn object
                The sklearn model object corresponding to the user-defined algorithm.
        """
        if algorithm == "RandomForest":
            model = RandomForestClassifier(class_weight='balanced')
        elif algorithm == "ExtraTrees":
            model = ExtraTreesClassifier(class_weight='balanced')
        elif algorithm == "KNN":
            model = KNeighborsClassifier()
        elif algorithm == "GradientBoost":
            model = GradientBoostingClassifier()
        elif algorithm == "AdaBoost":
            model = AdaBoostClassifier()
        elif algorithm == "MultinomialLogit":
            model = LogisticRegression()
        else:
            raise ValueError('{} is not supported.'.format(algorithm))

        return model

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
        model = self._initialise_model(algorithm)

        if classifier_chain:
            model = ClassifierChain(model, require_dense=True)

        # fit model on training data
        model.fit(self.train_x, self.train_y)

        # save model
        if not os.path.exists(os.path.join(os.path.dirname(__file__), config['activity_models_directory'])):
            os.makedirs(os.path.join(os.path.dirname(__file__), config['activity_models_directory']))

        dump(model, os.path.join(os.path.dirname(__file__),
                                 config['activity_models_directory'] +
                                 'model_{}.joblib'.format(algorithm)))
        return None

    def evaluate(self, algorithm=None, classifier_chain=False):
        """
        Evaluates the performance of the trained model based on test dataset.

        Parameters:
            algorithm: str
                Indicates the name of the algorithm used to train the model.
            classifier_chain: bool
                Indicates whether the problem will be transformed into a classifier chain
        """
        # load model
        self.model = load(os.path.join(os.path.dirname(__file__),
                                       config['activity_models_directory'] +
                                       'model_{}.joblib'.format(algorithm)))

        # perform inference on test set
        test_pred = self.model.predict(self.test_x)

        # generate evaluation scores
        print('Algorithm: {}'.format(algorithm))
        activity_labels = [col.replace('MappedActivity.', '') for col in self.train_y.columns]
        if classifier_chain:
            test_pred = pd.DataFrame.sparse.from_spmatrix(test_pred, columns=self.train_y.columns).astype(int)
        else:
            test_pred = pd.DataFrame(test_pred, columns=self.train_y.columns)

        print('Classes: {}'.format(activity_labels))
        print('Accuracy: {}'.format(accuracy_score(self.test_y, test_pred)))
        print('Hamming Loss: {}'.format(hamming_loss(self.test_y, test_pred)))
        print('Jaccard Score')
        print(jaccard_score(self.test_y, test_pred, average=None))
        print('ROC AUC Score')
        print(roc_auc_score(self.test_y.values, test_pred.values, average=None))
        print('Zero One Loss: {}'.format(zero_one_loss(self.test_y, test_pred)))
        print('Classification Report:')
        print(classification_report(self.test_y, test_pred, target_names=activity_labels, zero_division=0))
        print()
        return None


if __name__ == '__main__':
    model = MLModel()
    train_data = model.train_data
    test_data = model.test_data
    train_x = model.train_x
    train_y = model.train_y
    test_x = model.test_x
    test_y = model.test_y

    # random forest
    model.train(algorithm='RandomForest', classifier_chain=False)
    model.evaluate(algorithm='RandomForest', classifier_chain=False)

    # extra trees
    model.train(algorithm='ExtraTrees', classifier_chain=False)
    model.evaluate(algorithm='ExtraTrees', classifier_chain=False)

    # KNN
    model.train(algorithm='KNN', classifier_chain=False)
    model.evaluate(algorithm='KNN', classifier_chain=False)

    # gradient boosting
    model.train(algorithm='GradientBoost', classifier_chain=True)
    model.evaluate(algorithm='GradientBoost', classifier_chain=True)

    # decision tree with adaptive boosting
    model.train(algorithm='AdaBoost', classifier_chain=True)
    model.evaluate(algorithm='AdaBoost', classifier_chain=True)

    # multinomial logit model
    model.train(algorithm='MultinomialLogit', classifier_chain=True)
    model.evaluate(algorithm='MultinomialLogit', classifier_chain=True)
