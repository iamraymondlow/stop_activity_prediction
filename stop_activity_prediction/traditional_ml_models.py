import os
import json
import pandas as pd
import argparse
from load_data import DataLoader
from joblib import dump, load
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             hamming_loss,
                             jaccard_score,
                             roc_auc_score,
                             zero_one_loss)
from skmultilearn.problem_transform import ClassifierChain
from sklearn.ensemble import (ExtraTreesClassifier,
                              RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


parser = argparse.ArgumentParser()
parser.add_argument("--INCLUDE_DURATION", type=bool, default=True)
parser.add_argument("--INCLUDE_STARTHOUR", type=bool, default=True)
parser.add_argument("--INCLUDE_DAYOFWEEK", type=bool, default=True)
parser.add_argument("--INCLUDE_PLACETYPE", type=bool, default=True)
parser.add_argument("--INCLUDE_CARGOTYPE", type=bool, default=True)
parser.add_argument("--INCLUDE_COMPANYINFO", type=bool, default=True)
parser.add_argument("--INCLUDE_VEHICLETYPE", type=bool, default=True)
parser.add_argument("--INCLUDE_POI", type=bool, default=True)
parser.add_argument("--INCLUDE_URALANDUSE", type=bool, default=True)
parser.add_argument("--INCLUDE_OTHERACTIVITY", type=bool, default=True)
parser.add_argument("--INCLUDE_PASTACTIVITY", type=bool, default=True)
parser.add_argument("--INCLUDE_LASTACTIVITY", type=bool, default=True)
args = parser.parse_args()


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
        # load train and test datasets
        loader = DataLoader()
        self.train_data, self.test_data = loader.train_test_split(test_ratio=0.25)

        # merge certain activity types  #TODO can be removed in the future if data processing code is rerun
        self.train_data["MappedActivity.DropoffPickupTrailerContainer"] = self.train_data[
                                                                              "MappedActivity.DropoffTrailerContainer"] + \
                                                                          self.train_data[
                                                                              "MappedActivity.PickupTrailerContainer"]
        self.test_data["MappedActivity.DropoffPickupTrailerContainer"] = self.test_data[
                                                                             "MappedActivity.DropoffTrailerContainer"] + \
                                                                         self.test_data[
                                                                             "MappedActivity.PickupTrailerContainer"]
        self.train_data["MappedActivity.DeliverPickupCargo"] = self.train_data["MappedActivity.DeliverCargo"] + \
                                                               self.train_data["MappedActivity.PickupCargo"]
        self.test_data["MappedActivity.DeliverPickupCargo"] = self.test_data["MappedActivity.DeliverCargo"] + \
                                                              self.test_data["MappedActivity.PickupCargo"]

        self.train_data.loc[self.train_data["MappedActivity.DropoffPickupTrailerContainer"] > 0,
                            'MappedActivity.DropoffPickupTrailerContainer'] = 1
        self.test_data.loc[self.test_data["MappedActivity.DropoffPickupTrailerContainer"] > 0,
                           'MappedActivity.DropoffPickupTrailerContainer'] = 1
        self.train_data.loc[self.train_data["MappedActivity.DeliverPickupCargo"] > 0,
                            'MappedActivity.DeliverPickupCargo'] = 1
        self.test_data.loc[self.test_data["MappedActivity.DeliverPickupCargo"] > 0,
                           'MappedActivity.DeliverPickupCargo'] = 1

        # define features that will be passed into model
        self.features = []

        if args.INCLUDE_DURATION:
            self.features.extend(["Duration"])

        if args.INCLUDE_STARTHOUR:
            self.features.extend(["StartHour"])

        if args.INCLUDE_DAYOFWEEK:
            self.features.extend(["DayOfWeek."])

        if args.INCLUDE_PLACETYPE:
            self.features.extend(["PlaceType."])

        if args.INCLUDE_CARGOTYPE:
            self.features.extend(["Commodity.", "SpecialCargo."])

        if args.INCLUDE_COMPANYINFO:
            self.features.extend(["Company.Type.", "Industry."])

        if args.INCLUDE_VEHICLETYPE:
            self.features.extend(["VehicleType."])

        if args.INCLUDE_POI:
            self.features.extend(["NumPOIs", "POI."])

        if args.INCLUDE_URALANDUSE:
            self.features.extend(["LandUse."])

        if args.INCLUDE_OTHERACTIVITY:
            self.features.extend(["Other.MappedActivity."])

        if args.INCLUDE_PASTACTIVITY:
            self.features.extend(["Past.MappedActivity."])

        if args.INCLUDE_LASTACTIVITY:
            self.features.extend(["LastActivity."])

        self.feature_cols = [col for col in self.train_data.columns
                             for feature in self.features
                             if feature == col[:len(feature)]]

        # mapped activity types
        self.activity_cols = ['MappedActivity.DeliverPickupCargo', 'MappedActivity.Other', 'MappedActivity.Shift',
                              'MappedActivity.Break', 'MappedActivity.DropoffPickupTrailerContainer',
                              'MappedActivity.Maintenance']

        self.train_x = self.train_data[self.feature_cols]
        self.train_y = self.train_data[self.activity_cols]
        self.test_x = self.test_data[self.feature_cols]
        self.test_y = self.test_data[self.activity_cols]
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
        self.model = model
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
        if self.model is None:
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
        print(jaccard_score(self.test_y, test_pred, average='macro'))
        print('ROC AUC Score')
        print(roc_auc_score(self.test_y.values, test_pred.values))
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

    print("Features used: {}".format(model.feature_cols))

    if not args.INCLUDE_DURATION:
        print('Feature dropped: {}'.format("Duration"))
    if not args.INCLUDE_STARTHOUR:
        print('Feature dropped: {}'.format("Start Hour"))
    if not args.INCLUDE_DAYOFWEEK:
        print('Feature dropped: {}'.format("Day of Week"))
    if not args.INCLUDE_PLACETYPE:
        print('Feature dropped: {}'.format("Place Type"))
    if not args.INCLUDE_CARGOTYPE:
        print('Feature dropped: {}'.format("Cargo Type"))
    if not args.INCLUDE_COMPANYINFO:
        print('Feature dropped: {}'.format("Company Info"))
    if not args.INCLUDE_VEHICLETYPE:
        print('Feature dropped: {}'.format("Vehicle Type"))
    if not args.INCLUDE_POI:
        print('Feature dropped: {}'.format("POI"))
    if not args.INCLUDE_URALANDUSE:
        print('Feature dropped: {}'.format("URA Land Use"))
    if not args.INCLUDE_OTHERACTIVITY:
        print('Feature dropped: {}'.format("Other Driver Activity"))
    if not args.INCLUDE_PASTACTIVITY:
        print('Feature dropped: {}'.format("Past Activity"))
    if not args.INCLUDE_LASTACTIVITY:
        print('Feature dropped: {}'.format("Last Activity"))
