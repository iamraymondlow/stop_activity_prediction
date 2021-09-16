import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from load_data import DataLoader
from joblib import dump, load
from sklearn.metrics import accuracy_score, classification_report, f1_score, hamming_loss, \
    jaccard_score, precision_recall_fscore_support, roc_auc_score, zero_one_loss
from tqdm import tqdm



# load config file
with open(os.path.join(os.path.dirname(__file__), '../config.json')) as f:
    config = json.load(f)

class DeepNeuralNetwork(nn.Module):
    """
    Trains and evaluates the performance of a DNN models based on the training and test dataset.
    """
    def __init__(self):
        """
        Initialises the model object by loading the training and test datasets as well as the DNN layers.
        """
        # load training and test datasets
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

        # initialise model layers (4 fully connected layers and 8/18 heads serving as binary classifiers)
        super(DeepNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(len(self.train_x.columns), 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)

        self.out1 = nn.Linear(256, 1)
        self.out2 = nn.Linear(256, 1)
        self.out3 = nn.Linear(256, 1)
        self.out4 = nn.Linear(256, 1)
        self.out5 = nn.Linear(256, 1)
        self.out6 = nn.Linear(256, 1)
        self.out7 = nn.Linear(256, 1)
        self.out8 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Performs forward pass through the layers.

        Parameters:
            x: tensor
                Input features of the model.

        Returns:
            out1, out2, out3, out4, out5, out6, out7, out8: tensor
                Outputs of the model for each activity class.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # each binary classifier head will have its own output
        out1 = F.sigmoid(self.out1(x))
        out2 = F.sigmoid(self.out2(x))
        out3 = F.sigmoid(self.out3(x))
        out4 = F.sigmoid(self.out4(x))
        out5 = F.sigmoid(self.out5(x))
        out6 = F.sigmoid(self.out6(x))
        out7 = F.sigmoid(self.out7(x))
        out8 = F.sigmoid(self.out8(x))

        return out1, out2, out3, out4, out5, out6, out7, out8

    def _calculate_loss(self, output, target):
        """
        Performs forward pass through the layers.

        Parameters:
            output: tensor
                Predicted outputs of the model.
            target: tensor
                Target outputs.
        Returns:
            float
                Binary cross entropy loss of model output.
        """
        out1, out2, out3, out4, out5, out6, out7, out8 = output
        t1, t2, t3, t4, t5, t6, t7, t8 = target
        loss1 = nn.BCELoss()(out1, t1)
        loss2 = nn.BCELoss()(out2, t2)
        loss3 = nn.BCELoss()(out3, t3)
        loss4 = nn.BCELoss()(out4, t4)
        loss5 = nn.BCELoss()(out5, t5)
        loss6 = nn.BCELoss()(out6, t6)
        loss7 = nn.BCELoss()(out7, t7)
        loss8 = nn.BCELoss()(out8, t8)
        return (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8) / 5

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
    model = DeepNeuralNetwork()

    # train and evaluate model performance
    model.train()
    model.evaluate()
