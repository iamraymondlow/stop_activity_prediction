import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
from load_data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, f1_score, hamming_loss, \
    jaccard_score, precision_recall_fscore_support, roc_auc_score, zero_one_loss
from tqdm import tqdm

# load config file
with open(os.path.join(os.path.dirname(__file__), '../config.json')) as f:
    config = json.load(f)

# load parameters
parser = argparse.ArgumentParser()
parser.add_argument("--train_model", type=bool, default=True)
parser.add_argument("--eval_model", type=bool, default=True)
args = parser.parse_args()


class DeepNeuralNetwork(nn.Module):
    """
    Trains and evaluates the performance of a DNN models based on the training and test dataset.
    """
    def __init__(self):
        """
        Initialises the DNN model architecture.
        """
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

    def calculate_loss(self, output, target):
        """
        Calculates the loss value for each activity class and finds the average.

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


def train(model, optimiser, input_features, target, device):
    """
    Train the model in batches for one epoch.
    """
    model.train()
    train_loss = 0.0
    for i in tqdm(range(len(input_features) // config['batch_size'])):
        batch_features = input_features.iloc[i*config['batch_size']: (i+1)*config['batch_size']].to(device)
        batch_target = target.iloc[i * config['batch_size']: (i + 1) * config['batch_size']]
        delivercargo_target = batch_target['MappedActivity.DeliverCargo'].to(device)
        pickupcargo_target = batch_target['MappedActivity.PickupCargo'].to(device)
        other_target = batch_target['MappedActivity.Other'].to(device)
        shift_target = batch_target['MappedActivity.Shift'].to(device)
        break_target = batch_target['MappedActivity.Break'].to(device)
        dropofftrailer_target = batch_target['MappedActivity.DropoffTrailerContainer'].to(device)
        pickuptrailer_target = batch_target['MappedActivity.PickupTrailerContainer'].to(device)
        maintenance_target = batch_target['MappedActivity.Maintenance'].to(device)

        # reset optimiser gradient to zero
        optimiser.zero_grad()

        # perform inference
        output = model(batch_features)
        target = (delivercargo_target, pickupcargo_target, other_target, shift_target,
                  break_target, dropofftrailer_target, pickuptrailer_target, maintenance_target)
        loss = model.calculate_loss(output, target)
        train_loss += loss.item()

        # perform backpropagation
        loss.backward()

        # update model parameters
        optimiser.step()

    # find the average loss of all batches in this epoch
    train_loss = train_loss / (i + 1)
    return train_loss


def plot_train_loss(train_loss):
    """
    Plot training loss and save figure locally.
    """
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-entropy Loss')
    plt.title('Training Loss for Deep Neural Network')

    if not os.path.exists(os.path.join(os.path.dirname(__file__), config['figures_directory'])):
        os.makedirs(os.path.join(os.path.dirname(__file__), config['figures_directory']))

    plt.savefig(os.path.join(os.path.dirname(__file__),
                             config['figures_directory'] + 'DNN_train_loss.png'))
    plt.show()


def inference(test_features):
    """
    Performs inference on the test features.
    """
    test_features = test_features.to(device)
    outputs = model(test_features)

    # get all the labels
    all_labels = []
    for out in outputs:
        if out >= 0.5:
            all_labels.append(1)
        else:
            all_labels.append(0)

    return all_labels


def evaluate(test_y, test_pred):
    """
    Evaluates the performance of the trained model based on test dataset.
    """
    # generate evaluation scores
    print('Deep Neural Network')
    print('classes: {}'.format(activity_cols))
    print('accuracy: {}'.format(accuracy_score(test_y, test_pred)))
    print('f1 score: {}'.format(f1_score(test_y, test_pred, average=None)))
    print('hamming loss: {}'.format(hamming_loss(test_y, test_pred)))
    print('jaccard score: {}'.format(jaccard_score(test_y, test_pred, average=None)))
    print('roc auc score: {}'.format(roc_auc_score(test_y, test_pred)))
    print('zero one loss: {}'.format(zero_one_loss(test_y, test_pred)))
    print('precision recall fscore report: {}'.format(precision_recall_fscore_support(test_y, test_pred,
                                                                                      average=None)))
    print('classification report: {}'.format(classification_report(test_y, test_pred)))
    return None


if __name__ == '__main__':
    # load training and test datasets
    loader = DataLoader()
    train_data, test_data = loader.train_test_split(test_ratio=0.25)

    # define features of interest
    features = ['DriverID', 'Duration', 'StartHour', 'DayOfWeek.', 'PlaceType.', 'Commodity.',
                'SpecialCargo.', 'Company.Type.', 'Industry.', 'VehicleType.', 'NumPOIs', 'POI.',
                'LandUse.', 'Other.MappedActivity.', 'Past.MappedActivity.']
    feature_cols = [col
                    for col in train_data.columns
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

    train_x = train_data[feature_cols]
    train_y = train_data[activity_cols]
    test_x = test_data[feature_cols]
    test_y = test_data[activity_cols]

    if args.train_model:  # perform model training
        # initialise model architecture
        model = DeepNeuralNetwork()

        # initialise optimiser and learning parameters
        optimiser = optim.Adam(params=model.parameters(), lr=config['learning_rate'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)  # load the model into the device (i.e., CPU or CUDA)

        # train model
        epoch_train_loss = []
        for epoch in range(config['epochs']):
            print('Epoch {}/{}'.format(epoch, config['epochs']))
            epoch_loss = train(model, optimiser, train_x, train_y, device)
            epoch_train_loss.append(epoch_loss)
            print('Epoch loss: {}'.format(epoch_loss))

        # save trained model
        torch.save(model.state_dict(),
                   os.path.join(os.path.dirname(__file__),
                                config['activity_models_directory'] + 'model_DNN.pth'))

        # plot train loss graph
        plot_train_loss(epoch_train_loss)

    if args.eval_model:  # perform inference on test dataset and evaluate model performance
        model = DeepNeuralNetwork()
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),
                                                      config['activity_models_directory'] + 'model_DNN.pth')))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        test_pred = inference(test_x)
        evaluate(test_y, test_pred)
