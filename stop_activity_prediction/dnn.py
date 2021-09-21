import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np
from load_data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, hamming_loss, \
    jaccard_score, roc_auc_score, zero_one_loss
from tqdm import tqdm

# load config file
with open(os.path.join(os.path.dirname(__file__), '../config.json')) as f:
    config = json.load(f)

# load parameters
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='DNN')
parser.add_argument("--train_model", type=bool, default=True)
parser.add_argument("--eval_model", type=bool, default=True)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--output_dim", type=int, default=1)
args = parser.parse_args()


class DeepNeuralNetwork(nn.Module):
    """
    Defines the architecture of a deep neural network.
    """
    def __init__(self, input_dim):
        """
        Initialises the DNN model architecture. 5 fully connected layers and 8/18 binary output nodes
        for each activity class.

        Parameters:
            input_dim: int
                Number of input features that will be passed into the model.
        """
        super(DeepNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)

        self.out1 = nn.Linear(8, args.output_dim)
        self.out2 = nn.Linear(8, args.output_dim)
        self.out3 = nn.Linear(8, args.output_dim)
        self.out4 = nn.Linear(8, args.output_dim)
        self.out5 = nn.Linear(8, args.output_dim)
        self.out6 = nn.Linear(8, args.output_dim)
        self.out7 = nn.Linear(8, args.output_dim)
        self.out8 = nn.Linear(8, args.output_dim)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        """
        Performs forward pass through the DNN layers.

        Parameters:
            x: torch.tensor
                Input features of the model.

        Returns:
            out1, out2, out3, out4, out5, out6, out7, out8: torch.tensor
                Model output for each activity class.
        """
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc4(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc5(x))
        x = self.dropout(x)

        # a binary classifier output node for each activity class
        out1 = torch.sigmoid(self.out1(x))
        out2 = torch.sigmoid(self.out2(x))
        out3 = torch.sigmoid(self.out3(x))
        out4 = torch.sigmoid(self.out4(x))
        out5 = torch.sigmoid(self.out5(x))
        out6 = torch.sigmoid(self.out6(x))
        out7 = torch.sigmoid(self.out7(x))
        out8 = torch.sigmoid(self.out8(x))

        return out1.float(), out2.float(), out3.float(), out4.float(), out5.float(), \
               out6.float(), out7.float(), out8.float()

    def calculate_loss(self, output, target):
        """
        Calculates the loss value for each activity class and sums them up.

        Parameters:
            output: torch.tensor
                Predicted outputs of the model.
            target: torch.tensor
                Target outputs.
        Returns:
            sum_loss: float
                Binary cross entropy loss of model output for all activity classes.
        """
        out1, out2, out3, out4, out5, out6, out7, out8 = output
        t1, t2, t3, t4, t5, t6, t7, t8 = target
        loss1 = nn.BCELoss()(out1, torch.reshape(t1, (-1, 1))).float()
        loss2 = nn.BCELoss()(out2, torch.reshape(t2, (-1, 1))).float()
        loss3 = nn.BCELoss()(out3, torch.reshape(t3, (-1, 1))).float()
        loss4 = nn.BCELoss()(out4, torch.reshape(t4, (-1, 1))).float()
        loss5 = nn.BCELoss()(out5, torch.reshape(t5, (-1, 1))).float()
        loss6 = nn.BCELoss()(out6, torch.reshape(t6, (-1, 1))).float()
        loss7 = nn.BCELoss()(out7, torch.reshape(t7, (-1, 1))).float()
        loss8 = nn.BCELoss()(out8, torch.reshape(t8, (-1, 1))).float()

        sum_loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
        return sum_loss


def train(model, optimiser, train_features, train_target, device):
    """
    Train the model in batches for one epoch.

    Parameters:
        model: DeepNeuralNetwork object
            Contains the model architecture of the DNN.
        optimiser: optimiser.Adam
            Contains the optimiser for the model.
        train_features: pd.Dataframe
            Contain the input features for the training dataset.
        train_target: pd.Dataframe
            Contain the true labels for the training dataset.
        device: torch.device
            Indicates whether the model will be trained using CPU or CUDA.

    Returns:
        train_loss: float
            Contains the average training loss for this epoch.
    """
    model.train()
    train_loss = 0.0
    for i in tqdm(range(len(train_features) // config['batch_size'])):
        batch_features = torch.tensor(train_features.iloc[i*config['batch_size']:
                                                          (i+1)*config['batch_size']].values).to(device)
        batch_target = train_target.iloc[i*config['batch_size']: (i+1)*config['batch_size']]
        delivercargo_target = torch.tensor(batch_target['MappedActivity.DeliverCargo'].values).to(device)
        pickupcargo_target = torch.tensor(batch_target['MappedActivity.PickupCargo'].values).to(device)
        other_target = torch.tensor(batch_target['MappedActivity.Other'].values).to(device)
        shift_target = torch.tensor(batch_target['MappedActivity.Shift'].values).to(device)
        break_target = torch.tensor(batch_target['MappedActivity.Break'].values).to(device)
        dropofftrailer_target = torch.tensor(batch_target['MappedActivity.DropoffTrailerContainer'].values).to(device)
        pickuptrailer_target = torch.tensor(batch_target['MappedActivity.PickupTrailerContainer'].values).to(device)
        maintenance_target = torch.tensor(batch_target['MappedActivity.Maintenance'].values).to(device)

        # reset optimiser gradient to zero
        optimiser.zero_grad()

        # perform inference
        output = model(batch_features.float())
        target = (delivercargo_target.float(), pickupcargo_target.float(), other_target.float(),
                  shift_target.float(), break_target.float(), dropofftrailer_target.float(),
                  pickuptrailer_target.float(), maintenance_target.float())
        loss = model.calculate_loss(output, target)
        train_loss += loss.item()

        # perform backpropagation
        loss.backward()

        # update model parameters
        optimiser.step()

    # find the average loss of all batches in this epoch
    train_loss = train_loss / (len(train_features) // config['batch_size'])
    return train_loss


def plot_train_loss(train_loss):
    """
    Plot training loss and save the figure locally.

    Parameters:
        train_loss: list
            Contains the training loss for each epoch.
    """
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-entropy Loss')

    if not os.path.exists(os.path.join(os.path.dirname(__file__), config['figures_directory'])):
        os.makedirs(os.path.join(os.path.dirname(__file__), config['figures_directory']))

    plt.savefig(os.path.join(os.path.dirname(__file__),
                             config['figures_directory'] + 'train_loss_{}.png'.format(args.name)))
    plt.show()


def inference(model, input_features):
    """
    Performs inference on the input features.

    Parameters:
        model: DeepNeuralNetwork object
            Contains the model architecture of the DNN model.
        input_features:
            Contains the input features to be pased into the model for inference.

    Returns:
        all_labels: list
            Contains the activity labels inferred by the model based on the input features.
    """
    input_features = torch.tensor(input_features.values).float().to(device)
    outputs = model(input_features)

    # get all the labels
    all_labels = None
    for out in outputs:
        out = out.cpu().detach().numpy()
        if all_labels is None:
            all_labels = np.where(out < 0.5, 0, 1)
        else:
            all_labels = np.hstack((all_labels, np.where(out < 0.5, 0, 1)))

    return all_labels


def evaluate(true_labels, pred_labels):
    """
    Evaluates the performance of the trained model based on different multi-label classification metrics.

    Parameters:
        true_labels: pd.Dataframe
            Contains the target labels.
        pred_labels: torch.tensor
            Contains the model's predicted labels.
    """
    # generate evaluation scores
    print('Deep Neural Network')
    activity_labels = [col.replace('MappedActivity.', '') for col in true_labels.columns]
    pred_labels = pd.DataFrame(pred_labels, columns=true_labels.columns)
    print('Classes: {}'.format(activity_labels))
    print('Accuracy: {}'.format(accuracy_score(true_labels, pred_labels)))
    print('Hamming Loss: {}'.format(hamming_loss(true_labels, pred_labels)))
    print('Jaccard Score')
    print(jaccard_score(true_labels, pred_labels, average=None))
    print('ROC AUC Score')
    print(roc_auc_score(true_labels, pred_labels))
    print('Zero One Loss: {}'.format(zero_one_loss(true_labels, pred_labels)))
    print('Classification Report:')
    print(classification_report(true_labels, pred_labels, target_names=activity_labels, zero_division=0))
    return None


if __name__ == '__main__':
    # load training and test datasets
    loader = DataLoader()
    train_data, test_data = loader.train_test_split(test_ratio=0.25)

    # define features of interest
    features = ['DriverID', 'Duration', 'StartHour', 'DayOfWeek.', 'PlaceType.', 'Commodity.',
                'SpecialCargo.', 'Company.Type.', 'Industry.', 'VehicleType.', 'NumPOIs', 'POI.',
                'LandUse.', 'Other.MappedActivity.', 'Past.MappedActivity.']
    feature_cols = [col for col in train_data.columns
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
        model = DeepNeuralNetwork(input_dim=len(feature_cols))

        # initialise optimiser and learning parameters
        optimiser = optim.Adam(params=model.parameters(), lr=config['dnn_learning_rate'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

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
                                config['activity_models_directory'] + 'model_{}.pth'.format(args.name)))

        # plot train loss graph
        plot_train_loss(epoch_train_loss)

    if args.eval_model:  # perform inference on test dataset and evaluate model performance
        model = DeepNeuralNetwork(input_dim=len(feature_cols))
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),
                                                      config['activity_models_directory'] +
                                                      'model_{}.pth'.format(args.name))))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        train_pred = inference(model, train_x)
        print('Training Result')
        evaluate(train_y, train_pred)

        test_pred = inference(model, test_x)
        print('Test Result')
        evaluate(test_y, test_pred)
