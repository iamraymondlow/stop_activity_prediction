import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np
from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator
from load_data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, hamming_loss, \
    jaccard_score, roc_auc_score, zero_one_loss
from tqdm import tqdm
from random import seed, random


# load config file
with open(os.path.join(os.path.dirname(__file__), '../config.json')) as f:
    config = json.load(f)

# load parameters
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='BLSTM')
parser.add_argument("--train_model", type=bool, default=True)
parser.add_argument("--eval_model", type=bool, default=True)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--output_dim", type=int, default=1)
parser.add_argument("--class_weighting", type=bool, default=True)
parser.add_argument("--adaptive_sampling", type=bool, default=True)
parser.add_argument("--num_classes", type=int, default=8)
parser.add_argument("--sample_num", type=int, default=5)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()
seed(args.seed)


@variational_estimator
class BayesianLongShortTermMemory(nn.Module):
    """
    Defines the architecture of a bayesian long short-term memory neural network.
    """
    def __init__(self, input_dim):
        """
        Initialises the bayesian LSTM model architecture. 5 stacked BLSTM layers and 8/18 binary output nodes
        for each activity class.

        Parameters:
            input_dim: int
                Number of input features that will be passed into the model.
        """
        super(BayesianLongShortTermMemory, self).__init__()
        # define number of layers and nodes in each layer
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers

        # Bayesian LSTM layer
        self.blstm1 = BayesianLSTM(self.input_dim, self.hidden_dim)
        self.blstm2 = BayesianLSTM(self.hidden_dim, self.hidden_dim)
        self.blstm3 = BayesianLSTM(self.hidden_dim, self.hidden_dim)
        self.blstm4 = BayesianLSTM(self.hidden_dim, self.hidden_dim)
        self.blstm5 = BayesianLSTM(self.hidden_dim, self.hidden_dim)

        # fully connected output layer for each activity class
        self.out1 = nn.Linear(self.hidden_dim, args.output_dim)
        self.out2 = nn.Linear(self.hidden_dim, args.output_dim)
        self.out3 = nn.Linear(self.hidden_dim, args.output_dim)
        self.out4 = nn.Linear(self.hidden_dim, args.output_dim)
        self.out5 = nn.Linear(self.hidden_dim, args.output_dim)
        self.out6 = nn.Linear(self.hidden_dim, args.output_dim)
        self.out7 = nn.Linear(self.hidden_dim, args.output_dim)
        self.out8 = nn.Linear(self.hidden_dim, args.output_dim)

    def forward(self, x):
        """
        Performs forward pass through the LSTM layers.

        Parameters:
            x: torch.tensor
                Input features of the model.

        Returns:
            out1, out2, out3, out4, out5, out6, out7, out8: torch.tensor
                Model output for each activity class.
        """
        # initialise hidden state for first input with zeros
        hidden_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        print('hidden_0 size: {}'.format(hidden_0.size()))
        print('hidden_0 detach size: {}'.format(hidden_0.detach().size()))

        # initialise cell state for first input with zeros
        cell_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        print('cell_0 size: {}'.format(cell_0.size()))
        print('cell_0 detach size: {}'.format(cell_0.detach().size()))

        # forward propagation by passing input, hidden state and cell state into model
        x, (hidden_1, cell_1) = self.blstm1(x, (hidden_0.detach(), cell_0.detach()))
        print('hidden_1 size: {}'.format(hidden_1.size()))
        print('hidden_1 detach size: {}'.format(hidden_1.detach().size()))
        print('cell_1 size: {}'.format(cell_1.size()))
        print('cell_1 detach size: {}'.format(cell_1.detach().size()))
        x, (hidden_2, cell_2) = self.blstm2(x, (hidden_1, cell_1))
        x, (hidden_3, cell_3) = self.blstm3(x, (hidden_2, cell_2))
        x, (hidden_4, cell_4) = self.blstm4(x, (hidden_3, cell_3))
        x, _ = self.blstm5(x, (hidden_4, cell_4))

        # reshape output which has shape (batch_size, seq_length, hidden size) to fit into FC layer
        x = x[:, -1, :]

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

        if args.class_weighting:
            sum_loss = loss1 * delivercargo_weight + loss2 * pickupcargo_weight + \
                       loss3 * other_weight + loss4 * shift_weight + \
                       loss5 * break_weight + loss6 * dropofftrailer_weight + \
                       loss7 * pickuptrailer_weight + loss8 * maintenance_weight
        else:
            sum_loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8

        return sum_loss


def train(model, optimiser, train_features, train_target, device):
    """
    Train the model in batches for one epoch.

    Parameters:
        model: LongShortTermMemory object
            Contains the model architecture of the LSTM.
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
        batch_features = torch.tensor(
            train_features.iloc[i*config['batch_size']: (i+1)*config['batch_size']].values
        ).view([config['batch_size'], -1, model.input_dim]).to(device)
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

        # perform inference and compute loss
        target = (delivercargo_target.float(), pickupcargo_target.float(), other_target.float(),
                  shift_target.float(), break_target.float(), dropofftrailer_target.float(),
                  pickuptrailer_target.float(), maintenance_target.float())

        # calculate loss based on average of each sample
        for _ in range(args.sample_num):
            output = model(batch_features.float())
            loss = model.calculate_loss(output, target)
            train_loss += loss.item() / args.sample_num

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
        model: BayesianLongShortTermMemory object
            Contains the model architecture of the Bayesian LSTM model.
        input_features:
            Contains the input features to be passed into the model for inference.

    Returns:
        final_labels: np.array
            Contains the activity labels inferred by the model based on the input features.
    """
    final_outputs = None
    for i in tqdm(range(len(input_features) // config['batch_size'])):
        batch_features = torch.tensor(
            input_features.iloc[i*config['batch_size']: (i+1)*config['batch_size']].values
        ).view([config['batch_size'], -1, model.input_dim]).float().to(device)

        batch_outputs = np.zeros((batch_features.size()[0], args.num_classes))
        for _ in range(args.sample_num):
            outputs = model(batch_features)
            for j in range(len(outputs)):
                batch_outputs[:, j] += outputs[j].cpu().detach().numpy().reshape(-1) / args.sample_num

        if final_outputs is None:
            final_outputs = batch_outputs
        else:
            final_outputs = np.vstack((final_outputs, batch_outputs))

    final_labels = np.where(final_outputs < 0.5, 0, 1)
    return final_labels


def print_evaluation_results(true_labels, pred_labels):
    """
    Evaluates the performance of the trained model based on different multi-label classification metrics.

    Parameters:
        true_labels: pd.Dataframe
            Contains the target labels.
        pred_labels: torch.tensor
            Contains the model's predicted labels.
    """
    # generate evaluation scores
    print('Bayesian Long Short-Term Memory')
    activity_labels = [col.replace('MappedActivity.', '') for col in true_labels.columns]
    pred_labels = pd.DataFrame(pred_labels, columns=true_labels.columns)
    print('Classes: {}'.format(activity_labels))
    print('Accuracy: {}'.format(accuracy_score(true_labels, pred_labels)))
    print('Hamming Loss: {}'.format(hamming_loss(true_labels, pred_labels)))
    print('Jaccard Score')
    print(jaccard_score(true_labels, pred_labels, average=None))
    print(jaccard_score(true_labels, pred_labels, average=True))
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

    # introduce class weights based on inverse of class frequency
    delivercargo_weight = len(train_x) / (len(activity_cols) * train_y['MappedActivity.DeliverCargo'].sum())
    pickupcargo_weight = len(train_x) / (len(activity_cols) * train_y['MappedActivity.PickupCargo'].sum())
    other_weight = len(train_x) / (len(activity_cols) * train_y['MappedActivity.Other'].sum())
    shift_weight = len(train_x) / (len(activity_cols) * train_y['MappedActivity.Shift'].sum())
    break_weight = len(train_x) / (len(activity_cols) * train_y['MappedActivity.Break'].sum())
    dropofftrailer_weight = len(train_x) / (
            len(activity_cols) * train_y['MappedActivity.DropoffTrailerContainer'].sum())
    pickuptrailer_weight = len(train_x) / (len(activity_cols) * train_y['MappedActivity.PickupTrailerContainer'].sum())
    maintenance_weight = len(train_x) / (len(activity_cols) * train_y['MappedActivity.Maintenance'].sum())

    if args.train_model:  # perform model training
        # initialise model architecture
        model = BayesianLongShortTermMemory(input_dim=len(feature_cols))

        # initialise optimiser and learning parameters
        optimiser = optim.Adam(params=model.parameters(), lr=config['lstm_learning_rate'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # train model
        epoch_train_loss = []
        for epoch in range(config['epochs']):
            print('Epoch {}/{}'.format(epoch+1, config['epochs']))
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
        model = BayesianLongShortTermMemory(input_dim=len(feature_cols))
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),
                                                      config['activity_models_directory'] +
                                                      'model_{}.pth'.format(args.name))))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        train_pred = inference(model, train_x)
        print('Training Result')
        print_evaluation_results(train_y.iloc[:train_pred.shape[0]], train_pred)

        test_pred = inference(model, test_x)
        print('Test Result')
        print_evaluation_results(test_y.iloc[:test_pred.shape[0]], test_pred)
