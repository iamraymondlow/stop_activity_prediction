import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np
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
parser.add_argument("--name", type=str, default='LSTM')
parser.add_argument("--train_model", type=bool, default=True)
parser.add_argument("--eval_model", type=bool, default=True)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=5)
parser.add_argument("--sequence_len", type=int, default=5)
parser.add_argument("--output_dim", type=int, default=1)
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
parser.add_argument("--bidirectional", type=bool, default=False)
parser.add_argument("--class_weighting", type=bool, default=False)
parser.add_argument("--label_weighting", type=bool, default=False)
parser.add_argument("--adaptive_sampling", type=bool, default=False)
parser.add_argument("--adaptive_resampling_prob", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()
seed(args.seed)


class LongShortTermMemory(nn.Module):
    """
    Defines the architecture of a long short-term memory neural network.
    """
    def __init__(self, input_dim):
        """
        Initialises the LSTM model architecture. 5 stacked LSTM layers and 8/18 binary output nodes
        for each activity class.

        Parameters:
            input_dim: int
                Number of input features that will be passed into the model.
        """
        super(LongShortTermMemory, self).__init__()
        # define number of layers and nodes in each layer
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers

        # LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,
                            batch_first=True, dropout=args.dropout, bidirectional=args.bidirectional)

        # fully connected output layer
        if args.bidirectional:
            self.out1 = nn.Linear(self.hidden_dim * 2, args.output_dim)
            self.out2 = nn.Linear(self.hidden_dim * 2, args.output_dim)
            self.out3 = nn.Linear(self.hidden_dim * 2, args.output_dim)
            self.out4 = nn.Linear(self.hidden_dim * 2, args.output_dim)
            self.out5 = nn.Linear(self.hidden_dim * 2, args.output_dim)
            self.out6 = nn.Linear(self.hidden_dim * 2, args.output_dim)
        else:
            self.out1 = nn.Linear(self.hidden_dim, args.output_dim)
            self.out2 = nn.Linear(self.hidden_dim, args.output_dim)
            self.out3 = nn.Linear(self.hidden_dim, args.output_dim)
            self.out4 = nn.Linear(self.hidden_dim, args.output_dim)
            self.out5 = nn.Linear(self.hidden_dim, args.output_dim)
            self.out6 = nn.Linear(self.hidden_dim, args.output_dim)

    def forward(self, x):
        """
        Performs forward pass through the LSTM layers.

        Parameters:
            x: torch.tensor
                Input features of the model.

        Returns:
            out1, out2, out3, out4, out5, out6: torch.tensor
                Model output for each activity class.
        """
        # initialise hidden state and cell state for first input with zeros
        if args.bidirectional:
            hidden_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_()
            cell_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_()
        else:
            hidden_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            cell_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # forward propagation by passing input, hidden state and cell state into model
        x, _ = self.lstm(x, (hidden_0.detach(), cell_0.detach()))

        # reshape output which has shape (batch_size, seq_length, hidden size) to fit into FC layer
        x = x[:, -1, :]

        # a binary classifier output node for each activity class
        out1 = torch.sigmoid(self.out1(x))
        out2 = torch.sigmoid(self.out2(x))
        out3 = torch.sigmoid(self.out3(x))
        out4 = torch.sigmoid(self.out4(x))
        out5 = torch.sigmoid(self.out5(x))
        out6 = torch.sigmoid(self.out6(x))

        return out1.float(), out2.float(), out3.float(), out4.float(), out5.float(), out6.float()

    def calculate_weight(self, pos_weight, neg_weight, mask_tensor):
        """
        Calculates the weights for BCELoss based on the positive and negative label weights.

        Parameters:
            pos_weight: float
                Weight applied to each positive class instance.
            neg_weight: float
                Weight applied to each negative class instance.
            mask_tensor: torch.tensor
                Contains an indicator tensor indicating if a particular activity is conducted or not in each batch.
        Returns:
            weight_tensor: torch.tensor
                Contains a tensor indicating the weight applied to each instance in each batch.
        """
        weight_tensor = pos_weight * mask_tensor + neg_weight * torch.abs((mask_tensor - 1.0))
        weight_tensor = torch.reshape(weight_tensor, (-1, 1))
        return weight_tensor

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
        out1, out2, out3, out4, out5, out6 = output
        t1, t2, t3, t4, t5, t6 = target

        if args.label_weighting:
            loss1 = nn.BCELoss(weight=self.calculate_weight(deliverpickupcargo_pos_weight, deliverpickupcargo_neg_weight, t1)) \
                (out1, torch.reshape(t1, (-1, 1))).float()
            loss2 = nn.BCELoss(weight=self.calculate_weight(other_pos_weight, other_neg_weight, t2)) \
                (out2, torch.reshape(t2, (-1, 1))).float()
            loss3 = nn.BCELoss(weight=self.calculate_weight(shift_pos_weight, shift_neg_weight, t3)) \
                (out3, torch.reshape(t3, (-1, 1))).float()
            loss4 = nn.BCELoss(weight=self.calculate_weight(break_pos_weight, break_neg_weight, t4)) \
                (out4, torch.reshape(t4, (-1, 1))).float()
            loss5 = nn.BCELoss(weight=self.calculate_weight(dropoffpickuptrailer_pos_weight, dropoffpickuptrailer_neg_weight, t5)) \
                (out5, torch.reshape(t5, (-1, 1))).float()
            loss6 = nn.BCELoss(weight=self.calculate_weight(maintenance_pos_weight, maintenance_neg_weight, t6)) \
                (out6, torch.reshape(t6, (-1, 1))).float()

        else:
            loss1 = nn.BCELoss()(out1, torch.reshape(t1, (-1, 1))).float()
            loss2 = nn.BCELoss()(out2, torch.reshape(t2, (-1, 1))).float()
            loss3 = nn.BCELoss()(out3, torch.reshape(t3, (-1, 1))).float()
            loss4 = nn.BCELoss()(out4, torch.reshape(t4, (-1, 1))).float()
            loss5 = nn.BCELoss()(out5, torch.reshape(t5, (-1, 1))).float()
            loss6 = nn.BCELoss()(out6, torch.reshape(t6, (-1, 1))).float()

        if args.class_weighting:
            sum_loss = loss1 * deliverpickupcargo_weight + \
                       loss2 * other_weight + \
                       loss3 * shift_weight + \
                       loss4 * break_weight + \
                       loss5 * dropoffpickuptrailer_weight + \
                       loss6 * maintenance_weight

        else:
            sum_loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

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
    batch_seq_sum = config['batch_size'] + args.sequence_len
    for i in tqdm(range(len(train_features) // batch_seq_sum)):
        start_idx = i * batch_seq_sum
        end_idx = (i+1) * batch_seq_sum
        batch_features = None
        for j in range(config['batch_size']):
            sub_batch_features = torch.tensor(
                train_features.iloc[(start_idx + j): (start_idx + args.sequence_len + j)].values
            ).view(-1, args.sequence_len, model.input_dim)

            if batch_features is None:
                batch_features = sub_batch_features
            else:
                batch_features = torch.cat((batch_features, sub_batch_features))

        batch_features = batch_features.to(device)
        batch_target = train_target.iloc[(start_idx + args.sequence_len): end_idx]
        deliverpickupcargo_target = torch.tensor(batch_target['MappedActivity.DeliverPickupCargo'].values).to(device)
        other_target = torch.tensor(batch_target['MappedActivity.Other'].values).to(device)
        shift_target = torch.tensor(batch_target['MappedActivity.Shift'].values).to(device)
        break_target = torch.tensor(batch_target['MappedActivity.Break'].values).to(device)
        dropoffpickuptrailer_target = torch.tensor(batch_target['MappedActivity.DropoffPickupTrailerContainer'].values).to(device)
        maintenance_target = torch.tensor(batch_target['MappedActivity.Maintenance'].values).to(device)

        # reset optimiser gradient to zero
        optimiser.zero_grad()

        # perform inference and compute loss
        output = model(batch_features.float())
        target = (deliverpickupcargo_target.float(),
                  other_target.float(),
                  shift_target.float(),
                  break_target.float(),
                  dropoffpickuptrailer_target.float(),
                  maintenance_target.float())
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

    if args.bidirectional:
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 config['figures_directory'] + 'train_loss_Bi{}.png'.format(args.name)))
    else:
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 config['figures_directory'] + 'train_loss_{}.png'.format(args.name)))
    plt.show()


def inference(model, input_features, raw_output=False):
    """
    Performs inference on the input features.

    Parameters:
        model: LongShortTermMemory object
            Contains the model architecture of the LSTM model.
        input_features:
            Contains the input features to be passed into the model for inference.
        raw_output: bool
            Indicates if the output should be the raw probability values or binary values.

    Returns:
        all_labels: list
            Contains the activity labels inferred by the model based on the input features.
    """
    all_labels = None
    raw_all_labels = []
    for i in range(len(input_features) // config['batch_size']):
        batch_features = torch.tensor(
            input_features.iloc[i*config['batch_size']: (i+1)*config['batch_size']].values
        ).view([config['batch_size'], -1, model.input_dim]).float().to(device)
        outputs = model(batch_features)

        if raw_output:
            if len(raw_all_labels) == 0:
                for j in range(len(outputs)):
                    raw_all_labels.append(outputs[j])
            else:
                for j in range(len(outputs)):
                    raw_all_labels[j] = torch.cat((raw_all_labels[j], outputs[j]), dim=0)

        else:
            # get batch labels
            batch_labels = None
            for out in outputs:
                out = out.cpu().detach().numpy()
                if batch_labels is None:
                    batch_labels = np.where(out < 0.5, 0, 1)
                else:
                    batch_labels = np.hstack((batch_labels, np.where(out < 0.5, 0, 1)))

            # concat batch results
            if all_labels is None:
                all_labels = batch_labels
            else:
                all_labels = np.vstack((all_labels, batch_labels))

    if raw_output:
        return tuple(raw_all_labels)
    else:
        return all_labels


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
    if args.bidirectional:
        print('Bidirectional Long Short-Term Memory')
    else:
        print('Long Short-Term Memory')
    activity_labels = [col.replace('MappedActivity.', '') for col in true_labels.columns]
    pred_labels = pd.DataFrame(pred_labels, columns=true_labels.columns)
    print('Classes: {}'.format(activity_labels))
    print('Accuracy: {}'.format(accuracy_score(true_labels, pred_labels)))
    print('Hamming Loss: {}'.format(hamming_loss(true_labels, pred_labels)))
    print('Jaccard Score')
    print(jaccard_score(true_labels, pred_labels, average=None))
    print(jaccard_score(true_labels, pred_labels, average='macro'))
    print('ROC AUC Score')
    print(roc_auc_score(true_labels, pred_labels))
    print('Zero One Loss: {}'.format(zero_one_loss(true_labels, pred_labels)))
    print('Classification Report:')
    print(classification_report(true_labels, pred_labels, target_names=activity_labels, zero_division=0))
    return None


def calculate_class_weight(total_stops, num_classes, num_activity):
    """
    Calculates the activity class weights, which is an inverse of the number of times the activity was conducted.

    Parameters:
        total_stops: int
            Contains the number of stops made in the training set.
        num_classes: int
            Contains the number of activity classes.
        num_activity: int
            Contains the number of times the activity was conducted.
    Returns:
        class_weight: float
    """
    class_weight = total_stops / (num_classes * num_activity)
    return class_weight


def calculate_label_weights(total_stops, num_activity):
    """
    Calculates the positive and negative weights, which is an inverse of the number of times the activity was conducted.

    Parameters:
        total_stops: int
            Contains the number of stops made in the training set.
        num_activity: int
            Contains the number of times the activity was conducted.
    Returns:
        pos_weight: float
        neg_weight: float
    """
    pos_weight = total_stops / num_activity
    neg_weight = total_stops / (total_stops - num_activity)

    return pos_weight, neg_weight


def calculate_trip_loss(model, train_data, feature_cols, epoch_num):
    """
    Calculates the normalised loss for each trip based on the existing model and rank them based on their loss.

    Parameters:
        model: pytorch model
            Contains the interim model.
        train_data: pandas.DataFrame
            Contains the complete training data.
        feature_cols: list
            Contains the list of input features.
        epoch_num: int:
            Contains the epoch number.

    Returns:
        log: pandas.DataFrame
            Contains the loss value for each trip
    """
    log = []
    for trip_id in train_data["TripID"].unique():
        trip_data = train_data[train_data["TripID"] == trip_id].reset_index(drop=True)
        trip_x = trip_data[feature_cols]

        if trip_x.shape[0] < config["batch_size"]:  # ignore trips that are shorter than the batch size
            continue

        trip_pred = inference(model, trip_x, raw_output=True)
        pred_size = trip_pred[0].size(dim=0)

        deliverpickupcargo_target = torch.tensor(trip_data['MappedActivity.DeliverPickupCargo'].values[:pred_size]).to(device)
        other_target = torch.tensor(trip_data['MappedActivity.Other'].values[:pred_size]).to(device)
        shift_target = torch.tensor(trip_data['MappedActivity.Shift'].values[:pred_size]).to(device)
        break_target = torch.tensor(trip_data['MappedActivity.Break'].values[:pred_size]).to(device)
        dropoffpickuptrailer_target = torch.tensor(trip_data['MappedActivity.DropoffPickupTrailerContainer'].values[:pred_size]).to(device)
        maintenance_target = torch.tensor(trip_data['MappedActivity.Maintenance'].values[:pred_size]).to(device)
        target = (deliverpickupcargo_target.float(),
                  other_target.float(),
                  shift_target.float(),
                  break_target.float(),
                  dropoffpickuptrailer_target.float(),
                  maintenance_target.float())

        trip_loss = model.calculate_loss(trip_pred, target).item() / len(trip_data)  # normalise trip loss based on stop number
        log.append({"trip_id": trip_id, "epoch_{}_trip_loss".format(epoch_num): trip_loss})

    log = pd.DataFrame(log)
    log.sort_values(by="epoch_{}_trip_loss".format(epoch_num), ascending=True, ignore_index=True, inplace=True)
    log["epoch_{}_rank".format(epoch_num)] = [i+1 for i in range(len(log))]
    return log


def assign_resample_prob(trip_rank, max_rank):
    """
    Assign the resampling probability for each trip based on its rank.

    Parameters:
        trip_rank: int
            Contains rank of the trip.
        max_rank: int
            Contains the rank of the worst trip.

    Returns:
        resample_prob: float
            Contains the resampling probability of a trip.
    """
    if trip_rank < int(0.1 * max_rank):
        resample_prob = 1
    else:
        resample_prob = args.adaptive_resampling_prob

    return resample_prob


def resample_trips(log, epoch_num):
    """
    Resamples trips for the next training iteration based on their resampling probabilities.

    Parameters:
        log: pandas.DataFrame
            Contains the resampling probabilities for each trip.
        epoch_num: int:
            Contains the epoch number.

    Returns:
        resampled_trips: list
            Contains the trips that has been resampled for the next training iteration.
        log: pandas.DataFrame
            Contains an additional column about whether a trip is being resampled.
    """
    log["epoch_{}_sampled".format(epoch_num)] = \
        log["epoch_{}_resample_prob".format(epoch_num)].apply(lambda x: 1 if x >= random() else 0)
    resampled_trips = log[log["epoch_{}_sampled".format(epoch_num)] == 1]["trip_id"].tolist()
    return resampled_trips, log


if __name__ == '__main__':
    # load training and test datasets
    loader = DataLoader()
    train_data, test_data = loader.train_test_split(test_ratio=0.25)

    # merge certain activity types  #TODO can be removed in the future if data processing code is rerun
    train_data["MappedActivity.DropoffPickupTrailerContainer"] = train_data[
                                                                     "MappedActivity.DropoffTrailerContainer"] + \
                                                                 train_data["MappedActivity.PickupTrailerContainer"]
    test_data["MappedActivity.DropoffPickupTrailerContainer"] = test_data[
                                                                    "MappedActivity.DropoffTrailerContainer"] + \
                                                                test_data["MappedActivity.PickupTrailerContainer"]
    train_data["MappedActivity.DeliverPickupCargo"] = train_data["MappedActivity.DeliverCargo"] + \
                                                      train_data["MappedActivity.PickupCargo"]
    test_data["MappedActivity.DeliverPickupCargo"] = test_data["MappedActivity.DeliverCargo"] + \
                                                     test_data["MappedActivity.PickupCargo"]

    train_data.loc[train_data["MappedActivity.DropoffPickupTrailerContainer"] > 0,
                   'MappedActivity.DropoffPickupTrailerContainer'] = 1
    test_data.loc[test_data["MappedActivity.DropoffPickupTrailerContainer"] > 0,
                  'MappedActivity.DropoffPickupTrailerContainer'] = 1
    train_data.loc[train_data["MappedActivity.DeliverPickupCargo"] > 0,
                   'MappedActivity.DeliverPickupCargo'] = 1
    test_data.loc[test_data["MappedActivity.DeliverPickupCargo"] > 0,
                  'MappedActivity.DeliverPickupCargo'] = 1

    # define features that will be passed into model
    features = []

    if args.INCLUDE_DURATION:
        features.extend(["Duration"])

    if args.INCLUDE_STARTHOUR:
        features.extend(["StartHour"])

    if args.INCLUDE_DAYOFWEEK:
        features.extend(["DayOfWeek."])

    if args.INCLUDE_PLACETYPE:
        features.extend(["PlaceType."])

    if args.INCLUDE_CARGOTYPE:
        features.extend(["Commodity.", "SpecialCargo."])

    if args.INCLUDE_COMPANYINFO:
        features.extend(["Company.Type.", "Industry."])

    if args.INCLUDE_VEHICLETYPE:
        features.extend(["VehicleType."])

    if args.INCLUDE_POI:
        features.extend(["NumPOIs", "POI."])

    if args.INCLUDE_URALANDUSE:
        features.extend(["LandUse."])

    if args.INCLUDE_OTHERACTIVITY:
        features.extend(["Other.MappedActivity."])

    if args.INCLUDE_PASTACTIVITY:
        features.extend(["Past.MappedActivity."])

    if args.INCLUDE_LASTACTIVITY:
        features.extend(["LastActivity."])

    feature_cols = [col for col in train_data.columns
                    for feature in features
                    if feature == col[:len(feature)]]

    # mapped activity types
    activity_cols = ['MappedActivity.DeliverPickupCargo', 'MappedActivity.Other', 'MappedActivity.Shift',
                     'MappedActivity.Break', 'MappedActivity.DropoffPickupTrailerContainer',
                     'MappedActivity.Maintenance']

    train_x = train_data[feature_cols]
    train_y = train_data[activity_cols]
    test_x = test_data[feature_cols]
    test_y = test_data[activity_cols]

    # introduce class weights based on inverse of class frequency
    deliverpickupcargo_weight = calculate_class_weight(len(train_x), len(activity_cols),
                                                       train_y['MappedActivity.DeliverPickupCargo'].sum())
    other_weight = calculate_class_weight(len(train_x), len(activity_cols),
                                          train_y['MappedActivity.Other'].sum())
    shift_weight = calculate_class_weight(len(train_x), len(activity_cols),
                                          train_y['MappedActivity.Shift'].sum())
    break_weight = calculate_class_weight(len(train_x), len(activity_cols),
                                          train_y['MappedActivity.Break'].sum())
    dropoffpickuptrailer_weight = calculate_class_weight(len(train_x), len(activity_cols),
                                                         train_y['MappedActivity.DropoffPickupTrailerContainer'].sum())
    maintenance_weight = calculate_class_weight(len(train_x), len(activity_cols),
                                                train_y['MappedActivity.Maintenance'].sum())

    # introduce label weights based on inverse of label frequency
    deliverpickupcargo_pos_weight, deliverpickupcargo_neg_weight = calculate_label_weights(
        len(train_x), train_y['MappedActivity.DeliverPickupCargo'].sum())
    other_pos_weight, other_neg_weight = calculate_label_weights(
        len(train_x), train_y['MappedActivity.Other'].sum())
    shift_pos_weight, shift_neg_weight = calculate_label_weights(
        len(train_x), train_y['MappedActivity.Shift'].sum())
    break_pos_weight, break_neg_weight = calculate_label_weights(
        len(train_x), train_y['MappedActivity.Break'].sum())
    dropoffpickuptrailer_pos_weight, dropoffpickuptrailer_neg_weight = calculate_label_weights(
        len(train_x), train_y['MappedActivity.DropoffPickupTrailerContainer'].sum())
    maintenance_pos_weight, maintenance_neg_weight = calculate_label_weights(
        len(train_x), train_y['MappedActivity.Maintenance'].sum())

    model = None
    if args.train_model:  # perform model training
        # initialise model architecture
        model = LongShortTermMemory(input_dim=len(feature_cols))

        # initialise optimiser and learning parameters
        optimiser = optim.Adam(params=model.parameters(), lr=config['lstm_learning_rate'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # train model
        epoch_train_loss = []
        adaptive_sampling_log = None
        for epoch in range(config['epochs']):
            print('Epoch {}/{}'.format(epoch+1, config['epochs']))
            epoch_loss = train(model, optimiser, train_x, train_y, device)
            epoch_train_loss.append(epoch_loss)
            print('Epoch loss: {}'.format(epoch_loss))

            if args.adaptive_sampling:
                # calculate loss score for each trip and perform ranking
                log = calculate_trip_loss(model, train_data, feature_cols, epoch+1)

                # assign resampling probability based on rank
                log["epoch_{}_resample_prob".format(epoch+1)] = log["epoch_{}_rank".format(epoch+1)]\
                    .apply(assign_resample_prob, max_rank=len(log))

                # perform resampling
                resampled_trips, log = resample_trips(log, epoch+1)
                sampled_train_data = train_data[train_data["TripID"].isin(resampled_trips)].reset_index(drop=True)
                train_x = sampled_train_data[feature_cols]
                train_y = sampled_train_data[activity_cols]

                # save resampling log
                if adaptive_sampling_log is None:
                    adaptive_sampling_log = log
                else:
                    adaptive_sampling_log = adaptive_sampling_log.merge(log, on="trip_id", how="left")

                # save adaptive sampling log
                if not os.path.exists(config["log_directory"]):
                    os.makedirs(config["log_directory"])

                if args.bidirectional:
                    adaptive_sampling_log.to_excel(config["log_directory"] + "log_{}.xlsx".format(args.name),
                                                   index=False)
                else:
                    adaptive_sampling_log.to_excel(config["log_directory"] + "log_Bi{}.xlsx".format(args.name),
                                                   index=False)

        # save trained model
        if args.bidirectional:
            torch.save(model.state_dict(),
                       os.path.join(os.path.dirname(__file__),
                                    config['activity_models_directory'] + 'model_Bi{}.pth'.format(args.name)))
        else:
            torch.save(model.state_dict(),
                       os.path.join(os.path.dirname(__file__),
                                    config['activity_models_directory'] + 'model_{}.pth'.format(args.name)))

        # plot train loss graph
        plot_train_loss(epoch_train_loss)

    if args.eval_model:  # perform inference on test dataset and evaluate model performance
        if model is None:
            model = LongShortTermMemory(input_dim=len(feature_cols))
            if args.bidirectional:
                model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),
                                                              config['activity_models_directory'] +
                                                              'model_Bi{}.pth'.format(args.name))))
            else:
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

    print('class weighting: {}'.format(args.class_weighting))
    print('label weighting: {}'.format(args.label_weighting))
    print('adaptive sampling: {}'.format(args.adaptive_sampling))
    print('resampling prob: {}'.format(args.resampling_prob))
    print("Features used: {}".format(feature_cols))

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
