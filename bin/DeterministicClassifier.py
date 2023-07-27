import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeterministicClassifier(nn.Module):
    def __init__(self, n_features_selected=6):
        super(DeterministicClassifier, self).__init__()
        self.n_features_selected = n_features_selected
        self.weights = torch.nn.Linear(n_features_selected, 1)
        self.weights.weight.data = torch.tensor([[ 1.1164, -0.3639,  0.1513, -0.3514, -0.7906, -0.0915,  0.2352,  2.2440]],
                                                requires_grad=True)
        self.weights.bias.data = torch.tensor([0.5817], requires_grad=True)
        print(self.weights.weight)
        print(self.weights.bias)

    def forward(self, x):
        x = torch.from_numpy(x)  # typecast x to float32
        x = x.to(torch.float32)
        out = torch.sigmoid(self.weights(x))
        return out


def fit(model, features, labels, loss_func):
    #labels = torch.from_numpy(labels)
    #labels = labels.to(torch.float32)
    #labels = labels.unsqueeze(1)
    running_loss = 0.0
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=3.e-4)
    #criterion = torch.nn.BCELoss()
    for feature, label in zip(features, labels):
        optimizer.zero_grad()
        out = model.forward(feature)
        loss = loss_func(model, out, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    model.eval()
    # TODO optimize lr
    return running_loss


def validate(model, features, labels, loss_func):
    valid_preds = []
    valid_loss = 0.0
    for feature, label in zip(features, labels):
        out = model.forward(feature)
        loss = loss_func(model, out, label)
        valid_loss += loss.item()
        valid_preds.append(torch.round(out).item())
    return valid_loss, valid_preds


def predict(model, features, labels, loss_func):
    predictions = []
    total_loss = 0.0
    for feature, label in zip(features, labels):
        out = model.forward(feature)
        predictions.append(torch.round(out).item())
        total_loss += loss_func(model, out, label).item()
    return predictions, total_loss


def predict_proba(model, features):
    probabilities = []
    for feature in features:
        out = model.forward(feature)
        probabilities.append(out.item())
    return probabilities


def XEL(model, out, target, l=2):
    result = (-1) * (target*torch.log(out) + (1-target)*torch.log(1-out)) + 0.5 * l * (torch.matmul(model.weights.weight, torch.t(model.weights.weight)) + model.weights.bias ** 2)
    return result
