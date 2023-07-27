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
        self.weights.weight.data.normal_(0, 1)
        self.weights.bias.data.normal_(0, 1)

    def forward(self, x):
        x = torch.from_numpy(x)  # typecast x to float32
        x = x.to(torch.float32)
        out = F.sigmoid(self.weights(x))
        return out


def fit(model, features, labels, loss_func):
    running_loss = 0.0
    optimizer = optim.Adam(model.parameters(), lr=1.e-3)
    for feature, label in zip(features, labels):
        optimizer.zero_grad()
        out = model.forward(feature)
        loss = loss_func(model, out, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss


def validate(model, features, labels, loss_func):
    valid_loss = 0.0
    for feature, label in zip(features, labels):
        out = model.forward(feature)
        loss = loss_func(model, out, label)
        valid_loss += loss.item()
    return valid_loss


def predict(model, features):
    predictions = []
    for feature in features:
        out = model.forward(feature)
        predictions.append(torch.round(out).item())
    return predictions

def predict_proba(model, features):
    probabilities = []
    for feature in features:
        out = model.forward(feature)
        probabilities.append(out)
    return probabilities



def XEL(model, out, target, l=1):
    result = (-1) * (target*torch.log(out) + (1-target)*torch.log(1-out)) + 0.5 * l * (torch.matmul(model.weights.weight, torch.t(model.weights.weight)) + model.weights.bias ** 2)
    return result
