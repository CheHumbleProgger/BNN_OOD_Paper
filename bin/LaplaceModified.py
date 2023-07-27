import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import math
import torch.optim as optim


class LaplaceApproximationClassifier(nn.Module):
    def __init__(self, n_features_selected=6):
        super(LaplaceApproximationClassifier, self).__init__()
        self.n_features_selected = n_features_selected
        self.theta = nn.Linear(n_features_selected+1,1,bias=False) # FIXME
        self.theta.weight.data = torch.tensor([[1.1164, -0.3639,  0.1513, -0.3514, -0.7906, -0.0915,  0.2352,  2.2440, 0.5817]], requires_grad=True)
        self.prior_mean = torch.zeros(n_features_selected+1)
        self.prior_sigma = (0.5) * torch.eye(n_features_selected+1) # TODO default set 0.5
        self.posterior_sigma = torch.eye(n_features_selected+1)


    def forward(self, x):
        #x = torch.from_numpy(x)
        #x = x.to(torch.float32)
        out = torch.sigmoid(self.theta(x))
        return out


def CrossEntropyWithLogPrior(model, out, target):

    return (-1) * ((target * torch.log(out)) + (1 - target) * torch.log(1 - out)) + 0.5 * (torch.matmul(torch.matmul((model.theta.weight - model.prior_mean),
                                                                                                                     torch.linalg.inv(model.prior_sigma)),
                                                                                                        torch.t(model.theta.weight - model.prior_mean)))


def fit(model, x_train, y_train, loss_func, flag):
    optimizer = optim.Adam(model.parameters(), lr=3.e-4)
    running_loss = 0.0
    hessian_delta = 0.0
    model.train()
    for x, y in zip(x_train, y_train):
        x = np.append(x, 1)
        x = torch.from_numpy(x)
        x = x.to(torch.float32)
        optimizer.zero_grad()
        out = model.forward(x)
        loss = loss_func(model, out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if flag == 1:
            hessian_delta += out * (1-out) * torch.matmul(x, torch.t(x))

    # hessian computation
    if flag == 1:
        model.posterior_sigma = torch.linalg.inv(torch.linalg.inv(model.prior_sigma) + hessian_delta)
    model.eval()
    return running_loss


def validate(model, features, labels, loss_function):
    valid_preds = []
    valid_loss = 0.0
    for feature, label in zip(features, labels):
        feature = np.append(feature, 1)
        distribution = dist.multivariate_normal.MultivariateNormal(loc=model.theta.weight, covariance_matrix=model.posterior_sigma)
        weights = torch.zeros(model.theta.weight.size())
        for i in range(100):
            weights += distribution.sample()
        average_weight = weights / 100
        out = F.sigmoid(average_weight @ feature)
        loss = loss_function(model, out, label)
        valid_loss += loss.item()
        valid_preds.append(torch.round(out).item())
    return valid_loss, valid_preds


def predict(model, features, labels, loss_func, n=100, verbose=False):
    predictions = []
    total_loss = 0.0
    j = 0
    variances = []
    for feature, label in zip(features, labels):
        feature = np.append(feature, 1)
        distribution = dist.multivariate_normal.MultivariateNormal(loc=model.theta.weight, covariance_matrix=model.posterior_sigma)
        weights = torch.zeros(model.theta.weight.size())
        sample_preds = []
        for i in range(n):
            sample_weight = distribution.sample()
            weights += sample_weight
            logit = sample_weight @ feature
            sample_pred = F.sigmoid(logit)
            sample_preds.append(sample_pred.item())
        average_weight = weights / n
        variance = np.var(sample_preds)
        variances.append(variance)
        out = F.sigmoid(average_weight @ feature)
        predictions.append(torch.round(out).item())
        total_loss += loss_func(model, out, label).item()
        j += 1
        if verbose:
            print('Datapoint: ', j, 'True label: ', label, 'Prediction: ', torch.round(out).item(), 'Variance: ', variance)
    return predictions, total_loss, variances


def predict_proba(model, features):
    probabilities = []
    for feature in features:
        feature = np.append(feature, 1)
        weights = torch.zeros(model.theta.weight.size())
        distribution = dist.multivariate_normal.MultivariateNormal(loc=model.theta.weight, covariance_matrix=model.posterior_sigma)
        for i in range(100):
            weights += distribution.sample()
        theta = weights / 100
        out = F.sigmoid(theta @ feature)
        probabilities.append(out.item())
    return probabilities
