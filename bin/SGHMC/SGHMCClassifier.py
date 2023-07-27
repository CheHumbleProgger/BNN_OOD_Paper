import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from .AdaptiveSGHMC import AdaptiveSGHMC
import copy


class SGHMCClassifier(nn.Module):
    def __init__(self, sample_cache=2, n_features_selected=6, prior_rate = 0.5):
        super(SGHMCClassifier, self).__init__()
        self.n_features_selected = n_features_selected
        self.theta = nn.Linear(n_features_selected+1, 1, bias=False)
        self.prior_mean = torch.zeros(n_features_selected+1)
        self.prior_sigma = prior_rate * torch.eye(n_features_selected+1)
        self.samples_list = []
        self.prior_dist = dist.multivariate_normal.MultivariateNormal(loc=self.prior_mean, covariance_matrix=self.prior_sigma)
        prior_tensor = self.prior_dist.sample()
        self.samples_list.append(prior_tensor)
        self.sample_cache = [0] * sample_cache

    def forward(self, x):
        out = torch.sigmoid(self.theta(x))
        return out


def CrossEntropyWithLogPrior(model, out, target, eps=1.e-6):

    return (-1) * ((target * torch.log(out + eps)) + (1 - target) * torch.log(1 - out + eps)) + 0.5 * (torch.matmul(torch.matmul((model.theta.weight - model.prior_mean),
                                                                                                                     torch.linalg.inv(model.prior_sigma)),
                                                                                                        torch.t(model.theta.weight - model.prior_mean)))


def fit(model, x_train, y_train, loss_func, keep_every=200, burn_in_steps=3000):
    sampler = AdaptiveSGHMC(model.parameters(), num_burn_in_steps=burn_in_steps)
    running_loss = 0.0
    losses = []
    epoch = 0
    model.train()
    n_samples = 0
    steps = 0
    while True:
        for x, y in zip(x_train, y_train):
            x = np.append(x, 1)
            x = torch.from_numpy(x)
            x = x.to(torch.float32)
            sampler.zero_grad()
            out = model.forward(x)
            loss = loss_func(model, out, y)
            loss.backward()
            sampler.step()
            running_loss += loss.item()

            if (steps > burn_in_steps) and ((steps-burn_in_steps) % keep_every == 0):
                model.samples_list.append(copy.deepcopy(model.theta.weight))
                model.sample_cache[n_samples] = model.theta.weight
                n_samples += 1
            steps += 1

        losses.append(running_loss)
        running_loss = 0.0

        epoch += 1

        if n_samples >= len(model.sample_cache):
            break

    model.eval()
    return losses, epoch


def validate(model, features, labels, loss_function):
    valid_preds = []
    valid_loss = 0.0
    for feature, label in zip(features, labels):

        feature = np.append(feature, 1)
        feature = torch.from_numpy(feature)
        feature = feature.to(torch.float32)
        sample_preds = []

        for weight in model.sample_cache:
            with torch.no_grad():
                sample_preds.append(F.sigmoid(weight @ feature).item())

        sample_preds = torch.FloatTensor(sample_preds)
        out = torch.mean(sample_preds)
        loss = loss_function(model, out, label)
        valid_loss += loss.item()
        valid_preds.append(torch.round(out).item())

    return valid_loss, valid_preds


def predict(model, features, labels, loss_func, verbose=False):
    predictions = []
    total_loss = 0.0
    j = 0
    variances = []
    for feature, label in zip(features, labels):

        feature = np.append(feature, 1)
        feature = torch.from_numpy(feature)
        feature = feature.to(torch.float32)
        sample_preds = []

        for weight in model.samples_list:
            with torch.no_grad():
                logit = weight @ feature
                sample_pred = F.sigmoid(logit)
            sample_preds.append(sample_pred.item())

        variance = np.var(sample_preds)
        variances.append(variance)
        sample_preds = torch.FloatTensor(sample_preds)
        out = torch.mean(sample_preds)
        predictions.append(torch.round(out).item())
        total_loss += loss_func(model, out, label).item()
        j += 1
        if verbose:
            print('Datapoint: ', j, 'True label: ', label, 'Prediction: ', torch.round(out).item(), 'Variance: ', variance)
    return predictions, total_loss, variances
