import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import math
import torch.optim as optim


# TODO Fix math, code is working
# Hardcoded in methods that bias is from N(0,1), weights dist can be variated


class LaplaceApproximationClassifier(nn.Module):
    def __init__(self, n_features_selected=6):
        super(LaplaceApproximationClassifier, self).__init__()
        self.n_features_selected = n_features_selected
        self.weights = torch.nn.Linear(n_features_selected, 1)
        self.weights.weight.data.normal_(0, 1)
        self.weights.bias.data.normal_(0, 1)
        self.prior_mean = torch.zeros(n_features_selected)
        self.prior_sigma = 1 * torch.eye(n_features_selected)
        self.posterior_w_sigma = torch.eye(n_features_selected)
        self.posterior_b_sigma = torch.ones(1)

    def forward(self, x):
        x = torch.from_numpy(x)
        x = x.to(torch.float32)
        out = F.sigmoid(self.weights(x))
        return out


def CrossEntropyWithLogPrior(model, out, target):


    return (target * math.log(out + 1.e-6)) + (1 - target) * math.log(1 - out + 1.e-6) + 0.5 * (torch.matmul(torch.matmul((model.weights.weight - model.prior_mean),
                                                                                                             torch.linalg.inv(model.prior_sigma)),
                                                                                             torch.t(model.weights.weight - model.prior_mean)) + model.weights.bias ** 2)


def fit(model, x_train, y_train, loss_func):
    optimizer = optim.Adam(model.parameters(), lr=1.e-3)
    running_loss = 0.0
    b_hessian_delta = 0.0
    w_hessian_delta = 0.0
    for x, y in zip(x_train, y_train):
        optimizer.zero_grad()
        out = model.forward(x)
        loss = loss_func(model, out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        w_hessian_delta += out * (1-out) * torch.matmul(model.weights.weight, torch.t(model.weights.weight))
        b_hessian_delta += out * (1-out)

    # hessian computation
    model.posterior_w_sigma = torch.linalg.inv(torch.linalg.inv(model.prior_sigma) + w_hessian_delta)
    model.posterior_b_sigma = 1/(1 + b_hessian_delta)

    return running_loss


def validate(model, features, labels, loss_function):
    valid_loss = 0.0
    for feature, label in zip(features, labels):
        weights = dist.multivariate_normal.MultivariateNormal(loc=model.weights.weight, covariance_matrix=model.posterior_w_sigma).sample()
        # bias = dist.multivariate_normal.MultivariateNormal(loc=model.weights.bias, covariance_matrix=model.posterior_b_sigma).sample()
        bias = dist.normal.Normal(loc=model.weights.bias, scale=model.posterior_b_sigma).sample() # !!! this is only when bias has ONE component
        out = F.sigmoid(weights @ feature + bias)
        loss = loss_function(model, out, label)
        valid_loss += loss.item()

    return valid_loss


def predict(model, features):
    predictions = []
    for feature in features:
        weights = dist.multivariate_normal.MultivariateNormal(loc=model.weights.weight, covariance_matrix=model.posterior_w_sigma).sample()
        # bias = dist.multivariate_normal.MultivariateNormal(loc=model.weights.bias, covariance_matrix=model.posterior_b_sigma).sample()
        bias = dist.normal.Normal(loc=model.weights.bias, scale=model.posterior_b_sigma).sample() # !!! this is only when bias has ONE component
        out = F.sigmoid(weights @ feature + bias)
        predictions.append(torch.round(out).item())
    return predictions

def predict_proba(model, features):
    probabilities = []
    for feature in features:
        weights = dist.multivariate_normal.MultivariateNormal(loc=model.weights.weight, covariance_matrix=model.posterior_w_sigma).sample()
        # bias = dist.multivariate_normal.MultivariateNormal(loc=model.weights.bias, covariance_matrix=model.posterior_b_sigma).sample()
        bias = dist.normal.Normal(loc=model.weights.bias, scale=model.posterior_b_sigma).sample() # !!! this is only when bias has ONE component
        out = F.sigmoid(weights @ feature + bias)
        probabilities.append(out)
    return probabilities





