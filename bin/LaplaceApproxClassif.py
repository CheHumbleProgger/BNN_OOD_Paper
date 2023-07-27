import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import math
import torch.optim as optim


class LaplaceApproximationClassifier(nn.Module):
    def __init__(self, n_features_selected=6):
        super(LaplaceApproximationClassifier, self).__init__()
        self.n_features_selected = n_features_selected
        self.weights = torch.nn.Linear(n_features_selected, 1)
        self.weights.weight.data = torch.tensor([[ 1.1164, -0.3639,  0.1513, -0.3514, -0.7906, -0.0915,  0.2352,  2.2440]],
                                          requires_grad=True)
        self.weights.bias.data = torch.tensor([0.5817], requires_grad=True)
        self.prior_mean = torch.zeros(n_features_selected)
        self.prior_sigma = (0.1) * torch.eye(n_features_selected)
        self.posterior_w_sigma = torch.eye(n_features_selected)
        self.posterior_b_sigma = torch.ones(1)
        print(self.weights.weight)
        print(self.weights.bias)

    def forward(self, x):
        #x = torch.from_numpy(x)
        #x = x.to(torch.float32)
        out = torch.sigmoid(self.weights(x))
        return out


def CrossEntropyWithLogPrior(model, out, target):

    return (-1) * ((target * torch.log(out)) + (1 - target) * torch.log(1 - out)) + 0.5 * (torch.matmul(torch.matmul((model.weights.weight - model.prior_mean),
                                                                                                                          torch.linalg.inv(model.prior_sigma)),
                                                                                                             torch.t(model.weights.weight - model.prior_mean)) + model.weights.bias ** 2)


def fit(model, x_train, y_train, loss_func, flag):
    optimizer = optim.Adam(model.parameters(), lr=3.e-4)
    running_loss = 0.0
    b_hessian_delta = 0.0
    w_hessian_delta = 0.0
    model.train()
    for x, y in zip(x_train, y_train):
        x = torch.from_numpy(x)
        x = x.to(torch.float32)
        optimizer.zero_grad()
        out = model.forward(x)
        loss = loss_func(model, out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if flag == 1:
            w_hessian_delta += out * (1-out) * torch.matmul(x, torch.t(x))
            b_hessian_delta += out * (1-out)

    # hessian computation
    if flag == 1:
        model.posterior_w_sigma = torch.linalg.inv(torch.linalg.inv(model.prior_sigma) + w_hessian_delta)
        model.posterior_b_sigma = 1/(1 + b_hessian_delta)
    model.eval()
    # TODO change hessian only during the last epoch
    return running_loss


def validate(model, features, labels, loss_function):
    valid_preds = []
    valid_loss = 0.0
    for feature, label in zip(features, labels):
        weights = dist.multivariate_normal.MultivariateNormal(loc=model.weights.weight, covariance_matrix=model.posterior_w_sigma).sample()
        # bias = dist.multivariate_normal.MultivariateNormal(loc=model.weights.bias, covariance_matrix=model.posterior_b_sigma).sample()
        bias = dist.normal.Normal(loc=model.weights.bias, scale=model.posterior_b_sigma).sample() # !!! this is only when bias has ONE component
        out = F.sigmoid(weights @ feature + bias)
        loss = loss_function(model, out, label)
        valid_loss += loss.item()
        valid_preds.append(torch.round(out).item())
    return valid_loss, valid_preds


def predict(model, features):
    predictions = []
    for feature in features:
        weights = dist.multivariate_normal.MultivariateNormal(loc=model.weights.weight, covariance_matrix=model.posterior_w_sigma).sample()
        # bias = dist.multivariate_normal.MultivariateNormal(loc=model.weights.bias, covariance_matrix=model.posterior_b_sigma).sample()
        bias = dist.normal.Normal(loc=model.weights.bias, scale=model.posterior_b_sigma).sample() # !!! this is only when bias has ONE component
        out = F.sigmoid(weights @ feature + bias)
        predictions.append(torch.round(out).item())
        # print(model.posterior_w_sigma, model.posterior_b_sigma)
    return predictions

def predict_proba(model, features):
    probabilities = []
    for feature in features:
        weights = dist.multivariate_normal.MultivariateNormal(loc=model.weights.weight, covariance_matrix=model.posterior_w_sigma).sample()
        # bias = dist.multivariate_normal.MultivariateNormal(loc=model.weights.bias, covariance_matrix=model.posterior_b_sigma).sample()
        bias = dist.normal.Normal(loc=model.weights.bias, scale=model.posterior_b_sigma).sample() # !!! this is only when bias has ONE component
        out = F.sigmoid(weights @ feature + bias)
        probabilities.append(out.item())
    return probabilities





