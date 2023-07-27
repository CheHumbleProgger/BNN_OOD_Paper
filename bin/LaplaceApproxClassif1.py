import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import math
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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

    def fit(self, features, labels, optimizer=optim.Adam([self.weights.weight, self.weights.bias], lr=1.e-3)):  # ???
        running_loss = 0.0
        i = 0

        for feature, label in zip(features, labels):
            s = 0
            for x, y in zip(features, labels):
                print('x_model: ', x, 'y_model: ', y)
                s += 1
                if s > 101:
                    break

            optimizer.zero_grad()
            out = self.forward(feature)
            loss = self.CrossEntropyWithLogPrior(out, label)
            i += 1
            # if i % 10 == 1:
            # print('Sample: ', i, ' Loss: ', loss)
            # print('weights: ', self.weights.weight, '\n', 'bias: ', self.weights.bias)
            # print('out:', out, ' target: ', label)
            # print('weights shape: ', self.weights.weight.size())
            loss.backward()  # TODO change optimize from grad. descent to Adam
            optimizer.step()
            #w_grad = self.weights.weight.grad
            #b_grad = self.weights.bias.grad
            #with torch.no_grad():  # ???
            #   self.weights.weight -= lr * w_grad
            #   self.weights.bias -= lr * b_grad
            #self.weights.weight.grad.zero_()
            #self.weights.bias.grad.zero_()

            running_loss += loss.item()


        input_tuple = (self.weights.weight, self.weights.bias)
        hessian = torch.autograd.functional.hessian(self.NegLnPrior, input_tuple)
        weights_h = hessian[0][0]
        bias_h = hessian[1][1]
        #print('weights_h: ', weights_h)
        #print('alpha: ', weights_h[0][0][0][0])
        #print('bias_h: ', bias_h)
        self.posterior_b_sigma = torch.linalg.inv(bias_h)  # TODO rebuild using analytical formula for hessian
        self.posterior_w_sigma *= 1/weights_h[0][0][0][0]

        return running_loss

    def predict(self, features):
        predictions = []
        for feature in features:
            weights = dist.multivariate_normal.MultivariateNormal(loc=self.weights.weight, covariance_matrix=self.posterior_w_sigma).sample()
            bias = dist.multivariate_normal.MultivariateNormal(loc=self.weights.bias, covariance_matrix=self.posterior_b_sigma).sample()
            out = F.sigmoid(weights @ feature + bias)
            predictions.append(torch.round(out))
        # print('out: ', torch.round(out))  # TODO add uncertainty quantification
        return predictions

    def CrossEntropyWithLogPrior(self, out, target):
        return (-target * math.log(out + 1.e-6)) + (1 - target) * math.log(1 - out + 1.e-6) + 0.5 * (torch.matmul(torch.matmul((self.weights.weight - self.prior_mean),
                                                                                                                               torch.linalg.inv(self.prior_sigma)),
                                                                                                                  torch.t(self.weights.weight - self.prior_mean)) + self.weights.bias ** 2)
    # TODO add epsilon (1.e-6) in log in order to counter log definition space

    def NegLnPrior(self, weight, bias):
        return 0.5 * (torch.matmul(torch.matmul((weight - self.prior_mean), torch.linalg.inv(self.prior_sigma)), torch.t(weight - self.prior_mean)) + bias ** 2)
        # TODO REWRITE THIS FUNCTION ACC. TO BISHOP. How to add parts with input data?
