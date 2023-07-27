import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import math
import numpy as np


class BayesianLayer(nn.Module):
    def __init__(self, n_features_selected=6):
        super(BayesianLayer, self).__init__()
        self.n_features_selected = n_features_selected
        self.mus_w = torch.zeros(n_features_selected,requires_grad=True) # should be torch.tensors all
        self.sigmas_w = 0.01 * torch.eye(n_features_selected, requires_grad=True)
        self.mus_b = torch.zeros(1, requires_grad=True)
        self.sigmas_b = 0.01 * torch.ones(1, requires_grad=True)
        self.weights = torch.nn.Linear(n_features_selected, 1)
        #TODO make mu and sigma as tensors w/grad, make dists array based on these tensors and fill w and b tensors w/out grad by sampling from dists, optimize mu and sigma w.r.t loss
        # ARCHITECTURE: V6->v1(?) cuz' of binary classif
    def sampling(self, n_features_selected): #FIXIT, failure in distribution plot
        #for weight, mu, sigma in zip(self.weights.weight, self.mus_w , self.sigmas_w ): #FIXIT rewrite using index and range
            #distr = dist.normal.Normal(loc=mu, scale=sigma)
            #self.weights.weight = nn.Parameter(distr.sample())
        for i in range(n_features_selected):
            distr = dist.normal.Normal(loc=self.mus_w[i], scale=self.sigmas_w[i][i])
            with torch.no_grad():
                self.weights.weight[0][i] = nn.Parameter(distr.sample())

        for weight, mu, sigma in zip(self.weights.bias, self.mus_b, self.sigmas_b):
            distr = dist.normal.Normal(loc=mu, scale=sigma)
            self.weights.bias = nn.Parameter(distr.sample())



    def forward(self, x):
        #x = np.insert(x,0,1) # make x[0]=1 next components to be as old x so that W(t)X+b becomes W(t)x
        #w = self.weights_prior.sample() # we need to save current weights to compute loss
        x = torch.from_numpy(x) # typecast x to float32
        x = x.to(torch.float32)
        print('x', x, 'w', self.weights)
        print(self.weights.weight.shape)
        self.sampling(self.n_features_selected)
        print('x', x.dtype, 'w', self.weights.weight.dtype, 'b', self.weights.bias.dtype)
        out = F.sigmoid(self.weights(x))
        #prod = np.ndarray([np.dot(x,w)])
        #out = F.sigmoid(prod) # x * W from weights_prior REDO using torch dot product or typecasting it to tensor
        return out

    def fit(self, features, labels, lr=1.e-3):
        for feature, label in zip(features, labels):
            print(feature.shape)
            output = self.forward(feature)
            loss = neg_notnorm_log_posterior(output, label, self)
            #loss.retain_grad()
            loss.backward()
            # optimizer.step() # probably, direct interpretation for mu and sigma is needed
            # direct computation of gradients for mu and sigma
            grad_mu_b = self.mus_b.grad #TODO fix getting gradients & grad desc
            grad_sigma_b = self.sigmas_b.grad() # NONE??????
            grad_mu_w = self.mus_w.grad
            grad_sigma_w = self.sigmas_w.grad() # NONE???? WTF is with the gradients??
            print('mub ', self.mus_b, ' sigmab ', self.sigmas_b, ' muw ', self.mus_w, ' sigmaw ', self.sigmas_w)
            print('mub_grad ', grad_mu_b,' sigmab_grad ', grad_sigma_b, ' muw_grad ', grad_mu_w, ' sigmaw_grad ', grad_sigma_w) # smth, none, smth, none
            print('weights ', self.weights.weight, ' bias ', self.weights.bias)
            print("loss: ", loss)
            with torch.no_grad():
                self.mus_w -= lr * grad_mu_w
                self.mus_b -= lr * grad_mu_b
                self.sigmas_w -= lr * grad_sigma_w # No gradient. WHY????
                self.sigmas_b -= lr * grad_sigma_b # No gradient. WHY????
            self.sigmas_b.grad.zero_()
            self.sigmas_w.grad.zero_()
            self.mus_w.grad.zero_()
            self.mus_b.grad.zero_()


    def predict(self, features, labels):
        predictions = []
        for feature, label in zip(features, labels):
            out = self.forward(feature)
            predictions.append(torch.round(out))
        return predictions


def neg_notnorm_log_posterior(out, target, self):
    return (-1) * (target * math.log(out) + (1-target) * math.log(1-out) + ((self.weights.weight - self.mus_w) @
                                                                            torch.t((self.weights.weight - self.mus_w)) + (self.weights.bias - self.mus_b) ** 2)/
                   ((torch.linalg.det(self.sigmas_w) * self.sigmas_b)**0.5))



