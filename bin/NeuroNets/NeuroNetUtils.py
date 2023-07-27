import netbios
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import copy
from braindecode.models import EEGNetv4, ShallowFBCSPNet
from .EEGNetV4 import squeeze_final_output, _transpose_1_0


class NewEEGNet(EEGNetv4):
    def __init__(self, net_params):
        super(NewEEGNet, self).__init__(**net_params)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

    def predict(self, x):
        self.eval()
        predictions = self.forward(x)
        predictions = torch.exp(predictions) # FIXME uncomment
        return predictions

    def forward(self, x): # FIXME was True
        return super().forward(x)


class NewShallowFBCSPNet(ShallowFBCSPNet):
    def __init__(self, net_params):
        super(NewShallowFBCSPNet, self).__init__(**net_params)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

    def predict(self, x):
        self.eval()
        predictions = self.forward(x)
        predictions = torch.exp(predictions) # FIXME uncomment
        return predictions

    def forward(self, x): # FIXME was True
        return super().forward(x)


class SaveLogits:
    def __init__(self):
        self.logits = []

    def __call__(self, module, input, output):
        raise NotImplementedError

    def clear(self):
        self.logits = []


class SaveEEGNetLogits(SaveLogits):
    def __call__(self, module, input, output):
        #self.logits.append(eegnet_postprod(output.detach().cpu()))
        self.logits.append(eegnet_postprod(output.detach().cpu())) # FIXME was line above


class SaveShallowFBCSPNetLogits(SaveLogits):
    def __call__(self, module, input, output):
        self.logits.append(fbcspnet_postprod(output.detach().cpu()))


def eegnet_postprod(x):
    return squeeze_final_output(_transpose_1_0(x))


def fbcspnet_postprod(x):
    return squeeze_final_output(x)

def fit(net, prior, likelihood, optimizer, cache_length, X_train, y_train, keep_every=200, burn_in_steps=3000, device='cpu'):
    model_cache = [0] * cache_length
    running_loss = 0.0
    losses = []
    train_accuracy = []
    epochs = 0
    n_samples = 0
    steps = 0
    len_dataset = len(y_train)
    net.train()
    optimizer.num_burn_in_steps = burn_in_steps

    y_train = torch.LongTensor(y_train)
    if isinstance(likelihood, GaussianLikelihood):
        y_train = torch.nn.functional.one_hot(y_train, 2)
    while n_samples < cache_length:
        right_counter = 0
        for feature, label in zip(X_train, y_train):
            x = torch.from_numpy(feature)
            label = torch.tensor(label, dtype=torch.float32)
            label = label.unsqueeze(-1)
            label = torch.t(label)
            x = x.unsqueeze(-1)
            x = x.permute(2, 0, 1)
            #x = x.to(device)
            #label = label.to(device)
            optimizer.zero_grad()
            out = net.forward(x)
            if torch.argmax(out).item() == torch.argmax(label).item():
                right_counter += 1

            print(out)
            loss = likelihood(out, label) + prior(net) / len_dataset # FIXME ?? check all signs in the loss function was added -1
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (steps > burn_in_steps) and (steps % keep_every == 0):
                sample = copy.deepcopy(net.state_dict())
                model_cache[n_samples] = sample
                n_samples += 1

            steps += 1

        train_accuracy.append(right_counter / len_dataset)
        losses.append(running_loss)
        running_loss = 0.0
        epochs += 1

    net.eval()

    return model_cache, losses, epochs, train_accuracy


def deterministic_fit(net, prior, likelihood, optimizer, X_train, y_train, epochs=50, device='cpu'):
    losses = []
    net.train()
    training_accuracies = []
    dataset_length = len(y_train)
    if likelihood is isinstance(GaussianLikelihood):
        y_train = torch.LongTensor(y_train)
        y_train = torch.nn.functional.one_hot(y_train, 2)

    for i in range(epochs):
        running_loss = 0.0
        right_counter = 0
        if i % 1000 == 0:
            print('Epoch: ', i)
        for feature, label in zip(X_train, y_train):
            x = torch.from_numpy(feature)
            label = torch.tensor(label, dtype=torch.float32)
            label = label.unsqueeze(-1)
            label = torch.t(label)
            x = x.unsqueeze(-1)
            x = x.permute(2, 0, 1)
            x = x.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            out = net.forward(x)
            if torch.argmax(out).item() == torch.argmax(label).item():
                right_counter += 1
            loss = likelihood(out, label) + prior(net) / dataset_length
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        losses.append(running_loss)
        training_accuracies.append(right_counter / dataset_length)

    net.eval()

    return losses, training_accuracies


def validate(net, prior, likelihood, model_cache, X_valid, y_valid, device='cpu'):
    valid_loss = 0.0
    valid_preds = []
    for feature, label in zip(X_valid, y_valid):
        sample_preds = []
        x = torch.from_numpy(feature)
        x = x.unsqueeze(-1)
        x = x.permute(2, 0, 1)
        #x = x.to(device)
        label = torch.tensor(label, dtype=torch.float32)
        label = label.unsqueeze(-1)
        label = label.unsqueeze(-1)
        #label = label.to(device)
        #model_cache = model_cache.to(device)
        # TODO send model_cache to device????
        for sample in model_cache:
            net.load_state_dict(sample)
            with torch.no_grad():
                sample_preds.append(torch.argmax(net.forward(x)))

        sample_preds = torch.FloatTensor(sample_preds)
        avg_pred = torch.mean(sample_preds)
        avg_pred = avg_pred.unsqueeze(-1)
        avg_pred = avg_pred.unsqueeze(-1)
        # loss = likelihood(avg_pred, label) + prior(net)
        #valid_loss += loss.item()
        valid_preds.append(torch.round(avg_pred).item())

    return valid_loss, valid_preds


def deterministic_validate(net, prior, likelihood, X_valid, y_valid, device='cpu'):
    valid_loss = 0.0
    valid_preds = []

    for feature, label in zip(X_valid, y_valid):
        x = torch.from_numpy(feature)
        x = x.unsqueeze(-1)
        x = x.permute(2, 0, 1)
        x = x.to(device)
        label = torch.tensor(label, dtype=torch.float32)
        label = label.unsqueeze(-1)
        label = label.unsqueeze(-1)
        label = label.to(device)
        out = torch.argmax(net.forward(x))
        out = out.unsqueeze(-1)
        out = out.unsqueeze(-1)
        loss = likelihood(out, label) + prior(net)
        valid_loss += loss.item()
        valid_preds.append(out.item())

    return valid_loss, valid_preds


def predict(net, sampled_weights, X_test, y_test, device='cpu'):
    test_preds = []
    variances = []

    for feature, label in zip(X_test, y_test):
        sample_preds = []

        x = torch.from_numpy(feature)
        x = x.unsqueeze(-1)
        x = x.permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.float32)
        label = label.unsqueeze(-1)
        label = label.unsqueeze(-1)
        #x = x.to(device)
        #label = label.to(device)
        #sampled_weights = sampled_weights.to(device)
        # TODO send sampled_weights to device??
        for sample in sampled_weights:
            net.load_state_dict(sample)
            with torch.no_grad():
                sample_preds.append(torch.argmax(net.forward(x)))

        sample_preds = torch.FloatTensor(sample_preds)
        avg_pred = torch.mean(sample_preds)
        variance = torch.var(sample_preds)
        variances.append(variance.item())
        test_preds.append(torch.round(avg_pred).item())

    return test_preds, variances


def deterministic_predict(net, X_test, y_test, device='cpu'):
    test_preds = []

    for feature, label in zip(X_test, y_test):
        x = torch.from_numpy(feature)
        x = x.unsqueeze(-1)
        x = x.permute(2, 0, 1)
        x = x.to(device)
        label = torch.tensor(label, dtype=torch.float32)
        label = label.unsqueeze(-1)
        label = label.unsqueeze(-1)
        label = label.to(device)
        out = torch.argmax(net.forward(x))
        out = out.unsqueeze(-1)
        out = out.unsqueeze(-1)
        test_preds.append(out.item())

    return test_preds


class GaussianPrior(nn.Module):
    def __init__(self, mu=0.0, std=1.0):
        super(GaussianPrior, self).__init__()
        self.mu = mu
        self.std = std

    def forward(self, net):
        return -self.logp(net)

    def initialize(self, net):
        for name, param in net.named_parameters():
            if param.requires_grad:
                value = self.sample(name, param)
                if value is not None:
                    param.data.copy_(value)

    def logp(self, net):
        res = 0.
        for name, param in net.named_parameters():
            mu, std = self._get_params_by_name(name)
            if (mu is None) and (std is None):
                continue
            var = std ** 2
            res -= torch.sum(((param - mu) ** 2) / (2 * var))
        return res

    def sample(self, name, param):
        mu, std = self._get_params_by_name(name)
        if (mu is None) and (std is None):
            return None

        return mu + std * torch.rand_like(param)

    def _get_params_by_name(self, name):
        if not (('.W' in name) or ('.b' in name)):
            return None, None
        else:
            return self.mu, self.std


class GaussianLikelihood(nn.Module):
    def __init__(self, var):
        super(GaussianLikelihood, self).__init__()
        self.loss = torch.nn.MSELoss(reduction='sum')
        self.var = var

    def forward(self, out, y):
        return - self.loglik(out, y)

    def loglik(self, out, y):
        return - 0.5 / self.var * self.loss(out, y)


class LikCategorical(nn.Module):
    def __init__(self):
        super(LikCategorical, self).__init__()
        self.loss = torch.nn.NLLLoss(reduction='sum')

    def loglik(self, out, y):
        return -self.loss(out, y)

    def forward(self, out, y):
        return - self.loglik(out, y)
