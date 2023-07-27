from optbnn.bnn.priors import PriorModule
from optbnn.bnn.likelihoods import LikelihoodModule
import math
import torch
import torch.distributions as dist
import numpy as np


def fill_cov_matrix(coordinates, scale=1.0):
    """Fills the covariantion matrix for Correlated Gaussian priors using the Matern Kernel""" # sigma == 1.0, rho == scale
    cov_matrix = torch.zeros(len(coordinates), len(coordinates))
    for x in range(len(coordinates)):
        for y in range(len(coordinates)):
            cov_matrix[x][y] = math.exp((-1) * np.linalg.norm(coordinates[x] - coordinates[y]) ** 2 / (2 * scale * scale))
    return cov_matrix


def params_counter(param):
    """Counts the total number of elements of param"""
    size = 1
    for i in param.size():
        size *= i
    return size


class StudentTPrior(PriorModule):
    """Class of t-Student Prior"""
    def __init__(self, mu=0.0, l=1.0, device="cpu"):
        super(StudentTPrior, self).__init__()
        self.mu = mu
        self.l = l
        self.df = 0

    def sample(self, name, param):
        """Sample parameters from prior"""
        mu, l, df = self._get_params_by_name(name)
        if (mu is None) and (l is None) and (df is None):
            return None
        if df == 0:
            self.df = params_counter(param) - 1
            df = self.df
        distribution = dist.studentT.StudentT(df=df, loc=mu, scale=l)
        return distribution.sample(param.size()).to(param.device)

    def logp(self, net):
        """Compute loglikelihood"""
        res = 0.
        for name, param in net.named_parameters():
            mu, l, df = self._get_params_by_name(name)
            if (mu is None) and (l is None) and (df is None):
                continue
            # df = params_counter(param) - 1
            res -= 0.5 * (df+1) * torch.sum(torch.log(1+(param-mu)**(2)/l))
        return res

    def _get_params_by_name(self, name):
        """Get params of prior (mu, l) by name"""
        if not (('.W' in name) or ('.b' in name)):
            return None, None, None
        else:
            return self.mu, self.l, self.df


class CombinedPrior(PriorModule):
    """Class which enables to use different priors for different convolutional and FC layers of network"""
    def __init__(self, default_prior, fc_prior, time_prior, spat_prior, device="cpu"):
        super(CombinedPrior, self).__init__()
        self.default_prior = default_prior
        self.fc_prior = fc_prior
        self.time_prior = time_prior
        self.spat_prior = spat_prior

    # TODO refactor CombinedPrior() class in: dict {name: prior} out -> same methods as before, but using dictionary
    def logp(self, net):
        """Compute loglikelihood using >1 priors"""
        res = 0.
        for name, param in net.named_parameters():
            # TODO if "classifier" or Linear then T-student, if "conv" and "spat" then Spat if "conv" and "time" then Time if "conv" and "time" and ".b" then fixed gauss
            if ("conv_classifier" in name) or (isinstance(param, torch.nn.Linear)):
                if isinstance(self.fc_prior, StudentTPrior):
                    mu, l, df = self.fc_prior._get_params_by_name(name)
                    if (mu is None) and (l is None) and (df is None):
                        continue
                    if df == 0:
                        df = params_counter(param) - 1
                        self.fc_prior.df = df
                    res -= 0.5 * (df+1) * torch.sum(torch.log(1+(param-mu)**(2)/l))
                else:
                    mu, std = self.default_prior._get_params_by_name(name)
                    if (mu is None) and (std is None):
                        continue
                    var = std ** 2
                    res -= torch.sum(((param - mu) ** 2) / (2 * var))


            elif ("conv_spat" in name): # spatial
                mu, cov_matrix = self.spat_prior._get_params_by_name(name)
                if (mu is None) and (cov_matrix is None):
                    continue
                for i in range(param.size()[0]):
                    for j in range(param.size()[1]):
                        diff = param[i][j] - mu
                        inv = torch.linalg.inv(cov_matrix)
                        l1 = torch.matmul(torch.matmul(diff, inv), torch.t(diff))
                        res -= 0.5 * torch.sum(l1)

            elif ("conv_time.w" in name) or ("conv_temporal.w" in name): # temporal
                mu, cov_matrix = self.time_prior._get_params_by_name(name)
                if (mu is None) and (cov_matrix is None):
                    continue
                for i in range(param.size()[0]):
                    for j in range(param.size()[1]):
                        diff = param[i][j] - mu
                        inv = torch.linalg.inv(cov_matrix)
                        l1 = torch.matmul(torch.matmul(diff, inv), torch.t(diff))
                        res -= 0.5 * torch.sum(l1)

            elif ("conv_time.b" in name): # default
                mu, std = self.default_prior._get_params_by_name(name)
                if (mu is None) and (std is None):
                    continue
                var = std ** 2
                res -= torch.sum(((param - mu) ** 2) / (2 * var))

            else: # default
                mu, std = self.default_prior._get_params_by_name(name)
                if (mu is None) and (std is None):
                    continue
                var = std ** 2
                res -= torch.sum(((param - mu) ** 2) / (2 * var))

        # TODO refactor it somehow

        return res

    def sample(self, name, param):
        """Sampling for parameters from different distributions"""
        # TODO if "classifier" or Linear then T-student, if "conv" and "spat" then Spat if "conv" and "time" then Time if "conv" and "time" and ".b" then fixed gauss
        if ("conv_classifier" in name) or (isinstance(param, torch.nn.Linear)): # T-student
            return self.fc_prior.sample(name, param)
        elif ("conv_spat" in name):
            return self.spat_prior.sample(name, param)# conv_spat
        elif ("conv_time.w" in name) or ("conv_temporal.w" in name): # conv_time
            return self.time_prior.sample(name, param)
        elif ("conv_time.b" in name):
            return self.default_prior.sample(name,param)
        else:
            return self.default_prior.sample(name, param)


class CorrelatedGaussianPrior(PriorModule):
    def __init__(self, mu, coordinates, scale=1.0, device="cpu"):
        super(CorrelatedGaussianPrior, self).__init__()
        self.mu = mu
        self.cov_matrix = fill_cov_matrix(coordinates, scale)

    def logp(self, net):
        """Compute loglikelihood"""
        res = 0.
        for name, param in net.named_parameters():
            mu, cov_matrix = self._get_params_by_name(name)
            if (mu is None) and (cov_matrix is None):
                continue
            # TODO calculate likelihood of layer using cycles if size 2 == 1 then A, elif size 3 == 1 then B, else raise Error
            for i in range(param.size()[0]):
                for j in range(param.size()[1]):
                    diff = param.data[i][j] - mu
                    inv = torch.linalg.inv(cov_matrix)
                    l1 = torch.matmul(torch.matmul(diff, inv), torch.t(diff))
                    res -= 0.5 * torch.sum(l1)
        return res

    def sample(self, name, param):
        """Sample parameters from prior"""
        mu, cov_matrix = self._get_params_by_name(name)
        if (mu is None) and (cov_matrix is None):
            return None
        # TODO fill weights according to its size using cycles if size 2 == 1 then A, elif size 3 == 1 then B, else raise Error
        distribution = dist.MultivariateNormal(loc=mu, covariance_matrix=cov_matrix)
        for i in range(param.size()[0]):
            for j in range(param.size()[1]):
                if param.size()[2] == 1:
                    param.data[i][j] = distribution.sample()
                elif param.size()[3] == 1:
                    param.data[i][j] = torch.unsqueeze(distribution.sample(), -1)
        return param.to(param.device) # FIXME????

    def _get_params_by_name(self, name):
        if not (('.W' in name) or ('.b' in name)):
            return None, None
        else:
            return self.mu, self.cov_matrix


class XELLikelihood(LikelihoodModule):
    def __init__(self):
        super(XELLikelihood, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')

    def loglik(self, fx, y):
        return -self.loss(fx, y)
