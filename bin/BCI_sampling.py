import torch
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from optbnn.bnn.likelihoods import LikCategorical
from optbnn.bnn.priors import OptimGaussianPrior
from optbnn.utils import util
from optbnn.sgmcmc_bayes_net.classification_net import ClassificationNet

bci_dataset