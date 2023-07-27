import numpy as np
import os
import torch
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, EEGNetv4
from braindecode import EEGClassifier
from skorch.callbacks import LRScheduler
import scipy.signal as signal
from scipy.signal import cheb2ord
from scipy.stats import entropy
import bin.LoadData as LoadData
import bin.DeterministicClassifier as DC
import bin.Preprocess as Preprocess
import bin.LaplaceModified as LCM
import matplotlib.pyplot as plt
import sklearn.feature_selection
from sklearn.feature_selection import mutual_info_classif
from .NeuroNets import NeuroNetUtils, EEGNetV4, ShallowFBCSPNet
from mne.decoding import CSP
from datetime import datetime
from .SGHMC import SGHMCClassifier as SGHMC
from sklearn.metrics import auc, roc_curve
import seaborn as sns
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from torch.optim import AdamW
from .SGHMC.AdaptiveSGHMC import AdaptiveSGHMC
from optbnn.bnn.likelihoods import LikCategorical
from optbnn.bnn.priors import FixedGaussianPrior
from bin.my_priors import StudentTPrior, CombinedPrior, CorrelatedGaussianPrior, XELLikelihood
from optbnn.utils import util
from optbnn.sgmcmc_bayes_net.classification_net import ClassificationNet
from optbnn.utils import logger
from .NeuroNets.EEGNetV4 import squeeze_final_output, _transpose_1_0

class MLEngine:
    def __init__(self, data_path='', file_to_load='', subject_id='', sessions=[1, 2], ntimes=1, kfold=2, m_filters=2,
                 window_details={}):
        self.data_path = data_path
        self.subject_id = subject_id
        self.file_to_load = file_to_load
        self.sessions = sessions
        self.kfold = kfold
        self.ntimes = ntimes
        self.window_details = window_details
        self.m_filters = m_filters # TODO update class parameters add network, prior var, smth else?

    def experiment(self, subj2_filename=None):

        n_best_features = 8
        subj1_filename = self.file_to_load
        X_cv, y_cv, X_test, y_test, ood_X_test, ood_y_test = self.preprocess(subj1_filename, subj2_filename, n_best_features)
        bayes_classifier = LCM.LaplaceApproximationClassifier(n_features_selected=n_best_features)
        det_classifier = DC.DeterministicClassifier(n_features_selected=n_best_features)
        train_curves = self.classifiers_train(X_cv, y_cv, bayes_classifier, det_classifier)

        bay_preds, bay_total_loss, variances = LCM.predict(bayes_classifier, X_test, y_test, LCM.CrossEntropyWithLogPrior, n=500, verbose=True)
        det_preds, det_total_loss = DC.predict(det_classifier, X_test, y_test, DC.XEL)

        if subj2_filename is None: # 1 subj experiment
            var_ind, var_ood = domain_split(variances, y_test)
            preds_ind, preds_ood = domain_split(bay_preds, y_test)
            sd = []
            for g in variances:
                sd.append(np.sqrt(g))
            sd_ind, sd_ood = domain_split(sd, y_test)

        else: # 2 subj experiment
            ood_bay_preds, ood_bay_total_loss, ood_variances = LCM.predict(bayes_classifier, ood_X_test, ood_y_test, LCM.CrossEntropyWithLogPrior, n=500, verbose=True)
            sd_ind = []
            sd_ood = []
            for g in ood_variances:
                sd_ood.append(np.sqrt(g))
            for g in variances:
                sd_ind.append(np.sqrt(g))

        log_path = r"C:\Users\user\DataspellProjects\BNNproject1\code\experiment_logs"

        logger, graph_path = self.create_logger(subj2_filename, dir=log_path)

        figpath = os.path.join(graph_path, 'train_curves.jpg')
        train_curves.savefig(figpath)

        buffer = str('DC parameters:'+'\n'+'weights: '+str(det_classifier.weights.weight)+'\n'+'bias: '+str(det_classifier.weights.bias)+'\n')
        print(buffer)
        logger.write(buffer)
        buffer = str('LCM parameters: '+'\n'+'weights: '+str(bayes_classifier.theta.weight)+'\n'+'covar_matrix: '+str(bayes_classifier.posterior_sigma)+'\n')
        print(buffer)
        logger.write(buffer)

        test_acc = np.sum(bay_preds == y_test, dtype=np.float) / len(y_test)
        buffer = str('Laplace modified classifier Testing accuracy: '+str(test_acc)+'\n')
        print(buffer)
        logger.write(buffer)
        test_acc = np.sum(det_preds == y_test, dtype=np.float) / len(y_test)
        buffer = str('Deterministic classifier Testing accuracy: '+str(test_acc)+'\n')
        print(buffer)
        logger.write(buffer)

        det_preds_in_domain = count_in_domain_accuracy(det_preds, y_test)
        bay_preds_in_domain = count_in_domain_accuracy(bay_preds, y_test)
        buffer = str('Deterministic classifier In-Domain Testing accuracy: '+str(det_preds_in_domain)+'\n')
        print(buffer)
        logger.write(buffer)
        buffer = str('Laplace modified classifier In-Domain Testing accuracy: '+str(bay_preds_in_domain)+'\n')
        print(buffer)
        logger.write(buffer)

        if subj2_filename is None:
            buffer = str('Mean variance for IND samples: ' + str(np.average(var_ind)) + '\n'+ 'Mean variance for OOD samples: ' + str(np.average(var_ood))+'\n')
            print(buffer)
            logger.write(buffer)
        else:
            buffer = str('Mean variance for IND samples: ' + str(np.average(variances)) + '\n'+ 'Mean variance for OOD samples: ' + str(np.average(ood_variances))+'\n')
            print(buffer)
            logger.write(buffer)

        sample_nums = [1, 5, 10, 25, 50, 75, 100, 150, 200, 250, 500]
        loss_delta = []
        for num in sample_nums:
            new_preds, total_loss, _ = LCM.predict(bayes_classifier, X_test, y_test, LCM.CrossEntropyWithLogPrior, n=num)
            loss_delta.append(total_loss-det_total_loss)

        if subj2_filename is not None:
            preds_ind = bay_preds
            preds_ood = ood_bay_preds
            var_ind = variances
            var_ood = ood_variances
            xs, ys, a = self.get_roc_data(var_ind, var_ood, 0.02, 0.155, logger)

        else:
            xs, ys, a = self.get_roc_data(var_ind, var_ood, 0.02, 0.155, logger)

        logger.close()

        fig3 = plt.figure()
        loss_discrepancy = fig3.add_subplot(111, title='loss delta', xlabel='n', ylabel='Loss_delta')
        loss_discrepancy.plot(sample_nums, loss_delta, 'b-', label='loss_delta vs n')
        loss_discrepancy.plot([0, 500], [0, 0], 'c--')
        figpath = os.path.join(graph_path, 'loss_delta_vs_n.jpg')
        fig3.savefig(figpath)

        fig4 = plt.figure()
        f_ind_variances = fig4.add_subplot(121, title='ind_preds', xlabel='sample', ylabel='pred+-sd')
        f_ind_variances.errorbar([n for n in range(len(preds_ind))], preds_ind, yerr=sd_ind, fmt='o')
        f_ind_variances.set_xlim([0, 20])
        f_ind_variances.set_ylim([-2.0, 2.0])
        f_ood_variances = fig4.add_subplot(122, title='ood_preds', xlabel='sample', ylabel='pred+-sd')
        f_ood_variances.errorbar([n for n in range(len(preds_ood))], preds_ood, yerr=sd_ood, fmt='o')
        f_ood_variances.set_xlim([0, 20])
        f_ood_variances.set_ylim([-2.0, 2.0])
        figpath = os.path.join(graph_path, 'ind_ood_variances.jpg')
        fig4.savefig(figpath)

        fig5 = plt.figure()
        roc = fig5.add_subplot(111, title='variance criteria roc')
        roc.plot(xs, ys, 'b-', label='roc, area = %0.3f' % a)
        roc.plot([0, 1], [0, 1], 'c--')
        roc.legend()
        figpath = os.path.join(graph_path, 'variance_roc.jpg')
        fig5.savefig(figpath)

        if subj2_filename is None:
            y_ind, y_ood = domain_split(y_test, y_test)
            vars_ind, vars_ood = domain_split(variances, y_test)
            pred_ind, pred_ood = domain_split(bay_preds, y_test)
            true_vars, false_vars = distinguish_variances(vars_ind, pred_ind, y_ind)

        else:
            true_vars, false_vars = distinguish_variances(variances, bay_preds, y_test)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        fig.suptitle('Variances distribution among truly and falsely classified samples')
        sns.histplot(ax=axes[0], x=true_vars, bins=7, kde=True, label='Truly classified')
        sns.histplot(ax=axes[1], x=false_vars, bins=7, kde=True, label='Falsely classified')
        figpath = os.path.join(graph_path, 'true_false_variances.jpg')
        fig.savefig(figpath)

        if subj2_filename is not None:
            var_ind = variances
            var_ood = ood_variances

        fig, ax = plt.subplots()
        fig.suptitle('Variances distribution among In-Domain and Out-of-Domain samples')

        sns.histplot(var_ind, bins=7, ax=ax, color='b', kde=True, label='In-Domain')
        sns.histplot(var_ood, bins=7, ax=ax, color='r', kde=True, label='Out-of-Domain')
        fig.legend(loc='center right')

        figpath = os.path.join(graph_path, 'variance_distribution.jpg')
        fig.savefig(figpath)

    def SGHMC_experiment(self, subj2_filename=None): # TODO set features, cache and weights to sample as arguments
        subj1_filename = self.file_to_load

        n_best_features = 8
        sample_cache = 20
        weights_to_sample = 120
        prior_rate = 150
        train_share = 0.75

        X_cv, y_cv, X_test, y_test, ood_X_test, ood_y_test = self.preprocess(subj1_filename, subj2_filename, n_best_features, train_share)

        sghmc_classifier = SGHMC.SGHMCClassifier(sample_cache=sample_cache, n_features_selected=n_best_features, prior_rate=prior_rate)

        #det_classifier = DC.DeterministicClassifier(n_features_selected=n_best_features)
        #train_curves = self.det_classifier_train(X_cv, y_cv, det_classifier)

        # TODO output of fit() and validate() ? Epochs?

        print('Sampling {} weights in {} batches'.format(weights_to_sample, weights_to_sample//sample_cache))

        print('Burn-in and first batch sampling')
        sghmc_losses, epochs = SGHMC.fit(sghmc_classifier, X_cv, y_cv, SGHMC.CrossEntropyWithLogPrior, keep_every=500, burn_in_steps=3000)
        print('Validation of first sample batch')

        sghmc_valid_loss, sghmc_valid_preds = SGHMC.validate(sghmc_classifier, X_test, y_test, SGHMC.CrossEntropyWithLogPrior)

        valid_losses = []
        valid_accuracy = []

        valid_losses.append(sghmc_valid_loss)
        valid_accuracy.append(count_in_domain_accuracy(sghmc_valid_preds, y_test))

        for i in range(weights_to_sample//sample_cache - 1):

            print('Sampling batch number: ', i)

            loss, epoch = SGHMC.fit(sghmc_classifier, X_cv, y_cv, SGHMC.CrossEntropyWithLogPrior, burn_in_steps=0)

            for item in loss:
                sghmc_losses.append(item)

            epochs += epoch

            print('Validating batch number: ', i)

            valid_loss, valid_preds = SGHMC.validate(sghmc_classifier, X_test, y_test, SGHMC.CrossEntropyWithLogPrior)

            valid_losses.append(valid_loss)
            valid_accuracy.append(count_in_domain_accuracy(valid_preds, y_test))

        log_path = r"C:\Users\user\DataspellProjects\BNNproject1\code\experiment_logs"

        logger, graph_path = self.create_logger(subj2_filename, dir=log_path)
        logger.write('!!!!!CONDUCTING SGHMC EXPERIMENT!!!!! \n')
        parameters_dict = {'n_best_features': n_best_features, 'weights_sampled': weights_to_sample,
                           'sample_batch_size': sample_cache, 'prior rate': prior_rate, 'test_share': 1-train_share}
        logger.write('Experiment parameters: ' + str(parameters_dict) + '\n')

        sghmc_train_curve = plt.figure()
        loss_curves = sghmc_train_curve.add_subplot(121, title='Loss Curves', xlabel='epoch', ylabel='Loss')
        loss_curves.plot([n for n in range(epochs)], sghmc_losses, 'r-', label='sghmc_train')
        loss_curves.legend()

        valid_acc = sghmc_train_curve.add_subplot(122, title='Validation accuracy', xlabel='epoch', ylabel='Accuracy')
        valid_acc.plot([n for n in range(weights_to_sample//sample_cache)], valid_accuracy, 'm-', label='sghmc_acc')
        valid_acc.legend()

        figpath = os.path.join(graph_path, 'train_curves.jpg')
        sghmc_train_curve.savefig(figpath)

        print('Predicting')
        sghmc_preds, sghmc_total_loss, sghmc_variances = SGHMC.predict(sghmc_classifier, X_test, y_test, SGHMC.CrossEntropyWithLogPrior, verbose=False)
        print('Accuracy: ', count_accuracy(sghmc_preds, y_test))
        logger.write('Accuracy: ' + str(count_accuracy(sghmc_preds, y_test)) + '\n')
        print('In-domain accuracy: ', count_in_domain_accuracy(sghmc_preds, y_test))
        logger.write('In-domain accuracy: ' + str(count_in_domain_accuracy(sghmc_preds, y_test)) + '\n')

        if subj2_filename is None:

            preds_ind, preds_ood = domain_split(sghmc_preds, y_test)
            var_ind, var_ood = domain_split(sghmc_variances, y_test)

        else:
            preds_ood, ood_sghmc_total_loss, var_ood = SGHMC.predict(sghmc_classifier, ood_X_test, ood_y_test, SGHMC.CrossEntropyWithLogPrior, verbose=False)
            print('Out-of-Domain Accuracy (for 2 subj only): ', count_accuracy(preds_ood, ood_y_test))
            logger.write('Out-of-Domain Accuracy (for 2 subj only): ' + str(count_accuracy(preds_ood, ood_y_test)) + '\n')
            preds_ind = sghmc_preds
            var_ind = sghmc_variances

        print('Mean IND variance: ', np.mean(var_ind))
        logger.write('Mean IND variance: ' + str(np.mean(var_ind)) + '\n')
        print('Mean OOD variance: ', np.mean(var_ood))
        logger.write('Mean OOD variance: ' + str(np.mean(var_ood)) + '\n')

        logger.close()

        sd_ind = [np.sqrt(n) for n in var_ind]
        sd_ood = [np.sqrt(n) for n in var_ood]

        fig4 = plt.figure()
        f_ind_variances = fig4.add_subplot(121, title='ind_preds', xlabel='sample', ylabel='pred+-sd')
        f_ind_variances.errorbar([n for n in range(len(preds_ind))], preds_ind, yerr=sd_ind, fmt='o')
        f_ind_variances.set_xlim([-0.5, 20])
        f_ind_variances.set_ylim([-2.0, 2.0])
        f_ood_variances = fig4.add_subplot(122, title='ood_preds', xlabel='sample', ylabel='pred+-sd')
        f_ood_variances.errorbar([n for n in range(len(preds_ood))], preds_ood, yerr=sd_ood, fmt='o')
        f_ood_variances.set_xlim([-0.5, 20])
        f_ood_variances.set_ylim([-2.0, 2.0])
        figpath = os.path.join(graph_path, 'ind_ood_variances.jpg')
        fig4.savefig(figpath)

        fig, ax = plt.subplots()
        fig.suptitle('Variances distribution among In-Domain and Out-of-Domain samples')

        sns.histplot(var_ind, bins=7, ax=ax, color='b', kde=True, label='In-Domain')
        sns.histplot(var_ood, bins=7, ax=ax, color='r', kde=True, label='Out-of-Domain')
        fig.legend(loc='center right')

        figpath = os.path.join(graph_path, 'variance_distribution.jpg')
        fig.savefig(figpath)

        if subj2_filename is None:
            y_ind, y_ood = domain_split(y_test, y_test)

        else:
            y_ind = y_test

        true_vars, false_vars = distinguish_variances(var_ind, preds_ind, y_ind)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        fig.suptitle('Variances distribution among truly and falsely classified samples')
        sns.histplot(ax=axes[0], x=true_vars, bins=7, kde=True, label='Truly classified')
        sns.histplot(ax=axes[1], x=false_vars, bins=7, kde=True, label='Falsely classified')
        figpath = os.path.join(graph_path, 'true_false_variances.jpg')
        fig.savefig(figpath)

        xs, ys, a = self.get_roc_data(var_ind, var_ood, 0.02, 0.155)

        fig5 = plt.figure()
        roc = fig5.add_subplot(111, title='variance criteria roc')
        roc.plot(xs, ys, 'b-', label='roc, area = %0.3f' % a)
        roc.plot([0, 1], [0, 1], 'c--')
        roc.legend()
        figpath = os.path.join(graph_path, 'variance_roc.jpg')
        fig5.savefig(figpath)

    def NeuroNetExperiment(self, network, weights_to_sample=120, subject_id=3): # TODO deprecate

        print('this method is deprecated. use NetExperiment instead')
        train_set, test_set = braindecode_preprocessing(subject_id)
        n_chans = train_set[0][0].shape[0]
        n_classes = 2
        input_window_samples = train_set[0][0].shape[1]

        print('dataset length: ', len(train_set))
        print('shape: ', train_set[0][0].shape)

        # y extraction
        X_train, y_train, windows_train = eeg_data_parser(train_set)
        X_test, y_test, windows_test = eeg_data_parser(test_set)

        X_train, y_train = clean_data(X_train, y_train)
        X_valid, y_valid = clean_data(X_test, y_test)

        print('X_train: ', X_train[0].shape)
        print('window: ', windows_train[0])


        if network == 'EEGNet':
            net = EEGNetV4.EEGNetv4(n_chans, n_classes, input_window_samples)
            #net = EEGNetv4(n_chans, n_classes, input_window_samples)
        elif network == 'ShallowFBCSPNet':
            net = ShallowFBCSPNet.ShallowFBCSPNet(n_chans, n_classes, input_window_samples, final_conv_length=69) # FIXME former was no final_conv_length
           #net = ShallowFBCSPNet(n_chans, n_classes, input_window_samples)

        var = 1.0

        prior = NeuroNetUtils.GaussianPrior(mu=0.0, std=var)
        likelihood = NeuroNetUtils.LikCategorical() # FIXME return to gaussan lik

        print('Initializing net weights.')
        prior.initialize(net)

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        #net.to(device)
        #likelihood.to(device)
        #prior.to(device)

        weights_in_batch = 10 # FIXME was 20

        model_weights = []
        valid_losses = []
        valid_accuracies = []

        optimizer = AdaptiveSGHMC(net.parameters(), lr=3.e-4) # FIXME was 3.e-4

        print('Sampling {} weights in {} batches'.format(weights_to_sample, weights_to_sample//weights_in_batch))
        print('Burn-in and first batch sampling')
        sampled_weights, losses, epochs, train_accuracies = NeuroNetUtils.fit(net, prior, likelihood, optimizer, weights_in_batch, X_train, y_train, keep_every=500, burn_in_steps=3000, device=device)
        print('Validation of first sample batch')
        valid_loss, valid_preds = NeuroNetUtils.validate(net, prior, likelihood, sampled_weights, X_valid, y_valid, device=device)
        valid_losses.append(valid_loss)
        valid_accuracies.append(count_in_domain_accuracy(valid_preds, y_valid))
        for weight in sampled_weights:
            model_weights.append(weight)

        for i in range(weights_to_sample//weights_in_batch - 1):

            print('Sampling batch number: ', i+2)
            sampled_weights, loss, epoch, train_accuracy = NeuroNetUtils.fit(net, prior, likelihood, optimizer, weights_in_batch, X_train, y_train, keep_every=500, burn_in_steps=0, device=device)
            print('Validating batch number: ', i+2)
            valid_loss, valid_preds = NeuroNetUtils.validate(net, prior, likelihood, sampled_weights, X_valid, y_valid, device=device)

            for weight in sampled_weights:
                model_weights.append(weight)
            for a in loss:
                losses.append(a)
            for acc in train_accuracy:
                train_accuracies.append(acc)

            epochs += epoch
            valid_losses.append(valid_loss)
            valid_accuracies.append(count_in_domain_accuracy(valid_preds, y_valid))

        # TODO look at train_accuracy

        print('Testing')
        test_preds, variances = NeuroNetUtils.predict(net, model_weights, X_test, y_test, device=device)

        preds_ind, preds_ood = domain_split(test_preds, y_test)
        var_ind, var_ood = domain_split(variances, y_test)
        y_ind, y_ood = domain_split(y_test, y_test)

        log_path = r"C:\Users\user\DataspellProjects\BNNproject1\code\experiment_logs"

        logger, graph_path = self.create_logger(subj2_filename=None, dir=log_path)
        logger.write('Using net architecture: ' + network)
        print('Accuracy: ', count_accuracy(test_preds, y_test))
        logger.write('Accuracy: ' + str(count_accuracy(test_preds, y_test)) + '\n')
        print('In-domain accuracy: ', count_in_domain_accuracy(test_preds, y_test))
        logger.write('In-domain accuracy: ' + str(count_in_domain_accuracy(test_preds, y_test)) + '\n')
        print('Mean IND variance: ', np.mean(var_ind))
        logger.write('Mean IND variance: ' + str(np.mean(var_ind)) + '\n')
        print('Mean OOD variance: ', np.mean(var_ood))
        logger.write('Mean OOD variance: ' + str(np.mean(var_ood)) + '\n')
        logger.close()

        net_train_curve = plt.figure()
        loss_curves = net_train_curve.add_subplot(131, title='Loss Curves', xlabel='epoch', ylabel='Loss')
        loss_curves.plot([n for n in range(epochs)], losses, 'r-', label='sghmc_train')
        loss_curves.legend()

        train_acc = net_train_curve.add_subplot(132, title='Train accuracy', xlabel='epoch', ylabel='Accuracy')
        train_acc.plot([n for n in range(epochs)], train_accuracies, 'm-', label='sghmc_acc')
        train_acc.legend()

        valid_acc = net_train_curve.add_subplot(133, title='Validation accuracy', xlabel='batch', ylabel='Accuracy')
        valid_acc.plot([n for n in range(weights_to_sample//weights_in_batch)], valid_accuracies, 'm-', label='sghmc_acc')
        valid_acc.legend()

        figpath = os.path.join(graph_path, 'train_curves.jpg')
        net_train_curve.savefig(figpath)

        sd_ind = [np.sqrt(n) for n in var_ind]
        sd_ood = [np.sqrt(n) for n in var_ood]

        fig4 = plt.figure()
        f_ind_variances = fig4.add_subplot(121, title='ind_preds', xlabel='sample', ylabel='pred+-sd')
        f_ind_variances.errorbar([n for n in range(len(preds_ind))], preds_ind, yerr=sd_ind, fmt='o')
        f_ind_variances.set_xlim([-0.5, 20])
        f_ind_variances.set_ylim([-2.0, 2.0])
        f_ood_variances = fig4.add_subplot(122, title='ood_preds', xlabel='sample', ylabel='pred+-sd')
        f_ood_variances.errorbar([n for n in range(len(preds_ood))], preds_ood, yerr=sd_ood, fmt='o')
        f_ood_variances.set_xlim([-0.5, 20])
        f_ood_variances.set_ylim([-2.0, 2.0])
        figpath = os.path.join(graph_path, 'ind_ood_variances.jpg')
        fig4.savefig(figpath)

        true_vars, false_vars = distinguish_variances(var_ind, preds_ind, y_ind)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        fig.suptitle('Variances distribution among truly and falsely classified samples')
        sns.histplot(ax=axes[0], x=true_vars, bins=7, kde=True, label='Truly classified')
        sns.histplot(ax=axes[1], x=false_vars, bins=7, kde=True, label='Falsely classified')
        figpath = os.path.join(graph_path, 'true_false_variances.jpg')
        fig.savefig(figpath)

        fig, ax = plt.subplots()
        fig.suptitle('Variances distribution among In-Domain and Out-of-Domain samples')

        sns.histplot(var_ind, bins=7, ax=ax, color='b', kde=True, label='In-Domain')
        sns.histplot(var_ood, bins=7, ax=ax, color='r', kde=True, label='Out-of-Domain')
        fig.legend(loc='center right')

        figpath = os.path.join(graph_path, 'variance_distribution.jpg')
        fig.savefig(figpath)

        xs, ys, a = self.get_roc_data(var_ind, var_ood, 0.02, 0.8) # FIXME former was 0.155

        fig5 = plt.figure()
        roc = fig5.add_subplot(111, title='variance criteria roc')
        roc.plot(xs, ys, 'b-', label='roc, area = %0.3f' % a)
        roc.plot([0, 1], [0, 1], 'c--')
        roc.legend()
        figpath = os.path.join(graph_path, 'variance_roc.jpg')
        fig5.savefig(figpath)

    def NetExperiment(self, network, prior_dict, sampler_config, subject_id=3, experiment_name='', domain=[0,1]):
        train_set, test_set, channel_locations = braindecode_preprocessing(subject_id)
        n_chans = train_set[0][0].shape[0]
        n_classes = 2
        input_window_samples = train_set[0][0].shape[1]

        print('dataset length: ', len(train_set))
        print('shape: ', train_set[0][0].shape)

        # y extraction
        # TODO typecast: list of ndarrays -> nparray or nd array of nd arrays -> torch tensor
        X_train, y_train, windows_train = eeg_data_parser(train_set)
        X_test, y_test, windows_test = eeg_data_parser(test_set)
        y_train, y_test = swap_class_labels(y_train, y_test, 0, 2)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)

        X_train, y_train = clean_data(X_train, y_train, domain)
        X_valid, y_valid = clean_data(X_test, y_test, domain)
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)

        print('X_train: ', X_train[0].shape)
        print('X_train: ', X_train[0][0].shape)
        print('window: ', windows_train[0])
        hook_handles = []

        if network == 'EEGNet':
            net_params = {'in_chans': train_set[0][0].shape[0], 'n_classes': 2, 'input_window_samples': train_set[0][0].shape[1]}
            net = NeuroNetUtils.NewEEGNet(net_params)
            time_coordinates = [n for n in range(net.kernel_length)]
            hook_func = NeuroNetUtils.SaveEEGNetLogits()
            handle = net.conv_classifier.register_forward_hook(hook_func)
            hook_handles.append(handle)

        elif network == 'ShallowFBCSPNet':
            net_params = {'in_chans': train_set[0][0].shape[0], 'n_classes': 2, 'input_window_samples': train_set[0][0].shape[1], 'final_conv_length': 69}
            net = NeuroNetUtils.NewShallowFBCSPNet(net_params) # FIXME former was no final_conv_length
            time_coordinates = [n for n in range(net.filter_time_length)]
            hook_func = NeuroNetUtils.SaveShallowFBCSPNetLogits()
            handle = net.conv_classifier.register_forward_hook(hook_func)
            hook_handles.append(handle)

        fc_prior = prior_dict['fc']
        default_prior = prior_dict['default']

        spat_coordinates = channel_locations

        if isinstance(prior_dict['time'], float):
            time_prior = CorrelatedGaussianPrior(mu=torch.zeros(len(time_coordinates)), coordinates=time_coordinates, scale=prior_dict['time'])
        else:
            time_prior = prior_dict['time']

        if isinstance(prior_dict['spat'], float):
            spat_prior = CorrelatedGaussianPrior(mu=torch.zeros(len(spat_coordinates)), coordinates=spat_coordinates, scale=prior_dict['spat'])
        else:
            spat_prior = prior_dict['spat']

        prior = CombinedPrior(default_prior, fc_prior, time_prior, spat_prior)
        #likelihood = XELLikelihood()
        likelihood = LikCategorical()

        sampler_batch_size = len(train_set)

        sampler_config['batch_size'] = sampler_batch_size
        log_path = r"C:\Users\user\DataspellProjects\BNNproject1\code\experiment_logs"
        prefix = experiment_name
        filename = prefix + network + '_subject_' + str(subject_id)
        logger, logger_path = self.create_logger(None, log_path, filename)

        logger.write(str(sampler_config) + '\n')
        logger.write(str(prior_dict) + '\n')
        #logger.write('Prior variance: ' + str(std) + '\n')
        logger.write('Subject id: ' + str(subject_id) + '\n')

        samples_dir = os.path.join(logger_path, 'samples')
        util.ensure_dir(samples_dir)

        bayes_net = ClassificationNet(net, likelihood, prior, logger_path, n_gpu=1)

        train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train),
                                                 batch_size=sampler_batch_size, shuffle=True)

        test_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test),
                                                       batch_size=sampler_batch_size, shuffle=True)

        bayes_net.sample_multi_chains(data_loader=train_dataloader, **sampler_config)

        # mean_preds, predictions = bayes_net.evaluate(train_dataloader, True, True)
        all_logits = []
        net.conv_classifier.register_forward_hook(hook_func)

        preds = []
        pred_means = []
        targets = []
        with torch.no_grad():
            for i, (data, target) in enumerate(test_dataloader):
                pred_mean, pred = bayes_net.predict(
                    data, return_individual_predictions=True,
                    all_sampled_weights=True)
                #print(len(hook_func.logits)) # 48
                #print(len(pred_mean)) # 288
                #print(len(pred)) # 6
                #print(hook_func.logits[0].size()) # 144x2

                all_logits.append(hook_func.logits)
                pred_means.append(pred_mean)
                preds.append(pred)
                targets.append(target)
                hook_func.clear()

        total_logits = []
        [total_logits.append(x) for x in all_logits[0] if x.size(0) == 288] # TODO replace 288 eith len(test_set)
        total_logits = torch.stack(total_logits, dim=0)
        total_logits = torch.unique(total_logits, dim=0)
        logit_variances = torch.var(total_logits, dim=0)
        logit_variances = logit_variances.numpy() # size: 288x2

        print('logit variances \n', logit_variances)

        print('\n', logit_variances.shape)

        handle.remove()

        pred_means = torch.cat(pred_means, dim=0).cpu().numpy()
        preds = torch.cat(preds, dim=1).cpu().numpy()
        targets = torch.cat(targets, dim=0).cpu().numpy()

        mean_predictions = np.argmax(pred_means, axis=1)

        print(network, 'In-Domain Accuracy: ', count_in_domain_accuracy(mean_predictions, targets, domain))
        logger.write(str(network) + ' In-Domain Accuracy: ' + str(count_in_domain_accuracy(mean_predictions, targets, domain)) + '\n')

        variances = np.var(preds, axis=0)
        variances = np.transpose(variances)[0]

        logger.write('Variances \n' + str(list(variances)) + '\n')
        logger.write('Predictions \n' + str(list(mean_predictions)) + '\n')
        logger.write('targets \n' + str(list(targets)) + '\n')

        ind_variance, ood_variance = domain_split(variances, targets, domain)
        true_variance, false_variance = distinguish_variances(variances, mean_predictions, targets)
        ind_predictions, ood_predictions = domain_split(mean_predictions, targets, domain)
        ind_targets, _ = domain_split(targets, targets, domain)
        ind_true_variance, ind_false_variance = distinguish_variances(ind_variance, ind_predictions, ind_targets)

        print('Mean Variance for IND datapoints', np.mean(ind_variance))
        logger.write('Mean Variance for IND datapoints: ' + str(np.mean(ind_variance)) + '\n')
        print('Mean Variance for OOD datapoints', np.mean(ood_variance))
        logger.write('Mean Variance for OOD datapoints: ' + str(np.mean(ood_variance)) + '\n')

        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        fig.suptitle('Variances distribution among truly and falsely classified IND datapoints') # FIXME maybe be better to plot 2 hists on one plot
        sns.histplot(ax=axes[0], x=ind_true_variance, bins=7, kde=True, label='Truly classified')
        sns.histplot(ax=axes[1], x=ind_false_variance, bins=7, kde=True, label='Falsely classified')
        figpath = os.path.join(logger_path, 'true_false_variances.jpg')
        fig.savefig(figpath)

        fig, ax = plt.subplots()
        fig.suptitle('Variances distribution among In-Domain and Out-of-Domain samples')
        sns.histplot(ind_variance, bins=7, ax=ax, color='b', kde=True, label='In-Domain')
        sns.histplot(ood_variance, bins=7, ax=ax, color='r', kde=True, label='Out-of-Domain')
        fig.legend(loc='center right')
        figpath = os.path.join(logger_path, 'IND_OOD_variances.jpg')
        fig.savefig(figpath)

        fpr, tpr, threshold = roc_curve(ind_targets, ind_predictions)
        roc_auc = auc(fpr, tpr)

        fig = plt.figure()
        roc = fig.add_subplot(111, title='In-Domain ROC')
        roc.plot(fpr, tpr, 'b-', label='roc, area = %0.3f' % roc_auc)
        roc.plot([0, 1], [0, 1], 'c--')
        roc.legend()
        figpath = os.path.join(logger_path, 'IND_ROC.jpg')
        fig.savefig(figpath)

        logger.write('IND AUC: ' + str(roc_auc) + '\n')

        xs, ys, a = self.get_roc_data(ind_variance, ood_variance, -1.0, 1.0)

        logger.write('OOD detection AUC: ' + str(a) + '\n')

        fig = plt.figure()
        roc = fig.add_subplot(111, title='variance criteria roc')
        roc.plot(xs, ys, 'b-', label='roc, area = %0.3f' % a)
        roc.plot([0, 1], [0, 1], 'c--')
        roc.legend()
        figpath = os.path.join(logger_path, 'variance_OOD_ROC.jpg')
        fig.savefig(figpath)

        # FIXME fix the scatterplot plotting script below

        fig = plt.figure()
        scatter = fig.add_subplot(111, title='IND/OOD variances distribution')

        marks = ['bo', 'ro']
        labels = ['IND', 'OOD']
        # logit_variance size: 288x2
        # domain_split(var:list, targets:list) -> ind_var:list, ood_var:list
        ind_var, ood_var = domain_split(logit_variances, targets, domain)
        scatter.plot(ind_var, marks[0], label=labels[0])
        scatter.plot(ood_var, marks[1], label=labels[1])
        scatter.legend()
        figpath = os.path.join(logger_path, 'variance_IND_OOD_scatter_BROKEN.jpg')
        fig.savefig(figpath)

        var_norms = []
        for var_vector in logit_variances:
            var_norm = np.linalg.norm(var_vector) #calculate norm for var_vector
            var_norms.append(var_norm)

        logger.write('Logit Variances Norms \n' + str(list(var_norms)) + '\n')

        ind_norms, ood_norms = domain_split(var_norms, targets, domain)

        xs, ys, a = self.get_roc_data(ind_norms, ood_norms, -1.0, max(var_norms), max(var_norms)/len(var_norms))

        logger.write('OOD detection AUC using logit var: ' + str(a) + '\n')

        fig = plt.figure()
        roc = fig.add_subplot(111, title='variance criteria roc usling logit var')
        roc.plot(xs, ys, 'b-', label='roc, area = %0.3f' % a)
        roc.plot([0, 1], [0, 1], 'c--')
        roc.legend()
        figpath = os.path.join(logger_path, 'logit_variance_OOD_ROC.jpg')
        fig.savefig(figpath)

        logger.close()

    def EnsembleExperiment(self, network, n_epochs, number_of_nets, subject_id=3, experiment_name='', domain=[0,1]):

        # TODO use logits and logit variances to plot ROC-curve as in NetExperiment

        train_set, test_set, channel_locations = braindecode_preprocessing(subject_id)
        n_chans = train_set[0][0].shape[0]
        n_classes = 2
        input_window_samples = train_set[0][0].shape[1]

        print('dataset length: ', len(train_set))
        print('shape: ', train_set[0][0].shape)

        # y extraction
        X_train, y_train, windows_train = eeg_data_parser(train_set)
        X_test, y_test, windows_test = eeg_data_parser(test_set)

        y_train, y_test = swap_class_labels(y_train, y_test, 0, 2)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)

        X_train, y_train = clean_data(X_train, y_train, domain)
        X_valid, y_valid = clean_data(X_test, y_test, domain)
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)

        cuda = torch.cuda.is_available()
        device = "cuda" if cuda else "cpu"
        if cuda:
            torch.backends.cudnn.benchmark = True

        lr = 0.0625 * 0.01
        weight_decay = 0
        batch_size = 64

        total_preds = []
        total_probabilities = []
        all_logits = []
        for i in range(number_of_nets):
            print('Model', i+1, 'out of', number_of_nets)
            seed = i
            set_random_seeds(seed=seed, cuda=cuda)

            if network == 'EEGNet':
                model = EEGNetv4(n_chans, n_classes, input_window_samples)
                hook_func = NeuroNetUtils.SaveEEGNetLogits()

            if network == 'ShallowFBCSPNet':
                model = ShallowFBCSPNet.ShallowFBCSPNet(n_chans, n_classes, input_window_samples, final_conv_length=69)
                hook_func = NeuroNetUtils.SaveShallowFBCSPNetLogits()

            model.cuda()

            clf = EEGClassifier(
                model,
            criterion=torch.nn.NLLLoss,
            optimizer=torch.optim.AdamW,
            train_split=None,
            optimizer__lr=lr,
            optimizer__weight_decay=weight_decay,
            batch_size=batch_size,
            callbacks=[
                "accuracy",
                ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)),
            ],
            device=device)

            clf.fit(X_train, y=y_train, epochs=n_epochs)

            handle = model.conv_classifier.register_forward_hook(hook_func)

            probabilities = clf.predict_proba(X_test)

            hook_func.clear()

            predictions = clf.predict(X_test)

            total_preds.append(predictions)

            total_probabilities.append(probabilities)
            all_logits.append(torch.cat(hook_func.logits, dim=0))

            handle.remove()

        log_path = r"C:\Users\user\DataspellProjects\BNNproject1\code\experiment_logs"
        prefix = experiment_name
        filename = prefix + network + '_subject_' + str(subject_id)
        logger, logger_path = self.create_logger(None, log_path, filename)

        logger.write('Architecture: ' + str(network) + '\n')
        logger.write('N nets: ' + str(number_of_nets) + '\n')
        logger.write('n_epochs: ' + str(n_epochs) + '\n')

        total_preds = np.array(total_preds)
        total_probabilities = np.array(total_probabilities)
        mean_predictions = np.mean(total_preds, axis=0)
        variances = np.var(total_probabilities, axis=0)
        mean_predictions = np.round(mean_predictions)
        variances = np.transpose(variances)[0]
        targets = y_test

        print(network, 'In-Domain Accuracy: ', count_in_domain_accuracy(mean_predictions, targets, domain))
        logger.write(str(network) + ' In-Domain Accuracy: ' + str(count_in_domain_accuracy(mean_predictions, targets, domain)) + '\n')

        ind_variance, ood_variance = domain_split(variances, targets, domain)
        true_variance, false_variance = distinguish_variances(variances, mean_predictions, targets)
        ind_predictions, ood_predictions = domain_split(mean_predictions, targets, domain)
        ind_targets, _ = domain_split(targets, targets, domain)
        ind_true_variance, ind_false_variance = distinguish_variances(ind_variance, ind_predictions, ind_targets)

        print('Mean Variance for IND datapoints', np.mean(ind_variance))
        logger.write('Mean Variance for IND datapoints: ' + str(np.mean(ind_variance)) + '\n')
        print('Mean Variance for OOD datapoints', np.mean(ood_variance))
        logger.write('Mean Variance for OOD datapoints: ' + str(np.mean(ood_variance)) + '\n')

        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        fig.suptitle('Variances distribution among truly and falsely classified IND datapoints') # FIXME maybe be better to plot 2 hists on one plot
        sns.histplot(ax=axes[0], x=ind_true_variance, bins=7, kde=True, label='Truly classified')
        sns.histplot(ax=axes[1], x=ind_false_variance, bins=7, kde=True, label='Falsely classified')
        figpath = os.path.join(logger_path, 'true_false_variances.jpg')
        fig.savefig(figpath)

        fig, ax = plt.subplots()
        fig.suptitle('Variances distribution among In-Domain and Out-of-Domain samples')
        sns.histplot(ind_variance, bins=7, ax=ax, color='b', kde=True, label='In-Domain')
        sns.histplot(ood_variance, bins=7, ax=ax, color='r', kde=True, label='Out-of-Domain')
        fig.legend(loc='center right')
        figpath = os.path.join(logger_path, 'IND_OOD_variances.jpg')
        fig.savefig(figpath)

        fpr, tpr, threshold = roc_curve(ind_targets, ind_predictions)
        roc_auc = auc(fpr, tpr)

        fig = plt.figure()
        roc = fig.add_subplot(111, title='In-Domain ROC')
        roc.plot(fpr, tpr, 'b-', label='roc, area = %0.3f' % roc_auc)
        roc.plot([0, 1], [0, 1], 'c--')
        roc.legend()
        figpath = os.path.join(logger_path, 'IND_ROC.jpg')
        fig.savefig(figpath)

        logger.write('IND AUC: ' + str(roc_auc) + '\n')

        xs, ys, a = self.get_roc_data(ind_variance, ood_variance, -1.0, 10.0)

        logger.write('OOD detection AUC: ' + str(a) + '\n')

        fig = plt.figure()
        roc = fig.add_subplot(111, title='variance criteria roc')
        roc.plot(xs, ys, 'b-', label='roc, area = %0.3f' % a)
        roc.plot([0, 1], [0, 1], 'c--')
        roc.legend()
        figpath = os.path.join(logger_path, 'variance_OOD_ROC.jpg')
        fig.savefig(figpath)

        for i in range(len(all_logits)): # FIXME use torch stack???
            all_logits[i] = torch.unsqueeze(all_logits[i], -1)
        all_logits = torch.cat(all_logits, dim=2).cpu()
        logit_variances = torch.var(all_logits, dim=2)
        logit_variances = logit_variances.numpy()
        # logit_variances = np.transpose(logit_variances)
        # TODO measure size
        print('logit_variances: ', logit_variances.shape) # was 2x288, need 288x2
        print(logit_variances)

        fig = plt.figure()
        scatter = fig.add_subplot(111, title='IND/OOD variances distribution')

        marks = ['bo', 'ro']
        labels = ['IND', 'OOD']
        ind_var, ood_var = domain_split(logit_variances, targets, domain)
        scatter.plot(ind_var, marks[0], label=labels[0])
        scatter.plot(ood_var, marks[1], label=labels[1])
        scatter.legend()
        figpath = os.path.join(logger_path, 'variance_IND_OOD_scatter_BROKEN.jpg')
        fig.savefig(figpath)

        # TODO plot ROC-curve using logits data
        var_norms = []
        for var_vector in logit_variances:
            var_norm = np.linalg.norm(var_vector) #calculate norm for var_vector
            var_norms.append(var_norm)

        ind_norms, ood_norms = domain_split(var_norms, targets, domain)

        xs, ys, a = self.get_roc_data(ind_norms, ood_norms, -1.0, 10.0)

        logger.write('OOD detection AUC using logit var: ' + str(a) + '\n')

        fig = plt.figure()
        roc = fig.add_subplot(111, title='variance criteria roc usling logit var')
        roc.plot(xs, ys, 'b-', label='roc, area = %0.3f' % a)
        roc.plot([0, 1], [0, 1], 'c--')
        roc.legend()
        figpath = os.path.join(logger_path, 'logit_variance_OOD_ROC.jpg')
        fig.savefig(figpath)

        logger.close()

    def FullsetExperiment(self, network, prior_dict, sampler_config, subject_id=3, experiment_name=''):
        train_set, test_set, channel_locations = braindecode_preprocessing(subject_id)
        n_chans = train_set[0][0].shape[0]
        n_classes = 4
        input_window_samples = train_set[0][0].shape[1]

        print('dataset length: ', len(train_set))
        print('shape: ', train_set[0][0].shape)

        # y extraction
        X_train, y_train, windows_train = eeg_data_parser(train_set)
        X_test, y_test, windows_test = eeg_data_parser(test_set)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)

        #X_train, y_train = clean_data(X_train, y_train)
        #X_valid, y_valid = clean_data(X_test, y_test)
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)

        print('X_train: ', X_train[0].shape)
        print('window: ', windows_train[0])
        hook_handles = []

        if network == 'EEGNet':
            net_params = {'in_chans': train_set[0][0].shape[0], 'n_classes': n_classes, 'input_window_samples': train_set[0][0].shape[1]}
            net = NeuroNetUtils.NewEEGNet(net_params)
            time_coordinates = [n for n in range(net.kernel_length)]
            hook_func = NeuroNetUtils.SaveEEGNetLogits()
            handle = net.conv_classifier.register_forward_hook(hook_func)
            hook_handles.append(handle)

        elif network == 'ShallowFBCSPNet':
            net_params = {'in_chans': train_set[0][0].shape[0], 'n_classes': n_classes,
                          'input_window_samples': train_set[0][0].shape[1], 'final_conv_length': 69}

            net = NeuroNetUtils.NewShallowFBCSPNet(net_params) # FIXME former was no final_conv_length
            time_coordinates = [n for n in range(net.filter_time_length)]
            hook_func = NeuroNetUtils.SaveShallowFBCSPNetLogits()
            handle = net.conv_classifier.register_forward_hook(hook_func)
            hook_handles.append(handle)

        fc_prior = prior_dict['fc']
        default_prior = prior_dict['default']

        spat_coordinates = channel_locations

        if isinstance(prior_dict['time'], float):
            time_prior = CorrelatedGaussianPrior(mu=torch.zeros(len(time_coordinates)), coordinates=time_coordinates, scale=prior_dict['time'])
        else:
            time_prior = prior_dict['time']

        if isinstance(prior_dict['spat'], float):
            spat_prior = CorrelatedGaussianPrior(mu=torch.zeros(len(spat_coordinates)), coordinates=spat_coordinates, scale=prior_dict['spat'])
        else:
            spat_prior = prior_dict['spat']

        prior = CombinedPrior(default_prior, fc_prior, time_prior, spat_prior)
        #likelihood = XELLikelihood()
        likelihood = LikCategorical()

        sampler_batch_size = len(train_set)

        sampler_config['batch_size'] = sampler_batch_size
        log_path = r"C:\Users\user\DataspellProjects\BNNproject1\code\experiment_logs"
        prefix = experiment_name
        filename = prefix + network + '_subject_' + str(subject_id)
        logger, logger_path = self.create_logger(None, log_path, filename)

        logger.write(str(sampler_config) + '\n')
        logger.write(str(prior_dict) + '\n')
        #logger.write('Prior variance: ' + str(std) + '\n')
        logger.write('Subject id: ' + str(subject_id) + '\n')

        samples_dir = os.path.join(logger_path, 'samples')
        util.ensure_dir(samples_dir)

        bayes_net = ClassificationNet(net, likelihood, prior, logger_path, n_gpu=1)

        train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train),
                                                       batch_size=sampler_batch_size, shuffle=True)

        test_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test),
                                                      batch_size=sampler_batch_size, shuffle=True)

        bayes_net.sample_multi_chains(data_loader=train_dataloader, **sampler_config)

        # mean_preds, predictions = bayes_net.evaluate(train_dataloader, True, True)
        all_logits = []
        net.conv_classifier.register_forward_hook(hook_func)

        preds = []
        pred_means = []
        targets = []
        with torch.no_grad():
            for i, (data, target) in enumerate(test_dataloader):
                pred_mean, pred = bayes_net.predict(
                    data, return_individual_predictions=True,
                    all_sampled_weights=True)
                #print(len(hook_func.logits)) # 48
                #print(len(pred_mean)) # 288
                #print(len(pred)) # 6
                #print(hook_func.logits[0].size()) # 144x2

                all_logits.append(hook_func.logits)
                pred_means.append(pred_mean)
                preds.append(pred)
                targets.append(target)
                hook_func.clear()

        total_logits = []
        [total_logits.append(x) for x in all_logits[0] if x.size(0) == 288] # TODO replace 288 eith len(test_set)
        total_logits = torch.stack(total_logits, dim=0)
        total_logits = torch.unique(total_logits, dim=0)
        logit_variances = torch.var(total_logits, dim=0)
        logit_variances = logit_variances.numpy()
        logit_variances = np.transpose(logit_variances)

        handle.remove()

        pred_means = torch.cat(pred_means, dim=0).cpu().numpy()
        preds = torch.cat(preds, dim=1).cpu().numpy()
        targets = torch.cat(targets, dim=0).cpu().numpy()

        mean_predictions = np.argmax(pred_means, axis=1)

        print(network, 'In-Domain Accuracy: ', count_accuracy(mean_predictions, targets))
        logger.write(str(network) + ' In-Domain Accuracy: ' + str(count_accuracy(mean_predictions, targets)) + '\n')

        variances = np.var(preds, axis=0)
        variances = np.transpose(variances)[0]

        ind_variance, ood_variance = domain_split(variances, targets)
        true_variance, false_variance = distinguish_variances(variances, mean_predictions, targets)
        ind_predictions, ood_predictions = domain_split(mean_predictions, targets)
        ind_targets, _ = domain_split(targets, targets)
        ind_true_variance, ind_false_variance = distinguish_variances(ind_variance, ind_predictions, ind_targets)

        print('Mean Variance for IND datapoints', np.mean(ind_variance))
        logger.write('Mean Variance for IND datapoints: ' + str(np.mean(ind_variance)) + '\n')
        print('Mean Variance for OOD datapoints', np.mean(ood_variance))
        logger.write('Mean Variance for OOD datapoints: ' + str(np.mean(ood_variance)) + '\n')

        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        fig.suptitle('Variances distribution among truly and falsely classified IND datapoints') # FIXME maybe be better to plot 2 hists on one plot
        sns.histplot(ax=axes[0], x=true_variance, bins=7, kde=True, label='Truly classified')
        sns.histplot(ax=axes[1], x=false_variance, bins=7, kde=True, label='Falsely classified')
        figpath = os.path.join(logger_path, 'true_false_variances.jpg')
        fig.savefig(figpath)

        logger.close()

    def FullsetDetExperiment(self, network, n_epochs, number_of_nets=1, subject_id=3, experiment_name=''):
        train_set, test_set, channel_locations = braindecode_preprocessing(subject_id)
        n_chans = train_set[0][0].shape[0]
        n_classes = 4
        input_window_samples = train_set[0][0].shape[1]

        print('dataset length: ', len(train_set))
        print('shape: ', train_set[0][0].shape)

        # y extraction
        X_train, y_train, windows_train = eeg_data_parser(train_set)
        X_test, y_test, windows_test = eeg_data_parser(test_set)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)

        #X_train, y_train = clean_data(X_train, y_train)
        #X_valid, y_valid = clean_data(X_test, y_test)
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)

        cuda = torch.cuda.is_available()
        device = "cuda" if cuda else "cpu"
        if cuda:
            torch.backends.cudnn.benchmark = True

        lr = 0.0625 * 0.01
        weight_decay = 0
        batch_size = 64

        total_preds = []
        total_probabilities = []
        all_logits = []
        for i in range(number_of_nets):
            print('Model', i+1, 'out of', number_of_nets)
            #seed = i
            #set_random_seeds(seed=seed, cuda=cuda)

            if network == 'EEGNet':
                model = EEGNetv4(n_chans, n_classes, input_window_samples)
                hook_func = NeuroNetUtils.SaveEEGNetLogits()

            if network == 'ShallowFBCSPNet':
                model = ShallowFBCSPNet.ShallowFBCSPNet(n_chans, n_classes, input_window_samples, final_conv_length=69)
                hook_func = NeuroNetUtils.SaveShallowFBCSPNetLogits()

            model.cuda()

            clf = EEGClassifier(
                model,
                criterion=torch.nn.NLLLoss,
                optimizer=torch.optim.AdamW,
                train_split=None,
                optimizer__lr=lr,
                optimizer__weight_decay=weight_decay,
                batch_size=batch_size,
                callbacks=[
                    "accuracy",
                    ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)),
                ],
                device=device)

            clf.fit(X_train, y=y_train, epochs=n_epochs)

            handle = model.conv_classifier.register_forward_hook(hook_func)

            probabilities = clf.predict_proba(X_test)

            hook_func.clear()

            predictions = clf.predict(X_test)

            total_preds.append(predictions)

            total_probabilities.append(probabilities)
            all_logits.append(torch.cat(hook_func.logits, dim=0))

            handle.remove()

        log_path = r"C:\Users\user\DataspellProjects\BNNproject1\code\experiment_logs"
        prefix = experiment_name
        filename = prefix + network + '_subject_' + str(subject_id)
        logger, logger_path = self.create_logger(None, log_path, filename)

        logger.write('Architecture: ' + str(network) + '\n')
        logger.write('N nets: ' + str(number_of_nets) + '\n')
        logger.write('n_epochs: ' + str(n_epochs) + '\n')

        total_preds = np.array(total_preds)
        print(total_preds)
        total_probabilities = np.array(total_probabilities)
        mean_predictions = np.mean(total_preds, axis=0)
        variances = np.var(total_probabilities, axis=0)
        mean_predictions = np.round(mean_predictions)
        variances = np.transpose(variances)[0]
        targets = y_test

        print(network, 'In-Domain Accuracy: ', count_accuracy(predictions, targets))
        logger.write(str(network) + ' In-Domain Accuracy: ' + str(count_accuracy(predictions, targets)) + '\n')

        ind_variance, ood_variance = domain_split(variances, targets)
        true_variance, false_variance = distinguish_variances(variances, mean_predictions, targets)
        ind_predictions, ood_predictions = domain_split(mean_predictions, targets)
        ind_targets, _ = domain_split(targets, targets)
        ind_true_variance, ind_false_variance = distinguish_variances(ind_variance, ind_predictions, ind_targets)

        print('Mean Variance for IND datapoints', np.mean(ind_variance))
        logger.write('Mean Variance for IND datapoints: ' + str(np.mean(ind_variance)) + '\n')
        print('Mean Variance for OOD datapoints', np.mean(ood_variance))
        logger.write('Mean Variance for OOD datapoints: ' + str(np.mean(ood_variance)) + '\n')

        logger.close()

    def HyperOptimizationExperiment(self, network, params_to_optimize, prior_dict, sampler_config, subject_id=3, experiment_name='', domain=[0,1]):
        # TODO make hyperopt exp using optuna
        pass

    def HyperOptimizationDetExperiment(self, network, n_epochs, number_of_nets=1, subject_id=3, experiment_name='', domain=[0,1]):
        # TODO make hyperopt exp using optuna
        pass

    def det_neuronet_experiment(self, network, epochs, subject_id=3):
        train_set, test_set = braindecode_preprocessing(subject_id)
        n_chans = train_set[0][0].shape[0]
        n_classes = 2
        input_window_samples = train_set[0][0].shape[1]

        print('dataset length: ', len(train_set))
        print('shape: ', train_set[0][0].shape)

        # y extraction
        X_train, y_train, windows_train = eeg_data_parser(train_set)
        X_test, y_test, windows_test = eeg_data_parser(test_set)

        X_train, y_train = clean_data(X_train, y_train)

        print('X_train: ', X_train[0].shape)
        print('window: ', windows_train[0])


        if network == 'EEGNet':
            net = EEGNetV4.EEGNetv4(n_chans, n_classes, input_window_samples)
            #net = EEGNetv4(n_chans, n_classes, input_window_samples)
        elif network == 'ShallowFBCSPNet':
            net = ShallowFBCSPNet.ShallowFBCSPNet(n_chans, n_classes, input_window_samples, final_conv_length=69) # FIXME former was no final_conv_length
        #net = ShallowFBCSPNet(n_chans, n_classes, input_window_samples)

        var = 1.0

        prior = NeuroNetUtils.GaussianPrior(mu=0.0, std=var)
        likelihood = NeuroNetUtils.GaussianLikelihood(var=var*var)

        print('Initializing net weights.')
        prior.initialize(net)

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        net.to(device)

        losses = []
        train_accuracies = []
        test_predictions = []
        optimizer = AdamW(net.parameters(), lr=3.e-4)

        print('Fitting')
        losses, train_accuracies = NeuroNetUtils.deterministic_fit(net, prior, likelihood, optimizer, X_train, y_train, epochs=epochs, device=device)
        print('Testing')
        test_predictions = NeuroNetUtils.deterministic_predict(net, X_test, y_test, device=device)

        log_path = r"C:\Users\user\DataspellProjects\BNNproject1\code\experiment_logs"

        logger, graph_path = self.create_logger(None, log_path)

        logger.write('Conducting deterministic experiment using net: ' + network + '\n')
        print('Accuracy:', count_accuracy(test_predictions, y_test))
        print('In-Domain Accuracy:', count_in_domain_accuracy(test_predictions, y_test))

        logger.write('Accuracy: ' + str(count_accuracy(test_predictions, y_test)) + '\n')
        logger.write('In-Domain Accuracy: ' + str(count_in_domain_accuracy(test_predictions, y_test)) + '\n')

        net_train_curve = plt.figure()
        loss_curves = net_train_curve.add_subplot(121, title='Loss Curves', xlabel='epoch', ylabel='Loss')
        loss_curves.plot([n for n in range(epochs)], losses, 'r-', label='sghmc_train')
        loss_curves.legend()

        train_acc = net_train_curve.add_subplot(122, title='Training accuracy', xlabel='epoch', ylabel='Accuracy')
        train_acc.plot([n for n in range(epochs)], train_accuracies, 'm-', label='train_acc')
        train_acc.legend()

        figpath = os.path.join(graph_path, 'train_curves.jpg')
        net_train_curve.savefig(figpath)

    def preprocess(self, subj1_filename, subj2_filename=None, n_best_features=8, train_share=0.8):

        bcic_data1 = LoadData.LoadBCIC(subj1_filename, self.data_path)
        filtered1, y_labels1, freq_bands = self.freq_filt(bcic_data1)
        X_cv, y_cv, X_test, y_test = self.train_test_split(filtered1, y_labels1, train_share)
        X_cv, y_cv = self.data_cleaning(X_cv, y_cv)

        if subj2_filename is not None:
            bcic_data2 = LoadData.LoadBCIC(subj2_filename, self.data_path)
            filtered2, y_labels2, freq_bands2 = self.freq_filt(bcic_data2)
            _, _1, ood_X_test, ood_y_test = self.train_test_split(filtered2, y_labels2)
            X_test, y_test = self.data_cleaning(X_test, y_test)
            ood_X_test, ood_y_test = self.data_cleaning(ood_X_test, ood_y_test)
        elif subj2_filename is None:
            ood_X_test = subj2_filename
            ood_y_test = subj2_filename

        csp = CSP(n_components=self.m_filters, log=True)
        X_cv, X_test, ood_X_test = self.space_filt(csp, X_cv, X_test, y_cv, freq_bands, ood_X_test)

        feature_extractor = sklearn.feature_selection.SelectKBest(mutual_info_classif, k=n_best_features)
        X_cv = self.extract_features(feature_extractor, X_cv, y_cv)
        X_test = self.extract_features(feature_extractor, X_test, train_flag=False)

        if ood_X_test is not None:
            ood_X_test = self.extract_features(feature_extractor, ood_X_test, train_flag=False)

        return X_cv, y_cv, X_test, y_test, ood_X_test, ood_y_test

    def freq_filt(self, data):
        eeg_data = data.get_epochs()
        fbank = FilterBank(eeg_data.get('fs'))
        fbank_coeff = fbank.get_filter_coeff()
        filtered_data = fbank.filter_data(eeg_data.get('x_data'), self.window_details)
        y_labels = eeg_data.get('y_labels')
        frequency_bands = filtered_data.shape[0]
        return filtered_data, y_labels, frequency_bands

    def space_filt(self, csp, X_cv, X_test, y_cv, frequency_bands=9, ood_X_test=None):
        CSPs = []
        new_X_cv = []
        new_ood_X_test = []
        new_X_test = []

        for i in range(frequency_bands):

            tmp_X_cv = X_cv[i, :, :, :]
            tmp_X_test = X_test[i, :, :, :]
            X_cv_filtered = csp.fit_transform(tmp_X_cv, y_cv)
            X_test_filtered = csp.transform(tmp_X_test)
            if ood_X_test is not None:
                tmp_ood_X_test = ood_X_test[i, :, :, :]
                ood_X_test_filtered = csp.transform(tmp_ood_X_test)
            CSPs.append(csp)

            if i == 0:
                new_X_cv = X_cv_filtered
                if ood_X_test is not None:
                    new_ood_X_test = ood_X_test_filtered
                new_X_test = X_test_filtered
            else:
                new_X_cv = np.concatenate((new_X_cv, X_cv_filtered), axis=1)
                if ood_X_test is not None:
                    new_ood_X_test = np.concatenate((new_ood_X_test, ood_X_test_filtered), axis=1)
                new_X_test = np.concatenate((new_X_test, X_test_filtered), axis=1)
        if ood_X_test is None: # was is not None
            return new_X_cv, new_X_test, None
        else:
            return new_X_cv, new_X_test, new_ood_X_test

    def train_test_split(self, X, y, threshold=0.8):

        # TODO random_state?

        train_indices = []
        test_indices = []
        for i in range(len(y)):
            flag = np.random.uniform(0, 1, 1)
            if flag < threshold:
                train_indices.append(i)
            else:
                test_indices.append(i)
        X_cv, X_test = self.split_xdata(X, train_indices, test_indices)
        y_cv, y_test = self.split_ydata(y, train_indices, test_indices)
        return X_cv, y_cv, X_test, y_test

    def data_cleaning(self, X, y, domain=[0, 1]):
        y_local = 0
        y_range = len(y)
        while y_local < y_range:
            if y[y_local] in domain:
                y_local += 1
            else:
                y = np.delete(y, y_local)
                X = np.delete(X, y_local, axis=1)
                y_range -= 1
        return X, y

    def extract_features(self, feature_extractor, X, y=None, train_flag=True):
        if train_flag:
            feature_extractor.fit(X, y)

        X_extracted = feature_extractor.transform(X)
        return X_extracted

    def create_logger(self, subj2_filename, dir, filename=None): # TODO deprecate date as logger name, get filename as argument
        now = datetime.now()
        now_str = now.strftime("%Y.%m.%d_%H-%M-%S")
        # TODO if filename is not none set it as filename+datetime, else datetime
        if filename is not None:
            directory = filename + '_' + now_str
        else:
            directory = now_str
        path = os.path.join(dir, directory)
        os.mkdir(path)
        logs_path = os.path.join(path, 'logs.txt')
        logger = open(logs_path, mode='w')
        logger.write(now_str)
        logger.write('\n')
        if subj2_filename is None:
            logger.write('1 subject experiment\n')
        else:
            logger.write('2 subject experiment\n')
        buffer = str('Subject: '+str(self.file_to_load)+'\n')
        logger.write(buffer)
        if subj2_filename is not None:
            buffer = str('Subject2: '+str(subj2_filename)+'\n')
            logger.write(buffer)
        return logger, path

    def classifiers_train(self, X_cv, y_cv, bayesian_classifier, det_classifier=None):

        epochs = []
        x_valid_accuracy = []
        bay_train_losses = []
        bay_valid_losses = []
        bay_valid_accuracy = []
        if det_classifier is not None:
            det_train_losses = []
            det_valid_losses = []
            det_valid_accuracy = []

        for k in range(self.ntimes):
            bay_running_loss = 0.0
            bay_run_val_loss = 0.0
            if det_classifier is not None:
                det_running_loss = 0.0
                det_run_val_loss = 0.0

            flag = 0
            if (self.ntimes - k) <= 1:
                flag = 1

            train_indices, valid_indices = self.cross_validate_sequential_split(y_cv)

            for i in range(self.kfold):
                train_idx = train_indices.get(i)
                valid_idx = valid_indices.get(i)
                print(f'Times {str(k)}, Fold {str(i)}\n')
                y_train, y_valid = self.split_ydata(y_cv, train_idx, valid_idx)
                x_train_fb = X_cv[train_idx]
                x_valid_fb = X_cv[valid_idx]

                bay_train_loss = LCM.fit(bayesian_classifier, x_train_fb, y_train, LCM.CrossEntropyWithLogPrior, flag)
                bay_valid_loss, bay_valid_preds = LCM.validate(bayesian_classifier, x_valid_fb, y_valid, LCM.CrossEntropyWithLogPrior)
                bay_val_acc = count_accuracy(bay_valid_preds, y_valid)
                bay_valid_accuracy.append(bay_val_acc)
                bay_running_loss += bay_train_loss
                bay_run_val_loss += bay_valid_loss

                if det_classifier is not None:
                    det_train_loss = DC.fit(det_classifier, x_train_fb, y_train, DC.XEL)
                    det_valid_loss, det_valid_preds = DC.validate(det_classifier, x_valid_fb, y_valid, DC.XEL)
                    det_val_acc = count_accuracy(det_valid_preds, y_valid)
                    det_valid_accuracy.append(det_val_acc)
                    det_running_loss += det_train_loss
                    det_run_val_loss += det_valid_loss

                x_valid_accuracy.append(k + i/self.kfold)

            bay_train_losses.append(bay_running_loss)
            bay_valid_losses.append(bay_run_val_loss)
            epochs.append(k)

            if det_classifier is not None:
                det_train_losses.append(det_running_loss)
                det_valid_losses.append(det_run_val_loss)

        train_curves = plt.figure()
        loss_curves = train_curves.add_subplot(121, title='Loss Curves', xlabel='epoch', ylabel='Loss')
        loss_curves.plot(epochs, bay_train_losses, 'r--', label='lap_mod_train')
        loss_curves.plot(epochs, bay_valid_losses, 'k--', label='lap_mod_valid')
        if det_classifier is not None:
            loss_curves.plot(epochs, det_train_losses, 'g-', label='det_train')
            loss_curves.plot(epochs, det_valid_losses, 'm-', label='det_valid')
        loss_curves.legend()

        valid_acc = train_curves.add_subplot(122, title='Validation accuracy', xlabel='epoch', ylabel='Accuracy')
        valid_acc.plot(x_valid_accuracy, bay_valid_accuracy, 'm-', label='laplace_mod')
        if det_classifier is not None:
            valid_acc.plot(x_valid_accuracy, det_valid_accuracy, 'g-', label='deterministic')
        valid_acc.legend()

        return train_curves

    def det_classifier_train(self, X_cv, y_cv, det_classifier):
        epochs = []
        x_valid_accuracy = []
        det_train_losses = []
        det_valid_losses = []
        det_valid_accuracy = []
        for k in range(self.ntimes):
            det_running_loss = 0.0
            det_run_val_loss = 0.0
            train_indices, valid_indices = self.cross_validate_sequential_split(y_cv)

            for i in range(self.kfold):
                train_idx = train_indices.get(i)
                valid_idx = valid_indices.get(i)
                print(f'Times {str(k)}, Fold {str(i)}\n')
                y_train, y_valid = self.split_ydata(y_cv, train_idx, valid_idx)
                x_train_fb = X_cv[train_idx]
                x_valid_fb = X_cv[valid_idx]
                det_train_loss = DC.fit(det_classifier, x_train_fb, y_train, DC.XEL)
                det_valid_loss, det_valid_preds = DC.validate(det_classifier, x_valid_fb, y_valid, DC.XEL)
                det_val_acc = count_accuracy(det_valid_preds, y_valid)
                det_valid_accuracy.append(det_val_acc)
                det_running_loss += det_train_loss
                det_run_val_loss += det_valid_loss
                x_valid_accuracy.append(k + i/self.kfold)

            epochs.append(k)
            det_train_losses.append(det_running_loss)
            det_valid_losses.append(det_run_val_loss)

        train_curves = plt.figure()
        loss_curves = train_curves.add_subplot(121, title='Loss Curves', xlabel='epoch', ylabel='Loss')
        loss_curves.plot(epochs, det_train_losses, 'g-', label='det_train')
        loss_curves.plot(epochs, det_valid_losses, 'm-', label='det_valid')
        loss_curves.legend()
        valid_acc = train_curves.add_subplot(122, title='Validation accuracy', xlabel='epoch', ylabel='Accuracy')
        valid_acc.plot(x_valid_accuracy, det_valid_accuracy, 'g-', label='deterministic')
        valid_acc.legend()

        return train_curves

    def cross_validate_Ntimes_Kfold(self, y_labels, ifold=0):
        from sklearn.model_selection import StratifiedKFold
        train_indices = {}
        test_indices = {}
        random_seed = ifold
        skf_model = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=random_seed)
        i = 0
        for train_idx, test_idx in skf_model.split(np.zeros(len(y_labels)), y_labels):
            train_indices.update({i: train_idx})
            test_indices.update({i: test_idx})
            i += 1
        return train_indices, test_indices

    def cross_validate_sequential_split(self, y_labels):
        from sklearn.model_selection import StratifiedKFold
        train_indices = {}
        test_indices = {}
        skf_model = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=42)
        i = 0
        for train_idx, test_idx in skf_model.split(np.zeros(len(y_labels)), y_labels):
            train_indices.update({i: train_idx})
            test_indices.update({i: test_idx})
            i += 1
        return train_indices, test_indices

    def cross_validate_half_split(self, y_labels):
        import math
        unique_classes = np.unique(y_labels)
        all_labels = np.arange(len(y_labels))
        train_idx = np.array([])
        test_idx = np.array([])
        for cls in unique_classes:
            cls_indx = all_labels[np.where(y_labels == cls)]
            if len(train_idx) == 0:
                train_idx = cls_indx[:math.ceil(len(cls_indx)/2)]
                test_idx = cls_indx[math.ceil(len(cls_indx)/2):]
            else:
                train_idx = np.append(train_idx, cls_indx[:math.ceil(len(cls_indx)/2)])
                test_idx = np.append(test_idx, cls_indx[math.ceil(len(cls_indx)/2):])

        train_indices = {0: train_idx}
        test_indices = {0: test_idx}

        return train_indices, test_indices

    def split_xdata(self, eeg_data, train_idx, test_idx):
        x_train_fb = np.copy(eeg_data[:, train_idx, :, :])
        x_test_fb = np.copy(eeg_data[:, test_idx, :, :])
        return x_train_fb, x_test_fb

    def split_ydata(self, y_true, train_idx, test_idx):
        y_train = np.copy(y_true[train_idx])
        y_test = np.copy(y_true[test_idx])

        return y_train, y_test

    def get_multi_class_label(self, y_predicted, cls_interest=0):
        y_predict_multi = np.zeros((y_predicted.shape[0]))
        for i in range(y_predicted.shape[0]):
            y_lab = y_predicted[i, :]
            lab_pos = np.where(y_lab == cls_interest)[0]
            if len(lab_pos) == 1:
                y_predict_multi[i] = lab_pos
            elif len(lab_pos > 1):
                y_predict_multi[i] = lab_pos[0]
        return y_predict_multi

    def get_multi_class_regressed(self, y_predicted):
        y_predict_multi = np.asarray([np.argmin(y_predicted[i, :]) for i in range(y_predicted.shape[0])])
        return y_predict_multi

    def get_roc_data(self, ind_variances, ood_variances, t_start, t_stop, t_step=0.005, logger=None):

        variances = []
        for var in ind_variances:
            variances.append(var)
        for var in ood_variances:
            variances.append(var)
        labels = [1] * len(ind_variances)
        ood_labels = [0] * len(ood_variances)
        for label in ood_labels:
            labels.append(label)
        xs = []
        ys = []
        t = t_start
        while t < t_stop:
            x, y = roc_dots_counter(variances, labels, t)
            xs.append(x)
            ys.append(y)
            t += t_step
        a = auc(xs, ys)

        if logger is not None:
            logger.write('ROC datapoints'+'\n')
            for i, j in zip(variances, labels):
                logger.write('variance: '+str(i)+' label: '+str(j)+'\n')

        return xs, ys, a


class NeuroNetEngine:
    def __init__(self, network='EEGNet', subject_id=3, prior_variance=1.0, likelihood='Categorical', prior='Gaussian'):
        pass # TODO initialize params

    def NeuroNetExperiment(self):
        pass # TODO copy from MLEngine

    def DetNeuroNetExperiment(self):
        pass # TODO copy from MLEngine


class FilterBank:
    def __init__(self, fs):
        self.fs = fs
        self.f_trans = 2
        self.f_pass = np.arange(4, 40, 4)  # (lower_bound, upper_bound, step)
        self.f_width = 4
        self.gpass = 3
        self.gstop = 30
        self.filter_coeff={}

    def get_filter_coeff(self):
        Nyquist_freq = self.fs/2

        for i, f_low_pass in enumerate(self.f_pass):
            f_pass = np.asarray([f_low_pass, f_low_pass+self.f_width])
            f_stop = np.asarray([f_pass[0]-self.f_trans, f_pass[1]+self.f_trans])
            wp = f_pass/Nyquist_freq
            ws = f_stop/Nyquist_freq
            order, wn = cheb2ord(wp, ws, self.gpass, self.gstop)
            b, a = signal.cheby2(order, self.gstop, ws, btype='bandpass')
            self.filter_coeff.update({i: {'b': b, 'a': a}})

        return self.filter_coeff

    def filter_data(self, eeg_data, window_details={}):
        n_trials, n_channels, n_samples = eeg_data.shape
        if window_details:
            n_samples = int(self.fs*(window_details.get('tmax')-window_details.get('tmin')))+1
        filtered_data=np.zeros((len(self.filter_coeff), n_trials, n_channels, n_samples))
        for i, fb in self.filter_coeff.items():
            b = fb.get('b')
            a = fb.get('a')
            eeg_data_filtered = np.asarray([signal.lfilter(b, a, eeg_data[j, :, :]) for j in range(n_trials)])
            if window_details:
                eeg_data_filtered = eeg_data_filtered[:, :, int((4.5+window_details.get('tmin'))*self.fs):int((4.5 + window_details.get('tmax'))*self.fs)+1]
            filtered_data[i, :, :, :] = eeg_data_filtered

        return filtered_data


def count_in_domain_accuracy(predictions, targets, domain=[0, 1]): # function to calculate accuracy on only in-domain datapoints
    in_domain_targets = []
    in_domain_preds = []
    right_preds = 0
    for i in range(len(targets)):
        if targets[i] in domain:
            in_domain_targets.append(targets[i])
            in_domain_preds.append(predictions[i])

    for j in range(len(in_domain_targets)):
        if in_domain_targets[j] == in_domain_preds[j]:
            right_preds += 1

    result = right_preds/len(in_domain_targets)
    return result


def count_accuracy(predictions, targets):
    count = 0
    for i in range(len(targets)):
        if predictions[i] == targets[i]:
            count += 1
    accuracy = count / len(targets)
    return accuracy


def domain_split(probs, targs, domain=[0,1]):
    probs_ind = []
    probs_ood = []
    for i in range(len(targs)):
        if targs[i] in domain:
            probs_ind.append(probs[i])
        else:
            probs_ood.append(probs[i])
    return probs_ind, probs_ood


def entropy_counter(probs_ind, probs_ood):
    entropies = []
    ood_ratio = []
    dist_factor = 1
    entropies.append(dist_factor * entropy(probs_ind))
    ood_ratio.append(0)
    for i in range(len(probs_ood)):
        probs_ind.append(probs_ood[i])
        ood_ratio.append((i+1)/len(probs_ind))
        entropies.append(entropy(probs_ind))
    return entropies, ood_ratio


def roc_dots_counter(vars, labels, threshold, flag='forward'):
    assert len(labels) == len(vars)
    preds = []
    tp = fp = tn = fn = 0
    for var in vars:
        if flag == 'forward':
            if var > threshold:
                preds.append(0)
            else:
                preds.append(1)
        elif flag == 'reverse':
            if var < threshold:
                preds.append(0)
            else:
                preds.append(1)
    for i in range(len(labels)):
        if labels[i] == 0 and preds[i] == 0:
            tn += 1
        elif labels[i] == 1 and preds[i] == 1:
            tp += 1
        elif labels[i] == 0 and preds[i] == 1:
            fp += 1
        elif labels[i] == 1 and preds[i] == 0:
            fn += 1
    x = fp/(tn+fp)
    y = tp/(tp+fn)
    return x, y


def distinguish_variances(variances, predictions, targets):
    false_variances = []
    true_variances = []
    for i in range(len(variances)):
        if predictions[i] == targets[i]:
            true_variances.append(variances[i])
        else:
            false_variances.append(variances[i])
    return true_variances, false_variances


def braindecode_preprocessing(subj_id):
    subject_id = subj_id
    dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])
    from braindecode.preprocessing import (
        exponential_moving_standardize, preprocess, Preprocessor, scale)

    low_cut_hz = 4.  # low cut frequency for filtering
    high_cut_hz = 38.  # high cut frequency for filtering
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000

    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
        Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convert from V to uV
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
        Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                     factor_new=factor_new, init_block_size=init_block_size)
    ]

    preprocess(dataset, preprocessors)

    from braindecode.preprocessing import create_windows_from_events

    trial_start_offset_seconds = -0.5
    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    # Create windows using braindecode function for this. It needs parameters to
    # define how trials should be used.
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=True,
    )
    # getting channels' coordinates for priors
    coordinates = []
    all_coordinates = dataset.datasets[0].raw.info['chs']
    for x in all_coordinates:
        channel_loc = x['loc'][0:3]
        coordinates.append(channel_loc)
    splitted = windows_dataset.split('session')
    train_set = splitted['session_T']
    test_set = splitted['session_E']
    return train_set, test_set, coordinates


def eeg_data_parser(dataset):
    X_data = []
    y_data = []
    windows_data = []
    for i in range(len(dataset)):
        X_data.append(dataset[i][0])
        y_data.append(dataset[i][1])
        windows_data.append(dataset[i][2])
    return X_data, y_data, windows_data


def clean_data(X, y, domain=[0,1]):
    y_range = len(y)
    y_local = 0
    clean_X = []
    clean_y = []
    for point, label in zip(X, y):
        if label in domain:
            clean_y.append(label)
            clean_X.append(point)

    return clean_X, clean_y


def swap_class_labels(y_train, y_test, label_1, label_2):
    # TODO swap labels FIXME
    swapped_y_train = []
    swapped_y_test = []
    for y in y_train:
        if y == label_1:
            swapped_y_train.append(label_2)
        elif y == label_2:
            swapped_y_train.append(label_1)
        else:
            swapped_y_train.append(y)

    for y in y_test:
        if y == label_1:
            swapped_y_test.append(label_2)
        elif y == label_2:
            swapped_y_test.append(label_1)
        else:
            swapped_y_test.append(y)

    return swapped_y_train, swapped_y_test
