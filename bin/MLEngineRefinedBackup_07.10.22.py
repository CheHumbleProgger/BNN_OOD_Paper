import numpy as np
import os
import torch
import scipy.signal as signal
from scipy.signal import cheb2ord
from scipy.stats import entropy
import bin.LoadData as LoadData
import bin.DeterministicClassifier as DC
import bin.Preprocess as Preprocess
import bin.LaplaceApproxClassif as LAC
import bin.LaplaceModified as LCM
import matplotlib.pyplot as plt
import sklearn.feature_selection
from sklearn.feature_selection import mutual_info_classif
from mne.decoding import CSP
from datetime import datetime
#from .SGHMC.SGHMCClassifier import SGHMCClassifier
import pickle
from sklearn.metrics import roc_auc_score, roc_curve, auc
import seaborn as sns

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
        self.m_filters = m_filters

    def experiment(self, subj2_filename=None):
        # TODO preprocess(+) -> train(+) -> predict labels(+) -> domain_split if needed(+) -> plot ind_ood_variances -> plot loss_delta if needed and other graphs
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

        log_path = r"C:\Users\user\DataspellProjects\BNNproject1\fbcsp_code\experiment_logs"

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
            xs, ys, a = self.get_roc_data(var_ind, var_ood, 0.02, 0.155, logger) #FIXME

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
            true_vars, false_vars = distinguish_variances(variances, bay_preds, y_test)
        else:
            total_variances = variances
            total_preds = bay_preds
            targets = list(y_test)
            for i in range(len(ood_variances)):
                total_variances.append(ood_variances[i])
                total_preds.append(ood_bay_preds[i])
                targets.append(ood_y_test[i])
            true_vars, false_vars = distinguish_variances(total_variances, total_preds, targets)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        fig.suptitle('Variances distribution among truly and falsely classified samples')
        sns.histplot(ax=axes[0], x=true_vars, bins=7, kde=True)
        sns.histplot(ax=axes[1], x=false_vars, bins=7, kde=True)
        figpath = os.path.join(graph_path, 'variance_histograms.jpg')
        fig.savefig(figpath)

    def SGHMC_experiment(self, subj1_filename, subj2_filename=None):
        n_best_features = 8
        sample_cache = 10
        weights_to_sample = 100
        X_cv, y_cv, X_test, y_test, ood_X_test, ood_y_test = self.preprocess(subj1_filename, subj2_filename, n_best_features)
        X_train, X_valid, y_train, y_valid = self.train_test_split(X_cv, y_cv, threshold=0.9)
        sghmc_classifier = SGHMC.SGHMCClassifier.SGHMC(sample_cache=sample_cache, n_features_selected=n_best_features)
        det_classifier = DC.DeterministicClassifier(n_features_selected=n_best_features)
        # TODO X split on train/valid (+) -> train and validate SGHMC classifier (+)

        sghmc_classifier.fit(sghmc_classifier, X_train, y_train, SGHMC.CrossEntropyWithLogPrior, burn_in_steps=3000) # FIXME

        sghmc_classifier.validate(sghmc_classifier, X_valid, y_valid, SGHMC.CrossEntropyWithLogPrior) # FIXME

        for i in range(weights_to_sample//sample_cache - 1):

            sghmc_classifier.fit(sghmc_classifier, X_train, y_train, SGHMC.CrossEntropyWithLogPrior, burn_in_steps=0) #FIXME
            sghmc_classifier.validate(sghmc_classifier, X_valid, y_valid, SGHMC.CrossEntropyWithLogPrior) #FIXME

        # TODO write CV of deterministic classifier

        # TODO predictions -> graphs

    def preprocess(self, subj1_filename, subj2_filename=None, n_best_features=8):

        bcic_data1 = LoadData.LoadBCIC(subj1_filename, self.data_path)
        filtered1, y_labels1, freq_bands = self.freq_filt(bcic_data1)
        X_cv, y_cv, X_test, y_test = self.train_test_split(filtered1, y_labels1)
        X_cv, y_cv = self.data_cleaning(X_cv, y_cv)

        if subj2_filename is not None:
            bcic_data2 = LoadData.LoadBCIC(subj2_filename, self.data_path)
            filtered2, y_labels2, freq_bands2 = self.freq_filt(bcic_data2)
            ood_X_test, ood_y_test, _, _1 = self.train_test_split(filtered2, y_labels2)
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

    def create_logger(self, subj2_filename, dir):
        now = datetime.now()
        now_str = now.strftime("%Y.%m.%d_%H-%M-%S")
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

        return 0 # FIXME

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

    def get_roc_data(self, ind_variances, ood_variances, t_start, t_stop, logger):

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
            x, y = roc_dots_counter(variances, labels, t) # FIXME
            xs.append(x)
            ys.append(y)
            t += 0.005
        a = auc(xs, ys)

        logger.write('ROC datapoints'+'\n')
        for i, j in zip(variances, labels):
            logger.write('variance: '+str(i)+' label: '+str(j)+'\n')

        return xs, ys, a


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
