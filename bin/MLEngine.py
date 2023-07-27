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
import joblib
import pickle
from sklearn.metrics import roc_auc_score, roc_curve, auc
import seaborn as sns


class MLEngine:
    def __init__(self,data_path='',file_to_load='',subject_id='',sessions=[1, 2],ntimes=1,kfold=2,m_filters=2,window_details={}):
        self.data_path = data_path
        self.subject_id=subject_id
        self.file_to_load = file_to_load
        self.sessions = sessions
        self.kfold = kfold
        self.ntimes=ntimes
        self.window_details = window_details
        self.m_filters = m_filters

    def experiment_1_subj(self):
        '''for BCIC Dataset'''
        bcic_data = LoadData.LoadBCIC(self.file_to_load, self.data_path)
        eeg_data = bcic_data.get_epochs()


        '''for KU dataset'''
        # ku_data = LoadData.LoadKU(self.subject_id,self.data_path)
        # eeg_data = ku_data.get_epochs(self.sessions)
        # preprocess = Preprocess.PreprocessKU()
        # eeg_data_selected_channels = preprocess.select_channels(eeg_data.get('x_data'),eeg_data.get('ch_names'))
        # eeg_data.update({'x_data':eeg_data_selected_channels})

        # Frequency filtering

        fbank = FilterBank(eeg_data.get('fs'))
        fbank_coeff = fbank.get_filter_coeff()
        filtered_data = fbank.filter_data(eeg_data.get('x_data'), self.window_details)
        y_labels = eeg_data.get('y_labels')

        # TODO pipeline: FreqFilt -> Split -> data exclusion -> CSP -> MIBIF -> classify -> save good model
        # NOTE!: mne.csp() is ONLY for BINARY classification (?)

        # exclude 2 and 3 classes from datasets, leave only 0 and 1 for ind and ood

        # Data splitting

        threshold = 0.8
        train_ind = []
        test_ind = []
        ood_train_ind = []
        ood_test_ind = []

        for i in range(len(y_labels)):
            flag = np.random.uniform(0,1,1)
            if flag < threshold:
                train_ind.append(i)
            else:
                test_ind.append(i)

        X_cv, X_test = self.split_xdata(filtered_data, train_ind, test_ind)
        y_cv, y_test = self.split_ydata(y_labels, train_ind, test_ind)


        # reform y to OOD and IND
        # Data exclusion from train/valid set

        y_local = 0
        y_range = len(y_cv)
        while y_local < y_range:
            if y_cv[y_local] == 2 or y_cv[y_local] == 3:
                y_cv = np.delete(y_cv, y_local)
                X_cv = np.delete(X_cv, y_local, axis=1)
                y_range = y_range - 1
            else:
                y_local = y_local + 1

        # CSP and concatenation

        csp = CSP(n_components=self.m_filters, log=True)
        CSPs = []
        new_X_cv = []
        new_X_test = []

        for i in range(filtered_data.shape[0]):

            tmp_X_cv = X_cv[i,:,:,:]
            tmp_X_test = X_test[i,:,:,:]
            X_cv_filtered = csp.fit_transform(tmp_X_cv, y_cv)
            X_test_filtered = csp.transform(tmp_X_test)
            CSPs.append(csp)

            if i == 0:
                new_X_cv = X_cv_filtered
                new_X_test = X_test_filtered
            else:
                new_X_cv = np.concatenate((new_X_cv, X_cv_filtered), axis=1)
                new_X_test = np.concatenate((new_X_test, X_test_filtered), axis=1)

        # Feature extraction

        n_best_features = 8
        feature_extractor = sklearn.feature_selection.SelectKBest(mutual_info_classif, k=n_best_features).fit(new_X_cv, y_cv)
        X_cv_extracted = feature_extractor.transform(new_X_cv)
        X_test_extracted = feature_extractor.transform(new_X_test)

        print('X_cv_extracted shape: ', X_cv_extracted.shape)
        print('X_test_extracted shape: ', X_test_extracted.shape)

        # Fitting classifier

        training_accuracy = []
        valid_accuracy = []
        det_valid_accuracy = []
        new_valid_accuracy = []
        x_valid_accuracy = []
        testing_accuracy = []
        valid_preds = []
        det_valid_preds = []
        new_valid_preds = []
        train_losses = []
        valid_losses = []
        det_train_losses = []
        det_valid_losses = []
        new_train_losses = []
        new_valid_losses = []
        epochs = []

        classifier = LAC.LaplaceApproximationClassifier(n_features_selected=n_best_features)
        new_classifier = LCM.LaplaceApproximationClassifier(n_features_selected=n_best_features)
        det_classifier = DC.DeterministicClassifier(n_features_selected=n_best_features)
        OI_classifier = LAC.LaplaceApproximationClassifier(n_features_selected=n_best_features)
        OI_det_classifier = DC.DeterministicClassifier(n_features_selected=n_best_features)
        # initialize classifiers for IND and OOD
        for k in range(self.ntimes):
            running_loss = 0.0
            running_val_loss = 0.0
            det_running_loss = 0.0
            det_running_val_loss = 0.0
            new_running_loss = 0.0
            new_running_val_loss = 0.0
            flag = 0
            if (self.ntimes - k) <= 1:
                flag = 1

            '''for N times x K fold CV'''
            # train_indices, test_indices = self.cross_validate_Ntimes_Kfold(y_labels,ifold=k)
            '''for K fold CV by sequential splitting'''
            train_indices, valid_indices = self.cross_validate_sequential_split(y_cv)
            #OI_train_indices, OI_valid_indices = self.cross_validate_sequential_split(y_OI_cv)
            # split for IND and OOD
            '''for one fold in half half split'''
            # train_indices, test_indices = self.cross_validate_half_split(y_labels)

            for i in range(self.kfold):
                # fit for OOD and IND
                train_idx = train_indices.get(i)
                valid_idx = valid_indices.get(i)
                #OI_train_idx = OI_train_indices.get(i)
                #OI_valid_idx = OI_valid_indices.get(i)
                print(f'Times {str(k)}, Fold {str(i)}\n')
                y_train, y_valid = self.split_ydata(y_cv, train_idx, valid_idx)
                #OI_y_train, OI_y_valid = self.split_ydata(y_OI_cv, OI_train_idx, OI_valid_idx)
                x_train_fb = X_cv_extracted[train_idx]
                x_valid_fb = X_cv_extracted[valid_idx]

                print('X_train shape: ', x_train_fb.shape)

                train_loss = LAC.fit(classifier, x_train_fb, y_train, LAC.CrossEntropyWithLogPrior, flag)
                valid_loss, valid_preds = LAC.validate(classifier, x_valid_fb, y_valid, LAC.CrossEntropyWithLogPrior)
                new_train_loss = LCM.fit(new_classifier, x_train_fb, y_train, LCM.CrossEntropyWithLogPrior, flag)
                new_valid_loss, new_valid_preds = LCM.validate(new_classifier, x_valid_fb, y_valid, LCM.CrossEntropyWithLogPrior)
                #OI_train_loss = LAC.fit(OI_classifier, OI_x_train_fb, OI_y_train, LAC.CrossEntropyWithLogPrior, flag)
                #OI_valid_loss, OI_valid_preds = LAC.validate(OI_classifier, OI_x_valid_fb, OI_y_valid, LAC.CrossEntropyWithLogPrior)

                det_train_loss = DC.fit(det_classifier, x_train_fb, y_train, DC.XEL)
                det_valid_loss, det_valid_preds = DC.validate(det_classifier, x_valid_fb, y_valid, DC.XEL)
                #OI_det_train_loss = DC.fit(OI_det_classifier, OI_x_train_fb, OI_y_train, DC.XEL)
                #OI_det_valid_loss, OI_det_valid_preds = DC.validate(OI_det_classifier, OI_x_valid_fb, OI_y_valid, DC.XEL)

                val_acc = count_accuracy(valid_preds, y_valid)
                valid_accuracy.append(val_acc)
                det_val_acc = count_accuracy(det_valid_preds, y_valid)
                det_valid_accuracy.append(det_val_acc)
                new_val_acc = count_accuracy(new_valid_preds, y_valid)
                new_valid_accuracy.append(new_val_acc)
                x_valid_accuracy.append(k + i/self.kfold)

                running_loss += train_loss
                running_val_loss += valid_loss

                det_running_loss += det_train_loss
                det_running_val_loss += det_valid_loss

                new_running_loss += new_train_loss
                new_running_val_loss += new_valid_loss

            train_losses.append(running_loss)
            valid_losses.append(running_val_loss)
            det_train_losses.append(det_running_loss)
            det_valid_losses.append(det_running_val_loss)
            new_train_losses.append(new_running_loss)
            new_valid_losses.append(new_running_val_loss)
            epochs.append(k)

        # file with logs initialization
        now = datetime.now()
        now_str = now.strftime("%Y.%m.%d_%H-%M-%S")
        parent_dir = r"C:\Users\user\DataspellProjects\BNNproject1\fbcsp_code\experiment_logs"
        directory = now_str
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
        logs_path = os.path.join(path, 'logs.txt')
        logs = open(logs_path, mode='w')

        logs.write(now_str)
        logs.write('\n')
        logs.write('1 subject experiment\n')
        buffer = str('Subject: '+str(self.file_to_load)+'\n')
        logs.write(buffer)

        # Test prediction

        #print("LAC parameters: ", classifier.weights.weight, classifier.weights.bias)
        print("DC parameters: ", det_classifier.weights.weight, det_classifier.weights.bias)
        buffer = str('DC parameters:'+'\n'+'weights: '+str(det_classifier.weights.weight)+'\n'+'bias: '+str(det_classifier.weights.bias)+'\n')
        logs.write(buffer)
        print("LCM parameters: ", new_classifier.theta.weight)
        buffer = str('LCM parameters: '+'\n'+'weights: '+str(new_classifier.theta.weight)+'\n'+'covar_matrix: '+str(new_classifier.posterior_sigma)+'\n')
        logs.write(buffer)
        print('LCM covar_matrix: ', new_classifier.posterior_sigma)

        # TODO make predictions on another subject


        print('last epoch average fold training accuracies:')
        last_avg = (new_valid_accuracy[-1] + new_valid_accuracy[-2] + new_valid_accuracy[-3]) / 3
        print('Laplace: ', last_avg)
        det_last_avg = (det_valid_accuracy[-1] + det_valid_accuracy[-2] + det_valid_accuracy[-3]) / 3
        print('Deterministic: ', det_last_avg)
        buffer = str('last epoch average fold training accuracies:' + '\n')
        logs.write(buffer)
        buffer = str('Laplace: ' + str(last_avg) + '\n' + 'Deterministic' + str(det_last_avg) + '\n')
        logs.write(buffer)
        #test_predictions = LAC.predict(classifier, X_test_extracted)
        #OI_test_predictions = LAC.predict(OI_classifier, X_test_extracted)
        #test_acc = np.sum(test_predictions == y_test, dtype=np.float) / len(y_test)
        #print('Laplace classifier Testing accuracy: ', test_acc)
        det_test_predictions, det_total_loss = DC.predict(det_classifier, X_test_extracted, y_test, DC.XEL)
        #OI_det_test_predictions = DC.predict(OI_det_classifier, X_test_extracted)
        new_test_predictions, total_loss, variances = LCM.predict(new_classifier, X_test_extracted, y_test, LCM.CrossEntropyWithLogPrior, n=500, verbose=True)
        test_acc = np.sum(new_test_predictions == y_test, dtype=np.float) / len(y_test)
        print('Laplace modified classifier Testing accuracy: ', test_acc)
        buffer = str('Laplace modified classifier Testing accuracy: '+str(test_acc)+'\n')
        logs.write(buffer)
        test_acc = np.sum(det_test_predictions == y_test, dtype=np.float) / len(y_test)
        print('Deterministic classifier Testing accuracy: ', test_acc)
        buffer = str('Deterministic classifier Testing accuracy: '+str(test_acc)+'\n')
        logs.write(buffer)
        #test_probs = LAC.predict_proba(classifier, X_test_extracted)
        #det_test_probs = DC.predict_proba(det_classifier, X_test_extracted)
        # new_test_probs = LCM.predict_proba(new_classifier, X_test_extracted)

        # counting delta loss between det and lap versus num of samples
        sample_nums = [1, 5, 10, 25, 50, 75, 100, 150, 200, 250, 500]
        loss_delta = []
        for num in sample_nums:
            new_preds, total_loss, v = LCM.predict(new_classifier, X_test_extracted, y_test, LCM.CrossEntropyWithLogPrior, n=num)
            loss_delta.append(total_loss-det_total_loss)

        # IND and OOD mean variances comparison
        var_ind, var_ood = domain_split(variances, y_test)
        preds_ind, preds_ood = domain_split(new_test_predictions, y_test)
        print('Mean variance for IND samples: ', np.average(var_ind),'\n', 'Mean variance for OOD samples: ', np.average(var_ood))
        buffer = str('Mean variance for IND samples: ' + str(np.average(var_ind)) + '\n'+ 'Mean variance for OOD samples: ' + str(np.average(var_ood))+'\n')
        logs.write(buffer)

        sd = []
        for g in variances:
            sd.append(np.sqrt(g))
        sd_ind, sd_ood = domain_split(sd, y_test)


        #det_ind, det_ood = domain_split(det_test_probs, y_test)
        #ind, ood = domain_split(test_probs, y_test)
        #det_entropies, det_ratio = entropy_counter(det_ind, det_ood)
        #entropies, ratio = entropy_counter(ind, ood)

        # In-domain Test prediction

        #test_in_domain = count_in_domain_accuracy(test_predictions, y_test)
        #print('Laplace classifier In-Domain Testing accuracy: ',test_in_domain)
        det_test_in_domain = count_in_domain_accuracy(det_test_predictions, y_test)
        print('Deterministic classifier In-Domain Testing accuracy: ', det_test_in_domain)
        buffer = str('Deterministic classifier In-Domain Testing accuracy: '+str(det_test_in_domain)+'\n')
        logs.write(buffer)
        new_test_in_domain = count_in_domain_accuracy(new_test_predictions, y_test)
        print('Laplace modified classifier In-Domain Testing accuracy: ', new_test_in_domain)
        buffer = str('Laplace modified classifier In-Domain Testing accuracy: '+str(new_test_in_domain)+'\n')
        logs.write(buffer)

        variances = []
        for var in var_ind:
            variances.append(var)
        for var in var_ood:
            variances.append(var)
        labels = [1] * len(var_ind)
        ood_labels = [0] * len(var_ood)
        for label in ood_labels:
            labels.append(label)
        xs = []
        ys = []
        t = 0.02 # last 0.045
        while t < 0.155: # last 0.095
            x, y = roc_dots_counter(variances, labels, t)
            xs.append(x)
            ys.append(y)
            t += 0.005

        a = auc(xs, ys)

        logs.write('ROC datapoints'+'\n')
        for i, j in zip(variances, labels):
            logs.write('variance: '+str(i)+' label: '+str(j)+'\n')
        logs.close()

        # Model saving (DEPRECATED)

        #if test_in_domain > 0.6 and det_test_in_domain > 0.6: # TODO make dir to put CSPs into, if exists, put there
        #   joblib.dump(feature_extractor, 'feature_extractor.pkl')
        #  for i in range(len(CSPs)):
        #     filename = r'C:\Users\user\DataspellProjects\BNNproject1\CSPs\CSP_' + str(i) + '.pkl' #FIXME  maybe delete all files in CSPs before replacing?
        #    joblib.dump(CSPs[i], filename)
        #torch.save(classifier.state_dict(), 'saved_model.pt')
        #torch.save(det_classifier.state_dict(), 'saved_det_model.pt')
        #print('Models saved.')


        # Graph plotting

        #print(det_valid_accuracy)

        fig = plt.figure()
        loss_curves = fig.add_subplot(121, title='Loss Curves', xlabel='epoch', ylabel='Loss')
        #loss_curves.plot(epochs, train_losses, 'b-', label='train')
        #loss_curves.plot(epochs, valid_losses, 'r-', label='valid')
        loss_curves.plot(epochs, det_train_losses, 'g-', label='det_train')
        loss_curves.plot(epochs, det_valid_losses, 'm-', label='det_valid')
        loss_curves.plot(epochs, new_train_losses, 'r--', label='lap_mod_train')
        loss_curves.plot(epochs, new_valid_losses, 'k--', label='lap_mod_valid')
        loss_curves.legend()

        valid_acc = fig.add_subplot(122, title='Validation accuracy', xlabel='epoch', ylabel='Accuracy')
        #valid_acc.plot(x_valid_accuracy, valid_accuracy, 'b-', label='laplace')
        valid_acc.plot(x_valid_accuracy, det_valid_accuracy, 'g-', label='deterministic')
        valid_acc.plot(x_valid_accuracy, new_valid_accuracy, 'm-', label='laplace_mod')
        valid_acc.legend()
        figpath = os.path.join(path, 'graphs.jpg')
        fig.savefig(figpath)



        # TODO plot ROC-curves for in-domain and out-of-domain data points

        # ROC-AUC metrics implementation (DEPRECATED)

        #lap_probs = LAC.predict_proba(OI_classifier, X_test_extracted)
        #det_probs = DC.predict_proba(OI_det_classifier, X_test_extracted)
        # print(lap_probs)
        # print(y_test)
        #lap_auc = roc_auc_score(y_OI_test, lap_probs, multi_class='ovr')
        #det_auc = roc_auc_score(y_OI_test, det_probs, multi_class='ovr')
        #lap_fpr, lap_tpr, lap_threshold = roc_curve(y_OI_test, lap_probs)
        #det_fpr, det_tpr, det_threshold = roc_curve(y_OI_test, det_probs)
        #lap_roc_auc = auc(lap_fpr, lap_tpr)
        #det_roc_auc = auc(det_fpr, det_tpr)
        #print('Laplace ROC-AUC: ', lap_auc)
        #print('Deterministic ROC-AUC: ', det_auc)


        #fig2 = plt.figure()
        #ax0 = fig2.add_subplot(121, title='Laplace_ROC')
        #ax0.plot(lap_fpr, lap_tpr, 'b-', label='Laplace_ROC, area = %0.3f' % lap_roc_auc)
        #ax0.plot([0, 1], [0, 1], 'c--')
        #ax1 = fig2.add_subplot(122, title='Deterministic_ROC')
        #ax1.plot(det_fpr, det_tpr, 'b-', label='Deterministic_ROC, area = %0.3f' % det_roc_auc)
        #ax1.plot([0, 1], [0, 1], 'c--')
        #fig2.savefig('ROC-AUC_Curves.jpg')

        fig3 = plt.figure()
        loss_discrepancy = fig3.add_subplot(111, title='loss delta', xlabel='n', ylabel='Loss_delta')
        loss_discrepancy.plot(sample_nums, loss_delta, 'b-', label='loss_delta vs n')
        loss_discrepancy.plot([0, 500], [0, 0], 'c--')
        figpath = os.path.join(path, 'loss_delta_vs_n.jpg')
        fig3.savefig(figpath)

        fig4 = plt.figure()
        ind_variances = fig4.add_subplot(121,title='ind_preds', xlabel='sample', ylabel='pred+-sd')
        ind_variances.errorbar([n for n in range(len(preds_ind))], preds_ind, yerr=sd_ind, fmt='o')
        ind_variances.set_xlim([0, 20])
        ind_variances.set_ylim([-2.0, 2.0])
        ood_variances = fig4.add_subplot(122,title='ood_preds', xlabel='sample', ylabel='pred+-sd')
        ood_variances.errorbar([n for n in range(len(preds_ood))], preds_ood, yerr=sd_ood, fmt='o')
        ood_variances.set_xlim([0, 20])
        ood_variances.set_ylim([-2.0, 2.0])
        figpath = os.path.join(path, 'ind_ood_variances.jpg')
        fig4.savefig(figpath)

        fig5 = plt.figure()
        roc = fig5.add_subplot(111, title='variance criteria roc')
        roc.plot(xs,ys, 'b-', label='roc, area = %0.3f' % a)
        roc.plot([0,1],[0,1], 'c--')
        roc.legend()
        figpath = os.path.join(path, 'variance_roc.jpg')
        fig5.savefig(figpath)

        true_vars, false_vars = distinguish_variances(variances, new_test_predictions, y_test)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        fig.suptitle('Variances distribution among truly and falsely classified samples')
        sns.histplot(ax=axes[0], x=true_vars, bins=7, kde=True)
        sns.histplot(ax=axes[1], x=false_vars, bins=7, kde=True)
        figpath = os.path.join(path, 'variance_histograms.jpg')
        fig.savefig(figpath)

    def experiment_2_subj(self, second_subj_file='A07T.gdf'):
        '''for BCIC Dataset'''
        bcic_data = LoadData.LoadBCIC(self.file_to_load, self.data_path)
        eeg_data = bcic_data.get_epochs()
        ood_bcic_data = LoadData.LoadBCIC(second_subj_file, self.data_path)
        ood_eeg_data = ood_bcic_data.get_epochs()

        # TODO duplicate  preprocessing pipeline for another subj (change names for OI variables)
        # TODO leave only labels 0 and 1 for both subj

        '''for KU dataset'''
        # ku_data = LoadData.LoadKU(self.subject_id,self.data_path)
        # eeg_data = ku_data.get_epochs(self.sessions)
        # preprocess = Preprocess.PreprocessKU()
        # eeg_data_selected_channels = preprocess.select_channels(eeg_data.get('x_data'),eeg_data.get('ch_names'))
        # eeg_data.update({'x_data':eeg_data_selected_channels})

        # Frequency filtering

        fbank = FilterBank(eeg_data.get('fs'))
        fbank_coeff = fbank.get_filter_coeff()
        filtered_data = fbank.filter_data(eeg_data.get('x_data'), self.window_details)
        y_labels = eeg_data.get('y_labels')

        ood_fbank = FilterBank(ood_eeg_data.get('fs'))
        ood_fbank_coeff = ood_fbank.get_filter_coeff()
        ood_filtered_data = fbank.filter_data(ood_eeg_data.get('x_data'), self.window_details)
        ood_y_labels = ood_eeg_data.get('y_labels')

        # TODO pipeline: FreqFilt -> Split -> data exclusion -> CSP -> MIBIF -> classify -> save good model
        # NOTE!: mne.csp() is ONLY for BINARY classification (?)

        # exclude 2 and 3 classes from datasets, leave only 0 and 1 for ind and ood

        y_local = 0
        y_range = len(y_labels)
        while y_local < y_range:
            if y_labels[y_local] == 2 or y_labels[y_local] == 3:
                y_labels = np.delete(y_labels, y_local)
                filtered_data = np.delete(filtered_data, y_local, axis=1)
                y_range = y_range - 1
            else:
                y_local = y_local + 1

        y_local = 0
        y_range = len(ood_y_labels)
        while y_local < y_range:
            if ood_y_labels[y_local] == 2 or ood_y_labels[y_local] == 3:
                ood_y_labels = np.delete(ood_y_labels, y_local)
                ood_filtered_data = np.delete(ood_filtered_data, y_local, axis=1)
                y_range = y_range - 1
            else:
                y_local = y_local + 1

        # Data splitting

        threshold = 0.8
        train_ind = []
        test_ind = []
        ood_train_ind = []
        ood_test_ind = []

        for i in range(len(y_labels)):
            flag = np.random.uniform(0,1,1)
            if flag < threshold:
                train_ind.append(i)
            else:
                test_ind.append(i)

        for i in range(len(ood_y_labels)):
            flag = np.random.uniform(0,1,1)
            if flag < threshold:
                ood_train_ind.append(i)
            else:
                ood_test_ind.append(i)


        X_cv, X_test = self.split_xdata(filtered_data, train_ind, test_ind)
        y_cv, y_test = self.split_ydata(y_labels, train_ind, test_ind)
        ood_X_cv, ood_X_test = self.split_xdata(ood_filtered_data, ood_train_ind, ood_test_ind)
        ood_y_cv, ood_y_test = self.split_ydata(ood_y_labels, ood_train_ind, ood_test_ind)

        # reform y to OOD and IND
        # Data exclusion from train/valid set

        #y_local = 0
        #y_range = len(y_cv)
        #while y_local < y_range:
        #    if y_cv[y_local] == 2 or y_cv[y_local] == 3:
        #        y_cv = np.delete(y_cv, y_local)
        #        X_cv = np.delete(X_cv, y_local, axis=1)
        #        y_range = y_range - 1
        #    else:
        #        y_local = y_local + 1

        # CSP and concatenation

        csp = CSP(n_components=self.m_filters, log=True)
        CSPs = []
        new_X_cv = []
        new_ood_X_test = []
        new_X_test = []

        for i in range(filtered_data.shape[0]):

            tmp_X_cv = X_cv[i,:,:,:]
            tmp_ood_X_test = ood_X_test[i,:,:,:]
            tmp_X_test = X_test[i,:,:,:]
            X_cv_filtered = csp.fit_transform(tmp_X_cv, y_cv)
            ood_X_test_filtered = csp.transform(tmp_ood_X_test)
            X_test_filtered = csp.transform(tmp_X_test)
            CSPs.append(csp)

            if i == 0:
                new_X_cv = X_cv_filtered
                new_ood_X_test = ood_X_test_filtered
                new_X_test = X_test_filtered
            else:
                new_X_cv = np.concatenate((new_X_cv, X_cv_filtered), axis=1)
                new_ood_X_test = np.concatenate((new_ood_X_test, ood_X_test_filtered), axis=1)
                new_X_test = np.concatenate((new_X_test, X_test_filtered), axis=1)

        # Feature extraction

        n_best_features = 8
        feature_extractor = sklearn.feature_selection.SelectKBest(mutual_info_classif, k=n_best_features).fit(new_X_cv, y_cv)
        X_cv_extracted = feature_extractor.transform(new_X_cv)
        ood_X_test_extracted = feature_extractor.transform(new_ood_X_test)
        X_test_extracted = feature_extractor.transform(new_X_test)

        print('X_cv_extracted shape: ', X_cv_extracted.shape)
        print('X_test_extracted shape: ', X_test_extracted.shape)

        # Fitting classifier

        training_accuracy = []
        valid_accuracy = []
        det_valid_accuracy = []
        new_valid_accuracy = []
        x_valid_accuracy = []
        testing_accuracy = []
        valid_preds = []
        det_valid_preds = []
        new_valid_preds = []
        train_losses = []
        valid_losses = []
        det_train_losses = []
        det_valid_losses = []
        new_train_losses = []
        new_valid_losses = []
        epochs = []

        classifier = LAC.LaplaceApproximationClassifier(n_features_selected=n_best_features)
        new_classifier = LCM.LaplaceApproximationClassifier(n_features_selected=n_best_features)
        det_classifier = DC.DeterministicClassifier(n_features_selected=n_best_features)
        OI_classifier = LAC.LaplaceApproximationClassifier(n_features_selected=n_best_features)
        OI_det_classifier = DC.DeterministicClassifier(n_features_selected=n_best_features)
        # initialize classifiers for IND and OOD
        for k in range(self.ntimes):
            running_loss = 0.0
            running_val_loss = 0.0
            det_running_loss = 0.0
            det_running_val_loss = 0.0
            new_running_loss = 0.0
            new_running_val_loss = 0.0
            flag = 0
            if (self.ntimes - k) <= 1:
                flag = 1

            '''for N times x K fold CV'''
            # train_indices, test_indices = self.cross_validate_Ntimes_Kfold(y_labels,ifold=k)
            '''for K fold CV by sequential splitting'''
            train_indices, valid_indices = self.cross_validate_sequential_split(y_cv)
            #OI_train_indices, OI_valid_indices = self.cross_validate_sequential_split(y_OI_cv)
            # split for IND and OOD
            '''for one fold in half half split'''
            # train_indices, test_indices = self.cross_validate_half_split(y_labels)

            for i in range(self.kfold):
                # fit for OOD and IND
                train_idx = train_indices.get(i)
                valid_idx = valid_indices.get(i)
                print(f'Times {str(k)}, Fold {str(i)}\n')
                y_train, y_valid = self.split_ydata(y_cv, train_idx, valid_idx)
                #OI_y_train, OI_y_valid = self.split_ydata(y_OI_cv, OI_train_idx, OI_valid_idx)

                x_train_fb = X_cv_extracted[train_idx]
                x_valid_fb = X_cv_extracted[valid_idx]

                print('X_train shape: ', x_train_fb.shape)

                train_loss = LAC.fit(classifier, x_train_fb, y_train, LAC.CrossEntropyWithLogPrior, flag)
                valid_loss, valid_preds = LAC.validate(classifier, x_valid_fb, y_valid, LAC.CrossEntropyWithLogPrior)
                new_train_loss = LCM.fit(new_classifier, x_train_fb, y_train, LCM.CrossEntropyWithLogPrior, flag)
                new_valid_loss, new_valid_preds = LCM.validate(new_classifier, x_valid_fb, y_valid, LCM.CrossEntropyWithLogPrior)
                #OI_train_loss = LAC.fit(OI_classifier, OI_x_train_fb, OI_y_train, LAC.CrossEntropyWithLogPrior, flag)
                #OI_valid_loss, OI_valid_preds = LAC.validate(OI_classifier, OI_x_valid_fb, OI_y_valid, LAC.CrossEntropyWithLogPrior)

                det_train_loss = DC.fit(det_classifier, x_train_fb, y_train, DC.XEL)
                det_valid_loss, det_valid_preds = DC.validate(det_classifier, x_valid_fb, y_valid, DC.XEL)
                #OI_det_train_loss = DC.fit(OI_det_classifier, OI_x_train_fb, OI_y_train, DC.XEL)
                #OI_det_valid_loss, OI_det_valid_preds = DC.validate(OI_det_classifier, OI_x_valid_fb, OI_y_valid, DC.XEL)

                val_acc = count_accuracy(valid_preds, y_valid)
                valid_accuracy.append(val_acc)
                det_val_acc = count_accuracy(det_valid_preds, y_valid)
                det_valid_accuracy.append(det_val_acc)
                new_val_acc = count_accuracy(new_valid_preds, y_valid)
                new_valid_accuracy.append(new_val_acc)
                x_valid_accuracy.append(k + i/self.kfold)

                running_loss += train_loss
                running_val_loss += valid_loss

                det_running_loss += det_train_loss
                det_running_val_loss += det_valid_loss

                new_running_loss += new_train_loss
                new_running_val_loss += new_valid_loss

            train_losses.append(running_loss)
            valid_losses.append(running_val_loss)
            det_train_losses.append(det_running_loss)
            det_valid_losses.append(det_running_val_loss)
            new_train_losses.append(new_running_loss)
            new_valid_losses.append(new_running_val_loss)
            epochs.append(k)

        # file with logs initialization
        now = datetime.now()
        now_str = now.strftime("%Y.%m.%d_%H-%M-%S")
        parent_dir = r"C:\Users\user\DataspellProjects\BNNproject1\fbcsp_code\experiment_logs"
        directory = now_str
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
        logs_path = os.path.join(path, 'logs.txt')
        logs = open(logs_path, mode='w')

        logs.write(now_str)
        logs.write('\n')
        logs.write('2 subject experiment\n')
        buffer = str('Subjects'+str(self.file_to_load)+', '+str(second_subj_file))
        logs.write(buffer)

        # Test prediction

        #print("LAC parameters: ", classifier.weights.weight, classifier.weights.bias)
        print("DC parameters: ", det_classifier.weights.weight, det_classifier.weights.bias)
        buffer = str('DC parameters:'+'\n'+'weights: '+str(det_classifier.weights.weight)+'\n'+'bias: '+str(det_classifier.weights.bias)+'\n')
        logs.write(buffer)
        print("LCM parameters: ", new_classifier.theta.weight)
        buffer = str('LCM parameters: '+'\n'+'weights: '+str(new_classifier.theta.weight)+'\n'+'covar_matrix: '+str(new_classifier.posterior_sigma)+'\n')
        logs.write(buffer)
        print('LCM covar_matrix: ', new_classifier.posterior_sigma)

        # TODO make predictions on another subject
        print('last epoch average fold training accuracies:')
        last_avg = (new_valid_accuracy[-1] + new_valid_accuracy[-2] + new_valid_accuracy[-3]) / 3
        print('Laplace: ', last_avg)
        det_last_avg = (det_valid_accuracy[-1] + det_valid_accuracy[-2] + det_valid_accuracy[-3]) / 3
        print('Deterministic: ', det_last_avg)
        buffer = str('last epoch average fold training accuracies:' + '\n')
        logs.write(buffer)
        buffer = str('Laplace: ' + str(last_avg) + '\n' + 'Deterministic' + str(det_last_avg) + '\n')
        logs.write(buffer)
        #test_predictions = LAC.predict(classifier, X_test_extracted)
        #OI_test_predictions = LAC.predict(OI_classifier, X_test_extracted)
        #test_acc = np.sum(test_predictions == y_test, dtype=np.float) / len(y_test)
        #print('Laplace classifier Testing accuracy: ', test_acc)
        det_test_predictions, det_total_loss = DC.predict(det_classifier, X_test_extracted, y_test, DC.XEL)
        #OI_det_test_predictions = DC.predict(OI_det_classifier, X_test_extracted)
        new_test_predictions, total_loss, ind_variances = LCM.predict(new_classifier, X_test_extracted, y_test, LCM.CrossEntropyWithLogPrior, n=500, verbose=True)
        ood_new_test_predictions, ood_total_loss, ood_variances = LCM.predict(new_classifier, ood_X_test_extracted, ood_y_test, LCM.CrossEntropyWithLogPrior, n=500, verbose=True)
        test_acc = np.sum(new_test_predictions == y_test, dtype=np.float) / len(y_test)
        print('Laplace modified classifier Testing accuracy: ', test_acc)
        buffer = str('Laplace modified classifier Testing accuracy: '+str(test_acc)+'\n')
        logs.write(buffer)
        test_acc = np.sum(det_test_predictions == y_test, dtype=np.float) / len(y_test)
        print('Deterministic classifier Testing accuracy: ', test_acc)
        buffer = str('Deterministic classifier Testing accuracy: '+str(test_acc)+'\n')
        logs.write(buffer)
        #test_probs = LAC.predict_proba(classifier, X_test_extracted)
        #det_test_probs = DC.predict_proba(det_classifier, X_test_extracted)
        # new_test_probs = LCM.predict_proba(new_classifier, X_test_extracted)

        # counting delta loss between det and lap versus num of samples
        sample_nums = [1, 5, 10, 25, 50, 75, 100, 150, 200, 250, 500]
        loss_delta = []
        for num in sample_nums:
            new_preds, total_loss, v = LCM.predict(new_classifier, X_test_extracted, y_test, LCM.CrossEntropyWithLogPrior, n=num)
            loss_delta.append(total_loss-det_total_loss)

        # IND and OOD mean variances comparison
        #var_ind, var_ood = domain_split(variances, y_test)
        #preds_ind, preds_ood = domain_split(new_test_predictions, y_test)
        print('Mean variance for IND samples: ', np.average(ind_variances),'\n', 'Mean variance for OOD samples: ', np.average(ood_variances))
        buffer = str('Mean variance for IND samples: ' + str(np.average(ind_variances)) + '\n'+ 'Mean variance for OOD samples: ' + str(np.average(ood_variances))+'\n')
        logs.write(buffer)

        #sd = []
        #for g in variances:
        #    sd.append(np.sqrt(g))
        #sd_ind, sd_ood = domain_split(sd, y_test)

        ood_sd = []
        ind_sd = []
        for g in ood_variances:
            ood_sd.append(np.sqrt(g))
        for g in ind_variances:
            ind_sd.append(np.sqrt(g))

        #det_ind, det_ood = domain_split(det_test_probs, y_test)
        #ind, ood = domain_split(test_probs, y_test)
        #det_entropies, det_ratio = entropy_counter(det_ind, det_ood)
        #entropies, ratio = entropy_counter(ind, ood)

        # In-domain Test prediction

        #test_in_domain = count_in_domain_accuracy(test_predictions, y_test)
        #print('Laplace classifier In-Domain Testing accuracy: ',test_in_domain)
        det_test_in_domain = count_in_domain_accuracy(det_test_predictions, y_test)
        print('Deterministic classifier In-Domain Testing accuracy: ', det_test_in_domain)
        buffer = str('Deterministic classifier In-Domain Testing accuracy: '+str(det_test_in_domain)+'\n')
        logs.write(buffer)
        new_test_in_domain = count_in_domain_accuracy(new_test_predictions, y_test)
        print('Laplace modified classifier In-Domain Testing accuracy: ', new_test_in_domain)
        buffer = str('Laplace modified classifier In-Domain Testing accuracy: '+str(new_test_in_domain)+'\n')
        logs.write(buffer)



        # roc-auc data getting

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
        t = 0.07
        while t < 0.155:
            x, y = roc_dots_counter(variances, labels, t)
            xs.append(x)
            ys.append(y)
            t += 0.005

        a = auc(xs, ys)

        logs.write('ROC datapoints'+'\n')
        for i, j in zip(variances, labels):
            logs.write('variance: '+str(i)+' label: '+str(j)+'\n')
        logs.close()
        # Model saving (DEPRECATED)

        #if test_in_domain > 0.6 and det_test_in_domain > 0.6: # TODO make dir to put CSPs into, if exists, put there
        #   joblib.dump(feature_extractor, 'feature_extractor.pkl')
        #  for i in range(len(CSPs)):
        #     filename = r'C:\Users\user\DataspellProjects\BNNproject1\CSPs\CSP_' + str(i) + '.pkl' #FIXME  maybe delete all files in CSPs before replacing?
        #    joblib.dump(CSPs[i], filename)
        #torch.save(classifier.state_dict(), 'saved_model.pt')
        #torch.save(det_classifier.state_dict(), 'saved_det_model.pt')
        #print('Models saved.')


        # Graph plotting

        #print(det_valid_accuracy)

        fig = plt.figure()
        loss_curves = fig.add_subplot(121, title='Loss Curves', xlabel='epoch', ylabel='Loss')
        #loss_curves.plot(epochs, train_losses, 'b-', label='train')
        #loss_curves.plot(epochs, valid_losses, 'r-', label='valid')
        loss_curves.plot(epochs, det_train_losses, 'g-', label='det_train')
        loss_curves.plot(epochs, det_valid_losses, 'm-', label='det_valid')
        loss_curves.plot(epochs, new_train_losses, 'r--', label='lap_mod_train')
        loss_curves.plot(epochs, new_valid_losses, 'k--', label='lap_mod_valid')
        loss_curves.legend()

        valid_acc = fig.add_subplot(122, title='Validation accuracy', xlabel='epoch', ylabel='Accuracy')
        #valid_acc.plot(x_valid_accuracy, valid_accuracy, 'b-', label='laplace')
        valid_acc.plot(x_valid_accuracy, det_valid_accuracy, 'g-', label='deterministic')
        valid_acc.plot(x_valid_accuracy, new_valid_accuracy, 'm-', label='laplace_mod')
        valid_acc.legend()
        figpath = os.path.join(path, 'graphs_2subj.jpg')
        fig.savefig(figpath)



        # TODO plot ROC-curves for in-domain and out-of-domain data points

        # ROC-AUC metrics implementation (DEPRECATED)

        #lap_probs = LAC.predict_proba(OI_classifier, X_test_extracted)
        #det_probs = DC.predict_proba(OI_det_classifier, X_test_extracted)
        # print(lap_probs)
        # print(y_test)
        #lap_auc = roc_auc_score(y_OI_test, lap_probs, multi_class='ovr')
        #det_auc = roc_auc_score(y_OI_test, det_probs, multi_class='ovr')
        #lap_fpr, lap_tpr, lap_threshold = roc_curve(y_OI_test, lap_probs)
        #det_fpr, det_tpr, det_threshold = roc_curve(y_OI_test, det_probs)
        #lap_roc_auc = auc(lap_fpr, lap_tpr)
        #det_roc_auc = auc(det_fpr, det_tpr)
        #print('Laplace ROC-AUC: ', lap_auc)
        #print('Deterministic ROC-AUC: ', det_auc)


        #fig2 = plt.figure()
        #ax0 = fig2.add_subplot(121, title='Laplace_ROC')
        #ax0.plot(lap_fpr, lap_tpr, 'b-', label='Laplace_ROC, area = %0.3f' % lap_roc_auc)
        #ax0.plot([0, 1], [0, 1], 'c--')
        #ax1 = fig2.add_subplot(122, title='Deterministic_ROC')
        #ax1.plot(det_fpr, det_tpr, 'b-', label='Deterministic_ROC, area = %0.3f' % det_roc_auc)
        #ax1.plot([0, 1], [0, 1], 'c--')
        #fig2.savefig('ROC-AUC_Curves.jpg')

        fig3 = plt.figure()
        loss_discrepancy = fig3.add_subplot(111, title='loss delta', xlabel='n', ylabel='Loss_delta')
        loss_discrepancy.plot(sample_nums, loss_delta, 'b-', label='loss_delta vs n')
        loss_discrepancy.plot([0, 500], [0, 0], 'c--')
        figpath = os.path.join(path, 'loss_delta_vs_n.jpg')
        fig3.savefig(figpath)

        fig4 = plt.figure()
        ind_variances_ = fig4.add_subplot(121,title='ind_preds', xlabel='sample', ylabel='pred+-sd')
        ind_variances_.errorbar([n for n in range(len(new_test_predictions))], new_test_predictions, yerr=ind_sd, fmt='o')
        ind_variances_.set_xlim([0, 20])
        ind_variances_.set_ylim([-2.0, 2.0])
        ood_variances_ = fig4.add_subplot(122,title='ood_preds', xlabel='sample', ylabel='pred+-sd')
        ood_variances_.errorbar([n for n in range(len(ood_new_test_predictions))], ood_new_test_predictions, yerr=ood_sd, fmt='o')
        ood_variances_.set_xlim([0, 20])
        ood_variances_.set_ylim([-2.0, 2.0])
        figpath = os.path.join(path, 'ind_ood_2subj_variances.jpg')
        fig4.savefig(figpath)

        fig5 = plt.figure()
        roc = fig5.add_subplot(111, title='variance criteria roc')
        roc.plot(xs,ys, 'b-', label='roc, area = %0.3f' % a)
        roc.plot([0,1],[0,1], 'c--')
        roc.legend()
        figpath = os.path.join(path, 'variance_roc.jpg')
        fig5.savefig(figpath)

        variances = ind_variances
        predictions = new_test_predictions
        targets = list(y_test)
        for i in range(len(ood_variances)):
            variances.append(ood_variances[i])
            predictions.append(ood_new_test_predictions[i])
            targets.append(ood_y_test[i])

        true_vars, false_vars = distinguish_variances(variances, predictions, targets)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        fig.suptitle('Variances distribution among truly and falsely classified samples')
        sns.histplot(ax=axes[0], x=true_vars, bins=7, kde=True)
        sns.histplot(ax=axes[1], x=false_vars, bins=7, kde=True)
        figpath = os.path.join(path, 'variance_histograms.jpg')
        fig.savefig(figpath)

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
        train_idx =np.array([])
        test_idx = np.array([])
        for cls in unique_classes:
            cls_indx = all_labels[np.where(y_labels==cls)]
            if len(train_idx)==0:
                train_idx = cls_indx[:math.ceil(len(cls_indx)/2)]
                test_idx = cls_indx[math.ceil(len(cls_indx)/2):]
            else:
                train_idx=np.append(train_idx,cls_indx[:math.ceil(len(cls_indx)/2)])
                test_idx=np.append(test_idx,cls_indx[math.ceil(len(cls_indx)/2):])

        train_indices = {0:train_idx}
        test_indices = {0:test_idx}

        return train_indices, test_indices

    def split_xdata(self,eeg_data, train_idx, test_idx):
        x_train_fb=np.copy(eeg_data[:,train_idx,:,:])
        x_test_fb=np.copy(eeg_data[:,test_idx,:,:])
        return x_train_fb, x_test_fb

    def split_ydata(self,y_true, train_idx, test_idx):
        y_train = np.copy(y_true[train_idx])
        y_test = np.copy(y_true[test_idx])

        return y_train, y_test

    def get_multi_class_label(self,y_predicted, cls_interest=0):
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
        y_predict_multi = np.asarray([np.argmin(y_predicted[i,:]) for i in range(y_predicted.shape[0])])
        return y_predict_multi


class FilterBank:
    def __init__(self,fs):
        self.fs = fs
        self.f_trans = 2
        self.f_pass = np.arange(4,40,4)  # (lower_bound, upper_bound, step)
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
            self.filter_coeff.update({i:{'b':b,'a':a}})

        return self.filter_coeff

    def filter_data(self,eeg_data,window_details={}):
        n_trials, n_channels, n_samples = eeg_data.shape
        if window_details:
            n_samples = int(self.fs*(window_details.get('tmax')-window_details.get('tmin')))+1
        filtered_data=np.zeros((len(self.filter_coeff),n_trials,n_channels,n_samples))
        for i, fb in self.filter_coeff.items():
            b = fb.get('b')
            a = fb.get('a')
            eeg_data_filtered = np.asarray([signal.lfilter(b,a,eeg_data[j,:,:]) for j in range(n_trials)])
            if window_details:
                eeg_data_filtered = eeg_data_filtered[:,:,int((4.5+window_details.get('tmin'))*self.fs):int((4.5+window_details.get('tmax'))*self.fs)+1]
            filtered_data[i,:,:,:]=eeg_data_filtered

        return filtered_data


def count_in_domain_accuracy(predictions,targets,domain=[0,1]): # function to calculate accuracy on only in-domain datapoints
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
