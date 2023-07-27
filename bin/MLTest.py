# TODO here should be the testing file of a  saved model in order to collect some metrics without fitting
import numpy as np
import torch
import scipy.signal as signal
from scipy.signal import cheb2ord
import bin.LoadData as LoadData
import bin.DeterministicClassifier as DC
import bin.Preprocess as Preprocess
import bin.LaplaceApproxClassif as LAC
import matplotlib.pyplot as plt
import sklearn.feature_selection
from sklearn.feature_selection import mutual_info_classif
from mne.decoding import CSP
import joblib


class MLTest:
    def __init__(self,data_path='',file_to_load='',subject_id='',sessions=[1, 2],ntimes=1,kfold=2,m_filters=2,window_details={}, model='', det_model=''):
        self.data_path = data_path
        self.subject_id=subject_id
        self.file_to_load = file_to_load
        self.sessions = sessions
        self.kfold = kfold
        self.ntimes=ntimes
        self.window_details = window_details
        self.m_filters = m_filters
        self.model = model
        self.det_model = det_model

    def experiment(self):

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


        X = filtered_data
        y = y_labels

       #TODO load CSPs -> load MIBIFs -> load classifiers -> predict -> plot graphs

        # CSP loading + filtering

        new_X = []

        for i in range(filtered_data.shape[0]):
            filename = r'C:\Users\user\DataspellProjects\BNNproject1\CSPs\CSP_' + str(i) +'.pkl'
            csp = joblib.load(filename)
            tmp_X= X[i,:,:,:]
            X_filtered = csp.transform(tmp_X)

            if i == 0:
                new_X = X_filtered
            else:
                new_X = np.concatenate((new_X, X_filtered), axis=1)

        # Feature extraction + extractor loading

        feature_extractor = joblib.load('feature_extractor.pkl')
        X_extracted = feature_extractor.transform(new_X)

        print('X_extracted shape: ', X_extracted.shape)

        # classifiers loading
        classifier = LAC.LaplaceApproximationClassifier(n_features_selected=8) #FIXME Load n_features_selected_from some file HOW?
        det_classifier = DC.DeterministicClassifier(n_features_selected=8)
        classifier_dict = torch.load(self.model)
        det_classifier_dict = torch.load(self.det_model)
        classifier.load_state_dict(classifier_dict)
        det_classifier.load_state_dict(det_classifier_dict)

        # Test prediction

        test_predictions = LAC.predict(classifier, X_extracted) # FIXME AttributeError: 'collections.OrderedDict' object has no attribute 'weights'
        test_acc = np.sum(test_predictions == y, dtype=np.float) / len(y)
        print('Laplace classifier Testing accuracy: ', test_acc)
        det_test_predictions = DC.predict(det_classifier, X_extracted)
        test_acc = np.sum(det_test_predictions == y, dtype=np.float) / len(y)
        print('Deterministic classifier Testing accuracy: ', test_acc)

        # In-domain Test prediction

        test_in_domain = count_in_domain_accuracy(test_predictions, y)
        print('Laplace classifier In-Domain Testing accuracy: ', test_in_domain)
        det_test_in_domain = count_in_domain_accuracy(det_test_predictions, y)
        print('Deterministic classifier In-Domain Testing accuracy: ', det_test_in_domain)

        # Graph plotting

        # TODO plot ROC-curves for in-domain and out-of-domain data points
        # TODO smth with entropy?

        #lap_precision = precision_score(y_test, test_predictions, average='micro')
        #det_precision = precision_score(y_test, det_test_predictions, average='micro')
        # print('Laplace classifier precision: ', lap_precision)
        # print('Deterministic classifier precision', det_precision)

        # print('Laplace preds: ', test_predictions)
        # print('Det preds: ', det_test_predictions)

        # ROC-AUC metrics implementation

        # lap_probs = LAC.predict_proba(classifier, X_test_extracted)
        # det_probs = DC.predict_proba(det_classifier, X_test_extracted)
        # print(lap_probs)
        # print(y_test)
        # lap_auc = roc_auc_score(y_test, lap_probs, multi_class='ovr')
        # det_auc = roc_auc_score(y_test, det_probs, multi_class='ovr')
        # lap_fpr, lap_tpr, lap_threshold = roc_curve(y_test, lap_probs)
        # det_fpr, det_tpr, det_threshold = roc_curve(y_test, det_probs)
        # lap_roc_auc = auc(lap_fpr, lap_tpr)
        # det_roc_auc = auc(det_fpr, det_tpr)
        # print('Laplace ROC-AUC: ', lap_auc)
        # print('Deterministic ROC-AUC: ', det_auc)


        # fig = plt.figure()
        # ax0 = fig.add_subplot(121, title='Laplace_ROC')
        # ax0.plot(lap_fpr, lap_tpr, 'b-', label='Laplace_ROC, area = %0.3f' % lap_roc_auc)
        # ax0.plot([0, 1], [0, 1], 'c--')
        # ax1 = fig.add_subplot(122, title='Deterministic_ROC')
        # ax1.plot(det_fpr, det_tpr, 'b-', label='Deterministic_ROC, area = %0.3f' % det_roc_auc)
        # ax1.plot([0, 1], [0, 1], 'c--')
        # fig.savefig('ROC-AUC_Curves.jpg')




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
        skf_model = StratifiedKFold(n_splits=self.kfold, shuffle=False)
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
        self.f_pass = np.arange(4,40,4)
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


