# TODO copy all utilities from MLEngineRefined
from scipy.stats import entropy
from braindecode.datasets import MOABBDataset


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

    splitted = windows_dataset.split('session')
    train_set = splitted['session_T']
    test_set = splitted['session_E']
    return train_set, test_set


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
