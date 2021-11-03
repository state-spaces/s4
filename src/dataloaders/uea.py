""" Load data for UEA datasets, in particular CharacterTrajectories

Adapted from https://github.com/patrick-kidger/NeuralCDE/blob/master/experiments/datasets/uea.py
"""
import os
import pathlib
import urllib.request
import zipfile
import sklearn.model_selection
# import sktime.utils.data_io
from sktime.utils.data_io import load_from_tsfile_to_dataframe
import numpy as np
import torch
import collections as co

# TODO deal with this path properly as an option
here = pathlib.Path(__file__).resolve().parent
valid_dataset_names = {
    'ArticularyWordRecognition',
    'FaceDetection',
    'NATOPS',
    'AtrialFibrillation',
    'FingerMovements',
    'PEMS - SF',
    'BasicMotions',
    'HandMovementDirection',
    'PenDigits',
    'CharacterTrajectories',
    'Handwriting',
    'PhonemeSpectra',
    'Cricket',
    'Heartbeat',
    'RacketSports',
    'DuckDuckGeese',
    'InsectWingbeat',
    'SelfRegulationSCP1',
    'EigenWorms',
    'JapaneseVowels',
    'SelfRegulationSCP2',
    'Epilepsy',
    'Libras',
    'SpokenArabicDigits',
    'ERing',
    'LSST',
    'StandWalkJump',
    'EthanolConcentration',
    'MotorImagery',
    'UWaveGestureLibrary',
}

def download():
    """ Download data if not exists """
    base_base_loc = here # / 'data'
    base_loc = base_base_loc / 'uea'
    loc = base_loc / 'Multivariate2018_ts.zip'
    if os.path.exists(loc):
        return
    if not os.path.exists(base_base_loc):
        os.mkdir(base_base_loc)
    if not os.path.exists(base_loc):
        os.mkdir(base_loc)
    urllib.request.urlretrieve('http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip',
                               str(loc))

    with zipfile.ZipFile(loc, 'r') as f:
        f.extractall(str(base_loc))

def load_data(dataset_name):
    """ Load X, y numpy data for given dataset """
    assert dataset_name in valid_dataset_names, "Must specify a valid dataset name."

    # base_filename = here / 'data' / 'UEA' / 'Multivariate_ts' / dataset_name / dataset_name
    base_filename = here / 'uea' / 'Multivariate_ts' / dataset_name / dataset_name
    train_X, train_y = load_from_tsfile_to_dataframe(str(base_filename) + '_TRAIN.ts')
    test_X, test_y = load_from_tsfile_to_dataframe(str(base_filename) + '_TEST.ts')
    train_X = train_X.to_numpy()
    test_X = test_X.to_numpy()
    X = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)
    return X, y

def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + '.pt')

def load_processed_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors

def wrap_data(train_X, val_X, test_X, train_y, val_y, test_y, train_final_index, val_final_index,
              test_final_index,
              ):
    """ Wrap data into Pytorch Dataset. """

    train_dataset = torch.utils.data.TensorDataset(train_X, train_y,
            # train_final_index
            )
    val_dataset = torch.utils.data.TensorDataset(val_X, val_y,
            # val_final_index
            )
    test_dataset = torch.utils.data.TensorDataset(test_X, test_y,
            # test_final_index
            )

    return train_dataset, val_dataset, test_dataset

def split_data(tensor, stratify):
    # 0.7/0.15/0.15 train/val/test split
    (train_tensor, testval_tensor,
     train_stratify, testval_stratify) = sklearn.model_selection.train_test_split(tensor, stratify,
                                                                                  train_size=0.7,
                                                                                  random_state=0,
                                                                                  shuffle=True,
                                                                                  stratify=stratify)

    val_tensor, test_tensor = sklearn.model_selection.train_test_split(testval_tensor,
                                                                       train_size=0.5,
                                                                       random_state=1,
                                                                       shuffle=True,
                                                                       stratify=testval_stratify)
    return train_tensor, val_tensor, test_tensor


def normalize_data(X, y):
    """ Normalize data by training statistics per channel.

    X: data tensor with channels as last dimension
    """
    train_X, _, _ = split_data(X, y)
    out = []
    for Xi, train_Xi in zip(X.unbind(dim=-1), train_X.unbind(dim=-1)):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        mean = train_Xi_nonan.mean()  # compute statistics using only training data.
        std = train_Xi_nonan.std()
        out.append((Xi - mean) / (std + 1e-5))
    out = torch.stack(out, dim=-1)
    return out


def preprocess_data(
        X, y,
        final_index,
        # append_times,
        append_intensity,
    ):
    X = normalize_data(X, y)

    # Append extra channels together. Note that the order here: time, intensity, original, is important, and some models
    # depend on that order.
    augmented_X = []
    # if append_times:
    #     augmented_X.append(times.unsqueeze(0).repeat(X.size(0), 1).unsqueeze(-1))
    if append_intensity: # Note this will append #channels copies of the same intensity
        intensity = ~torch.isnan(X)  # of size (batch, stream, channels)
        intensity = intensity.to(X.dtype).cumsum(dim=1)
        augmented_X.append(intensity)
    augmented_X.append(X)
    if len(augmented_X) == 1:
        X = augmented_X[0]
    else:
        X = torch.cat(augmented_X, dim=2)

    train_X, val_X, test_X = split_data(X, y) # TODO split data should just return y? or list of indices corresponding to splits
    train_y, val_y, test_y = split_data(y, y)
    train_final_index, val_final_index, test_final_index = split_data(final_index, y)

    # train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, train_X)
    # val_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, val_X)
    # test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, test_X)

    in_channels = X.size(-1)

    return (
            # times,
            # train_coeffs, val_coeffs, test_coeffs,
            train_X, val_X, test_X,
            train_y, val_y, test_y,
            train_final_index, val_final_index, test_final_index,
            in_channels
            )

def process_data(dataset_name, intensity):
    # We begin by loading both the train and test data and using our own train/val/test split.
    # The reason for this is that (a) by default there is no val split and (b) the sizes of the train/test splits are
    # really janky by default. (e.g. LSST has 2459 training samples and 2466 test samples.)


    X, y = load_data(dataset_name)

    lengths = torch.tensor([len(Xi[0]) for Xi in X])
    final_index = lengths - 1
    maxlen = lengths.max()
    # X is now a numpy array of shape (batch, channel)
    # Each channel is a pandas.core.series.Series object of length corresponding to the length of the time series
    def _pad(channel, maxlen):
        channel = torch.tensor(channel)
        out = torch.full((maxlen,), channel[-1])
        out[:channel.size(0)] = channel
        return out
    X = torch.stack([torch.stack([_pad(channel, maxlen) for channel in batch], dim=0) for batch in X], dim=0)
    # X is now a tensor of shape (batch, channel, length)
    X = X.transpose(-1, -2)
    # X is now a tensor of shape (batch, length, channel)
    times = torch.linspace(0, X.size(1) - 1, X.size(1))


    # generator = torch.Generator().manual_seed(56789)
    # for Xi in X:
    #     removed_points = torch.randperm(X.size(1), generator=generator)[:int(X.size(1) * missing_rate)].sort().values
    #     Xi[removed_points] = float('nan')

    # Now fix the labels to be integers from 0 upwards
    targets = co.OrderedDict()
    counter = 0
    for yi in y:
        if yi not in targets:
            targets[yi] = counter
            counter += 1
    y = torch.tensor([targets[yi] for yi in y])


    (train_X, val_X, test_X,
            train_y, val_y, test_y,
            train_final_index, val_final_index,
     test_final_index,
     input_channels) = preprocess_data(
             X, y, final_index,
             # append_times=True,
             append_intensity=intensity,
             )

    num_classes = counter

    assert num_classes >= 2, f"Have only {num_classes} classes."

    return (
        # times,
        train_X, val_X, test_X,
        train_y, val_y, test_y,
        train_final_index, val_final_index, test_final_index,
        num_classes, input_channels
    )

def get_data(
        dataset_name,
        intensity,
        train_hz=1,
        eval_hz=1,
        timestamp=False,
        train_ts=1,
        eval_ts=1,
    ):
    # We begin by loading both the train and test data and using our own train/val/test split.
    # The reason for this is that (a) by default there is no val split and (b) the sizes of the train/test splits are
    # really janky by default. (e.g. LSST has 2459 training samples and 2466 test samples.)

    assert dataset_name in valid_dataset_names, "Must specify a valid dataset name."

    base_base_loc = here / 'processed_data'
    base_loc = base_base_loc / 'uea'
    loc = base_loc / (dataset_name + ('_intensity' if intensity else ''))
    try:
        tensors = load_processed_data(loc)
        train_X           = tensors['train_X']
        val_X             = tensors['val_X']
        test_X            = tensors['test_X']
        train_y           = tensors['train_y']
        val_y             = tensors['val_y']
        test_y            = tensors['test_y']
        train_final_index = tensors['train_final_index']
        val_final_index   = tensors['val_final_index']
        test_final_index  = tensors['test_final_index']
        num_classes       = int(tensors['num_classes'])
        input_channels    = int(tensors['input_channels'])
    except:
        print(f"Could not find preprocessed data. Loading {dataset_name}...")
        download() # download the UEA data if necessary
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(base_loc):
            os.mkdir(base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        ( train_X, val_X, test_X, train_y, val_y, test_y, train_final_index, val_final_index,
         test_final_index, num_classes, input_channels ) = process_data(dataset_name, intensity)
        save_data(
            loc,
            train_X=train_X, val_X=val_X, test_X=test_X,
            train_y=train_y, val_y=val_y, test_y=test_y, train_final_index=train_final_index,
            val_final_index=val_final_index, test_final_index=test_final_index,
            num_classes=torch.as_tensor(num_classes), input_channels=torch.as_tensor(input_channels),
        )

    return (
        train_X, val_X, test_X,
        train_y, val_y, test_y,
        train_final_index, val_final_index, test_final_index,
        num_classes, input_channels,
    )


def _subsample(X, hz=1, uniform=True, mask=False):
    """ Subsample X non-uniformly at hz frequency

    Input:
    - X : (dim, length)

    Returns:
    - Subsampled X
    - Original timestamps of the preserved data

    If mask=True, then the dimensionality of X is preserved; dropped inputs are replaced by 0,
    and an additional channel is appended indicating the positions original elements
    """
    L = X.shape[1]
    # create subsampler
    if uniform:
        removed_points = torch.arange(int(L*hz)) // hz
        removed_points = removed_points.to(int)
        time_gen = lambda: removed_points
    else:
        generator = torch.Generator().manual_seed(56789)
        time_gen = lambda: torch.randperm(L, generator=generator)[:int(L*hz)].sort().values

    X_ = []
    T_ = []
    for Xi in X:
        times = time_gen()
        if mask:
            Xi_copy = Xi.clone()
            Xi_copy[times] = 0.0
            Xi_ = Xi.clone() - Xi_copy # Keep original data at [times]
            mask_ = torch.zeros_like(Xi_[:,:1])
            mask_[times] = 1.0
            Xi_ = torch.cat([Xi_, mask_], dim=1)
        else:
            Xi_ = Xi[times]
        times_ = times.to(torch.float32).unsqueeze(-1)
        X_.append(Xi_)
        T_.append(times_)
    return torch.stack(X_, dim=0), torch.stack(T_, dim=0)

def postprocess_data(
        train_X, val_X, test_X,
        train_y, val_y, test_y,
        train_final_index, val_final_index, test_final_index,
        train_hz=1,
        eval_hz=1,
        train_uniform=True,
        eval_uniform=True,
        timestamp=False,
        train_ts=1,
        eval_ts=1,
        mask=False,
    ):
    """
    train_hz, eval_hz: subsampling multiplier of original data
        e.g. train_hz=0.5 means data is sampled at half speed, so remove every other element of the sequence
        Since the original data is sampled from a trajectory at 200Hz, this corresponds to a sampling rate of 100Hz
    train_uniform, eval_uniform: whether subsampling is uniformly spaced or random
    timestamp: data comes with timestamps
    train_ts, eval_ts: timestamp multiplier
    mask: Whether to mask the data instead of removing. Should not be simultaneously True with timestamp

    Example configurations:
    train_hz=1.0, eval_hz=0.5, {train,eval}_uniform=True, timestamp=False
    - non-timestamped, uniformly sampled data, where evaluation sequences have every other element removed

    {train,eval}_uniform=False, timestamp=True, train_ts=1.0, eval_ts=0.5
    - timestamped, randomly sampled data, where evaluation sequences have timestamps halved

    Both of the above configurations test train->evaluation generalization of halving the timescale frequency, either from the measurement sampling rate decreasing (from 200Hz -> 100hz), or the subject drawing half as fast.
    """


    train_X, train_T = _subsample(train_X, train_hz, train_uniform, mask)
    val_X, val_T = _subsample(val_X, eval_hz, eval_uniform, mask)
    test_X, test_T = _subsample(test_X, eval_hz, eval_uniform, mask)

    if timestamp:
        train_X = torch.cat([train_ts*train_T, train_X], dim=-1)
        val_X = torch.cat([eval_ts*val_T, val_X], dim=-1)
        test_X = torch.cat([eval_ts*test_T, test_X], dim=-1)

    train_dataset, val_dataset, test_dataset = wrap_data(
            # times,
            train_X, val_X, test_X,
            train_y, val_y, test_y,
            train_final_index, val_final_index, test_final_index
            )
    return train_dataset, val_dataset, test_dataset # , num_classes, input_channels


if __name__ == '__main__':
    *data, numclasses, input_channels = get_data(
        'CharacterTrajectories',
        intensity=False,
    )

    train_dataset, val_dataset, test_dataset = postprocess_data(
        *data,
        train_hz=1,
        eval_hz=0.5,
        train_uniform=True,
        eval_uniform=False,
        timestamp=True,
        train_ts=1,
        eval_ts=0.5,
    )

    breakpoint()
