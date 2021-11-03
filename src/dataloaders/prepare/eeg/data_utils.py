import numpy as np
import random
import os
import sys

sys.path.append("../")
import pyedflib

# from constants import INCLUDED_CHANNELS, FREQUENCY
from scipy.fftpack import fft
from scipy.signal import resample

INCLUDED_CHANNELS = [
    "EEG FP1",
    "EEG FP2",
    "EEG F3",
    "EEG F4",
    "EEG C3",
    "EEG C4",
    "EEG P3",
    "EEG P4",
    "EEG O1",
    "EEG O2",
    "EEG F7",
    "EEG F8",
    "EEG T3",
    "EEG T4",
    "EEG T5",
    "EEG T6",
    "EEG FZ",
    "EEG CZ",
    "EEG PZ",
]

FREQUENCY = 200


def computeFFT(signals, n):
    """
    Args:
        signals: EEG signals, (number of channels, number of data points)
        n: length of positive frequency terms of fourier transform
    Returns:
        FT: log amplitude of FFT of signals, (number of channels, number of data points)
        P: phase spectrum of FFT of signals, (number of channels, number of data points)
    """
    # fourier transform
    fourier_signal = fft(signals, n=n, axis=-1)  # FFT on the last dimension

    # only take the positive freq part
    idx_pos = int(np.floor(n / 2))
    fourier_signal = fourier_signal[:, :idx_pos]
    amp = np.abs(fourier_signal)
    amp[amp == 0.0] = 1e-8  # avoid log of 0

    FT = np.log(amp)
    P = np.angle(fourier_signal)

    return FT, P


def get_swap_pairs(channels):
    """
    Swap select adjacenet channels
    Args:
        channels: list of channel names
    Returns:
        list of tuples, each a pair of channel indices being swapped
    """
    swap_pairs = []
    if ("EEG FP1" in channels) and ("EEG FP2" in channels):
        swap_pairs.append((channels.index("EEG FP1"), channels.index("EEG FP2")))
    if ("EEG Fp1" in channels) and ("EEG Fp2" in channels):
        swap_pairs.append((channels.index("EEG Fp1"), channels.index("EEG Fp2")))
    if ("EEG F3" in channels) and ("EEG F4" in channels):
        swap_pairs.append((channels.index("EEG F3"), channels.index("EEG F4")))
    if ("EEG F7" in channels) and ("EEG F8" in channels):
        swap_pairs.append((channels.index("EEG F7"), channels.index("EEG F8")))
    if ("EEG C3" in channels) and ("EEG C4" in channels):
        swap_pairs.append((channels.index("EEG C3"), channels.index("EEG C4")))
    if ("EEG T3" in channels) and ("EEG T4" in channels):
        swap_pairs.append((channels.index("EEG T3"), channels.index("EEG T4")))
    if ("EEG T5" in channels) and ("EEG T6" in channels):
        swap_pairs.append((channels.index("EEG T5"), channels.index("EEG T6")))
    if ("EEG O1" in channels) and ("EEG O2" in channels):
        swap_pairs.append((channels.index("EEG O1"), channels.index("EEG O2")))

    return swap_pairs


def random_augmentation(EEG_seq):
    """
    Random augmentation of EEG sequence by randomly swapping channels
    and randomly scaling the signals.
    Args:
        EEG_seq: shape (seq_length, num_nodes, num_data_point)
    Returns:
        augmented signals, same shape as input
    """
    for pair in get_swap_pairs(INCLUDED_CHANNELS):
        if random.choice([True, False]):
            EEG_seq[:, [pair[0], pair[1]], :] = EEG_seq[:, [pair[1], pair[0]], :]
    if random.choice([True, False]):
        EEG_seq = EEG_seq * np.random.uniform(0.8, 1.2)
    return EEG_seq


def getOrderedChannels(file_name, verbose, labels_object, channel_names):
    labels = list(labels_object)
    for i in range(len(labels)):
        labels[i] = labels[i].split("-")[0]

    ordered_channels = []
    for ch in channel_names:
        try:
            ordered_channels.append(labels.index(ch))
        except:
            if verbose:
                print(file_name + " failed to get channel " + ch)
            raise Exception("channel not match")
    return ordered_channels


def getChannelIndices(labelsObject, channel_names):
    """
    labelsObject: labels object from eeghdf file
    channel_names: list of channel names
    """
    labels = labelsObject
    channelIndices = [labels.index(ch) for ch in channel_names]
    return channelIndices


def getSeizureTimes(file_name):
    """
    Args:
        file_name: file name of .edf file etc.
    Returns:
        seizure_times: list of times of seizure onset in seconds
    """
    tse_file = file_name.split(".edf")[0] + ".tse_bi"

    seizure_times = []
    with open(tse_file) as f:
        for line in f.readlines():
            if "seiz" in line:  # if seizure
                # seizure start and end time
                seizure_times.append(
                    [
                        float(line.strip().split(" ")[0]),
                        float(line.strip().split(" ")[1]),
                    ]
                )
    return seizure_times


def getEDFsignals(edf):
    n = edf.signals_in_file
    samples = edf.getNSamples()[0]
    signals = np.zeros((n, samples))
    for i in range(n):
        try:
            signals[i, :] = edf.readSignal(i)
        except:
            pass
    return signals


def resampleData(signals, to_freq=200, window_size=4):
    """
    Resample signals from its original sampling freq to another freq
    Args:
        signals: EEG signal slice, (num_channels, num_data_points)
        to_freq: Re-sampled frequency in Hz
        window_size: time window in seconds
    Returns:
        resampled: (num_channels, resampled_data_points)
    """
    num = int(to_freq * window_size)
    resampled = resample(signals, num=num, axis=1)
    return resampled


class StandardScaler:
    """
    Standardize the input
    """

    def __init__(self, mean, std):
        self.mean = mean  # (1,num_nodes,1)
        self.std = std  # (1,num_nodes,1)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Masked inverse transform
        Args:
            data: data for inverse scaling
            is_tensor: whether data is a tensor
            device: device
        """
        return data * self.std + self.mean
