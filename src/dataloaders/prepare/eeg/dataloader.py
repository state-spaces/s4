from pathlib import Path
from scipy.signal import correlate

# import scipy
import pickle
import os
import numpy as np
import h5py
import torch
import json
from torch.utils.data import Dataset, DataLoader

from datasets.eeg.data_utils import (
    # from data_utils import (
    getOrderedChannels,
    getEDFsignals,
    getSeizureTimes,
    computeFFT,
    getChannelIndices,
    StandardScaler,
    get_swap_pairs,
    resampleData,
)

from datasets.eeg.constants import (
    # from constants import (
    INCLUDED_CHANNELS,
    FREQUENCY,
    INCLUDED_CHANNELS_STANFORD,
)

import pdb

FILEMARKER_DIR = "/home/ksaab/Documents/hippo/datasets/eeg/file_markers"  # "/home/workspace/hippo/datasets/eeg/file_markers"
# sys.path.append(FILEMARKER_DIR)

NUM_SENSORS = 19


def computeSliceMatrix(
    edf_fn,
    channel_names,
    clip_idx,
    h5_fn=None,
    time_step_size=1,
    clip_len=60,
    stride=60,
    is_fft=False,
    use_cnn_features=False,
    dataset_name="TUH",
    min_sz_len=5,
):
    """
    Comvert entire EEG sequence into clips of length clip_len
    Args:
        edf_fn: edf/eeghdf file name, full path
        channel_names: list of channel names
        clip_idx: index of current clip/sliding window, int
        h5_fn: file name of resampled signal h5 file (full path)
        time_step_size: length of each time step, in seconds, int
        clip_len: sliding window size or EEG clip length, in seconds, int
        stride: stride size, by how many seconds the sliding window moves, int
        is_fft: whether to perform Fourier Transform on raw EEG data
        use_cnn_features: whether or not the input is CNN features
        dataset_name: name of the dataset, "TUH", "Stanford" or "LPCH"
        min_sz_len: minimum length to be considered as a seizure, in seconds
    Returns:
        eeg_clip: EEG clip, shape (clip_len//time_step_size, num_channels, time_step_size*freq)
        seizure_labels: per-time-step seizure labels, shape (clip_len//time_step_size,)
        is_seizure: overall label, 1 if at least one seizure within this clip, otherwise 0
    """
    if dataset_name.lower() not in ["tuh", "stanford", "lpch"]:
        raise NotImplementedError

    physical_clip_len = int(FREQUENCY * clip_len)

    start_window = clip_idx * FREQUENCY * stride

    if not use_cnn_features:
        with h5py.File(h5_fn, "r") as f:
            signal_array = f["resampled_signal"][()]
            resampled_freq = f["resample_freq"][()]
            assert resampled_freq == FREQUENCY

        # (num_channels, physical_clip_len)
        end_window = np.minimum(
            signal_array.shape[-1], start_window + physical_clip_len
        )
        curr_slc = signal_array[
            :, start_window:end_window
        ]  # (num_channels, FREQ*clip_len)
        physical_time_step_size = int(FREQUENCY * time_step_size)

        start_time_step = 0
        time_steps = []
        while start_time_step <= curr_slc.shape[1] - physical_time_step_size:
            end_time_step = start_time_step + physical_time_step_size
            # (num_channels, physical_time_step_size)
            curr_time_step = curr_slc[:, start_time_step:end_time_step]
            if is_fft:
                # curr_time_step, _ = computeFFT(curr_time_step, n=FREQUENCY)
                curr_time_step, _ = computeFFT(
                    curr_time_step, n=curr_time_step.shape[-1]
                )

            time_steps.append(curr_time_step)
            start_time_step = end_time_step

        eeg_clip = np.stack(time_steps, axis=0)
    elif use_cnn_features:
        with h5py.File(h5_fn, "r") as hf:
            cnn_features = hf["cnn_features"][
                ()
            ]  # (num_total_time_steps, cnn_feature_dim)
        # start_ts_idx = clip_idx * (clip_len // time_step_size)
        start_ts_idx = clip_idx * (stride // time_step_size)
        end_ts_idx = np.minimum(
            cnn_features.shape[0], start_ts_idx + clip_len // time_step_size
        )
        eeg_clip = cnn_features[
            start_ts_idx:end_ts_idx, :
        ]  # (num_time_steps, cnn_feature_dim)
        end_window = end_ts_idx * time_step_size * FREQUENCY

    # get seizure times, take min_sz_len into account
    if ".edf" in edf_fn:
        # TODO: Also enforce a min_sz_len for TUH annotations?
        seizure_times_raw = getSeizureTimes(edf_fn.split(".edf")[0])
        seizure_times = [
            sz_time
            for sz_time in seizure_times_raw
            if (sz_time[1] - sz_time[0]) > min_sz_len
        ]
    else:
        raise NotImplementedError

    # get per-time-step seizure labels
    num_time_steps = eeg_clip.shape[0]
    seizure_labels = np.zeros((num_time_steps)).astype(int)
    is_seizure = 0
    for t in seizure_times:
        start_t = int(t[0] * FREQUENCY)
        end_t = int(t[1] * FREQUENCY)
        if not ((end_window < start_t) or (start_window > end_t)):
            is_seizure = 1

            start_t_sec = int(t[0])  # start of seizure in int seconds
            end_t_sec = int(t[1])  # end of seizure in int seconds

            # shift start_t_sec and end_t_sec so that they start at current clip
            start_t_sec = np.maximum(0, start_t_sec - int(start_window / FREQUENCY))
            end_t_sec = np.minimum(clip_len, end_t_sec - int(start_window / FREQUENCY))
            # print("start_t_sec: {}; end_t_sec: {}".format(start_t_sec, end_t_sec))

            # time step size may not be 1-sec
            start_time_step = int(np.floor(start_t_sec / time_step_size))
            end_time_step = int(np.ceil(end_t_sec / time_step_size))

            seizure_labels[start_time_step:end_time_step] = 1

    return eeg_clip, seizure_labels, is_seizure


class SeizureDataset(Dataset):
    def __init__(
        self,
        input_dir,
        raw_data_dir,
        clip_len=60,
        time_step_size=1,
        stride=60,
        standardize=False,
        scaler=None,
        split="train",
        balance_train=False,
        data_augment=False,
        use_fft=False,
        use_cnn_features=False,
        padding=False,
        padding_val=0,
        dataset_name="TUH",
        min_sz_len=5,
        cv_fold=None,
    ):
        """
        Args:
            input_dir: dir to resampled signals h5 files
            raw_data_dir: dir to raw edf files
            clip_len: EEG clip length, in seconds, int
            time_step_size: how many seconds is one time step? int
            stride: how many seconds between subsequent clips, int
            standardize: if True, will z-normalize wrt train set mean and std
            scaler: scaler object for standardization
            split: train, dev or test
            data_augment: if True, perform random scaling & random reflecting along midline
            use_fft: if True, will perform Fourier transform to raw EEG signals
            padding: if True, will pad to clip_len//time_step_size
            padding_val: int, value used for padding to clip_len
            dataset_name: name of the dataset, "TUH", "Stanford" or "LPCH"
            min_sz_len: minimum length to be considered as a seizure, in seconds
            cv_fold: cv fold, int
        """
        if standardize and (scaler is None):
            raise ValueError("To standardize, please provide scaler.")
        if use_fft and use_cnn_features:
            raise ValueError("Either use_fft or use_cnn_features.")

        self.input_dir = input_dir
        self.raw_data_dir = raw_data_dir
        self.time_step_size = time_step_size
        self.clip_len = clip_len
        # only use smaller strides for train set
        self.stride = stride if split == "train" else clip_len
        self.standardize = standardize
        self.scaler = scaler
        self.split = split
        self.balance_train = balance_train
        self.data_augment = data_augment
        self.use_fft = use_fft
        self.use_cnn_features = use_cnn_features
        self.padding_val = padding_val
        self.padding = padding
        self.dataset_name = dataset_name
        self.min_sz_len = min_sz_len
        self.cv_fold = cv_fold

        # get full paths to all raw edf files
        self.edf_files = []
        if dataset_name.lower() == "tuh":
            for path, subdirs, files in os.walk(raw_data_dir):
                for name in files:
                    if ".edf" in name:
                        self.edf_files.append(os.path.join(path, name))

        # get number of clips for each eeg file
        self.file_tuples = []  # list of tuples (h5_file_name, clip_index)
        if split == "train" and balance_train:
            if padding:
                file_marker = os.path.join(
                    os.path.join(FILEMARKER_DIR, "variable_length"),
                    split
                    + "_cliplen"
                    + str(clip_len)
                    + "_stride"
                    + str(self.stride)
                    + "_timestep"
                    + str(time_step_size)
                    + "_balanced.txt",
                )
            else:
                file_marker = os.path.join(
                    FILEMARKER_DIR,
                    split
                    + "_cliplen"
                    + str(clip_len)
                    + "_stride"
                    + str(self.stride)
                    + "_balanced.txt",
                )
        else:
            if padding:
                file_marker = os.path.join(
                    os.path.join(FILEMARKER_DIR, "variable_length"),
                    split
                    + "_cliplen"
                    + str(clip_len)
                    + "_stride"
                    + str(self.stride)
                    + "_timestep"
                    + str(time_step_size)
                    + ".txt",
                )
            else:
                file_marker = os.path.join(
                    FILEMARKER_DIR,
                    split
                    + "_cliplen"
                    + str(clip_len)
                    + "_stride"
                    + str(self.stride)
                    + ".txt",
                )

        print(file_marker)
        with open(file_marker, "r") as f:
            file_str = f.readlines()
        file_tuples = [curr_str.strip("\n").split(",") for curr_str in file_str]

        # filter out signals shorter than time_step_size
        self.file_tuples = []
        if self.use_cnn_features:
            for tup in file_tuples:
                edf_fn = tup[0]
                h5_fn = os.path.join(input_dir, edf_fn.split(".edf")[0] + ".h5")
                with h5py.File(h5_fn, "r") as hf:
                    feature = hf["cnn_features"][()]
                if feature.shape[0] > 0:
                    self.file_tuples.append(tup)
        else:
            self.file_tuples = file_tuples

        self.size = len(self.file_tuples)

        # Get sensor ids
        self.sensor_ids = [x.split(" ")[-1] for x in INCLUDED_CHANNELS]

    def __len__(self):
        return self.size

    def _random_reflect(self, EEG_seq):
        """
        Randomly reflect along midline
        Args:
            EEG_seq: shape (seq_length, num_nodes, feature_dim)
        Returns:
            EEG_seq_reflect: augmented signal, same shape as EEG_seq
            swap_pairs: swapped EEG channel pairs
        """
        swap_pairs = get_swap_pairs(INCLUDED_CHANNELS)
        EEG_seq_reflect = EEG_seq.copy()
        if np.random.choice([True, False]):
            for pair in swap_pairs:
                EEG_seq_reflect[:, [pair[0], pair[1]], :] = EEG_seq[
                    :, [pair[1], pair[0]], :
                ]
        else:
            swap_pairs = None
        return EEG_seq_reflect, swap_pairs

    def _random_scale(self, EEG_seq):
        """
        Scale EEG signals by a random number between 0.8 and 1.2
        Args:
            EEG_seq: shape (seq_length, num_nodes, feature_dim)
        Returns:
            augmented signals, same shape as input
        """
        scale_factor = np.random.uniform(0.8, 1.2)
        if self.use_fft:
            EEG_seq += np.log(scale_factor)
        else:
            EEG_seq *= scale_factor
        return EEG_seq

    def __getitem__(self, idx):
        """
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            a tuple of (feature, label, writeout_file_name)
        """
        edf_fn, clip_idx, _ = self.file_tuples[idx]
        edf_file = [file for file in self.edf_files if edf_fn in file]
        assert len(edf_file) == 1
        edf_file = edf_file[0]
        clip_idx = int(clip_idx)
        writeout_fn = edf_fn + "_" + str(clip_idx)
        h5_fn = os.path.join(self.input_dir, edf_fn.split(".edf")[0] + ".h5")

        # get the clip
        eeg_clip, seizure_labels, is_seizure = computeSliceMatrix(
            h5_fn=h5_fn,
            edf_fn=edf_file,
            channel_names=INCLUDED_CHANNELS,
            clip_idx=clip_idx,
            time_step_size=self.time_step_size,
            clip_len=self.clip_len,
            stride=self.stride,
            is_fft=self.use_fft,
            use_cnn_features=self.use_cnn_features,
            dataset_name=self.dataset_name,
            min_sz_len=self.min_sz_len,
        )

        # data augmentation
        if self.data_augment and (
            not self.use_cnn_features
        ):  # no augmentation with cnn features
            curr_feature, swap_nodes = self._random_reflect(eeg_clip)
            curr_feature = self._random_scale(curr_feature)
        else:
            swap_nodes = None
            curr_feature = eeg_clip

        # standardize wrt train mean and std
        if (
            self.standardize
            and (not self.use_cnn_features)
            and (not self.use_hippo_proj)
        ):
            curr_feature = self.scaler.transform(curr_feature)

        # padding, will be useful for full signal later
        curr_len = curr_feature.shape[0]
        seq_len = np.minimum(curr_len, self.clip_len // self.time_step_size)
        if self.padding and (curr_len < (self.clip_len // self.time_step_size)):
            len_pad = self.clip_len // self.time_step_size - curr_len
            if not self.use_cnn_features:
                padded_feature = (
                    np.ones((len_pad, curr_feature.shape[1], curr_feature.shape[2]))
                    * self.padding_val
                )
            else:
                padded_feature = (
                    np.ones((len_pad, curr_feature.shape[1])) * self.padding_val
                )
            padded_feature = np.concatenate((curr_feature, padded_feature), axis=0)

            # for labels, pad with -1 ?
            padded_label = np.ones((len_pad)) * -1.0
            padded_label = np.concatenate((seizure_labels, padded_label), axis=0)
        else:
            padded_feature = curr_feature.copy()
            padded_label = seizure_labels.copy()

        # convert to tensors
        # (clip_len//time_step_size, num_channels, input_dim) or (clip_len//time_step_size, cnn_feature_dim)
        x = torch.FloatTensor(padded_feature)
        y = torch.FloatTensor(padded_label)  # (clip_len//time_step_size,)
        seq_len = torch.LongTensor([seq_len])  # (1,)

        # return (x, y, seq_len, writeout_fn)
        # return (x, y, seq_len, is_seizure)

        if self.use_fft:
            # reshape such that x is (L,19*100)
            x = x.view(x.shape[0], -1)
        else:
            # if raw, reshape such that x is (L,19) where L = clip_len*200
            x = x.transpose(1, 2)
            x = x.reshape(-1, x.shape[2])

        return (x, is_seizure)


def load_dataset(
    input_dir,
    raw_data_dir,
    train_batch_size=64,
    test_batch_size=64,
    clip_len=60,
    time_step_size=1,
    stride=60,
    standardize=False,
    num_workers=8,
    padding=False,
    padding_val=0.0,
    augmentation=False,
    balance_train=False,
    use_fft=False,
    use_cnn_features=False,
    eval_clip_len=None,
    dataset_name="TUH",
    min_sz_len=5,
    cv_fold=None,
    means_dir=None,
    stds_dir=None,
):
    """
    Args:
        input_dir: dir to preprocessed h5 file
        train_batch_size: int
        test_batch_size: int
        standardize: if True, will z-normalize wrt train set
        num_workers: int
        padding_val: value used for padding
        augmentation: if True, perform random augmentation of EEG seq
    Returns:
        dataloaders, datasets: dictionaries of train/dev/test dataloaders and datasets
    """

    if standardize:
        if (means_dir is not None) and (stds_dir is not None):
            means_file = means_dir
            stds_file = stds_dir
        else:
            raise ValueError
        with open(means_file, "rb") as f:
            means = pickle.load(f)
        with open(stds_file, "rb") as f:
            stds = pickle.load(f)

        scaler = StandardScaler(mean=means, std=stds)
    else:
        scaler = None

    dataloaders = {}
    if dataset_name.lower() == "tuh":
        splits = ["train", "dev", "test"]
    else:
        splits = ["train", "test"]
    for split in splits:
        if split == "train":
            data_augment = augmentation
        else:
            data_augment = False  # never do augmentation on dev/test sets

        # allows different clip length for evaluation
        if (eval_clip_len is not None) and (split != "train"):
            curr_clip_len = eval_clip_len
        else:
            curr_clip_len = clip_len
        dataset = SeizureDataset(
            input_dir=input_dir,
            raw_data_dir=raw_data_dir,
            time_step_size=time_step_size,
            clip_len=curr_clip_len,
            stride=stride,
            standardize=standardize,
            scaler=scaler,
            split=split,
            balance_train=balance_train,
            data_augment=data_augment,
            use_fft=use_fft,
            use_cnn_features=use_cnn_features,
            padding=padding,
            padding_val=padding_val,
            dataset_name=dataset_name,
            min_sz_len=min_sz_len,
            cv_fold=cv_fold,
        )

        if split == "train":
            shuffle = True
            batch_size = train_batch_size
        else:
            shuffle = False
            batch_size = test_batch_size

        loader = DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        dataloaders[split] = loader

    return dataloaders


if __name__ == "__main__":

    input_dir = "/media/nvme_data/siyitang/TUH_eeg_seq_v1.5.2/resampled_signal"
    raw_data_dir = "/media/nvme_data/TUH/v1.5.2/edf"
    dataloaders = load_dataset(
        input_dir,
        raw_data_dir,
        clip_len=60,
        stride=60,
        use_fft=True,
        train_batch_size=1,
    )
    dl_train = dataloaders["train"]

    for batch in dl_train:
        pdb.set_trace()
