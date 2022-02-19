"""Google speech commands dataset."""
__author__ = 'Yuan Xu'
"""With modifications by Karan Goel to support training an SC09 classifier."""

import os
import numpy as np

import librosa

from torch.utils.data import Dataset

__all__ = [ 'CLASSES', 'SpeechCommandsDataset', 'BackgroundNoiseDataset' ]

CLASSES = 'zero, one, two, three, four, five, six, seven, eight, nine'.split(', ')

class SpeechCommandsDataset(Dataset):
    """Google speech commands dataset. Only 'yes', 'no', 'up', 'down', 'left',
    'right', 'on', 'off', 'stop' and 'go' are treated as known classes.
    All other classes are used as 'unknown' samples.
    See for more information: https://www.kaggle.com/c/tensorflow-speech-recognition-challenge
    """

    def __init__(self, folder, transform=None, classes=CLASSES):
        all_classes = classes
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        data = []
        for c in all_classes:
            d = os.path.join(folder, c)
            target = class_to_idx[c]
            for f in os.listdir(d):
                path = os.path.join(d, f)
                data.append((path, target))

        self.classes = classes
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        data = {'path': path, 'target': target}

        if self.transform is not None:
            data = self.transform(data)

        return data

    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        nclasses = len(self.classes)
        count = np.zeros(nclasses)
        for item in self.data:
            count[item[1]] += 1

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[item[1]]
        return weight

class BackgroundNoiseDataset(Dataset):
    """Dataset for silence / background noise."""

    def __init__(self, folder, transform=None, sample_rate=16000, sample_length=1):
        audio_files = [d for d in os.listdir(folder) if os.path.isfile(os.path.join(folder, d)) and d.endswith('.wav')]
        samples = []
        for f in audio_files:
            path = os.path.join(folder, f)
            s, sr = librosa.load(path, sample_rate)
            samples.append(s)

        samples = np.hstack(samples)
        c = int(sample_rate * sample_length)
        r = len(samples) // c
        self.samples = samples[:r*c].reshape(-1, c)
        self.sample_rate = sample_rate
        self.classes = CLASSES
        self.transform = transform
        self.path = folder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = {'samples': self.samples[index], 'sample_rate': self.sample_rate, 'target': 1, 'path': self.path}

        if self.transform is not None:
            data = self.transform(data)

        return data
