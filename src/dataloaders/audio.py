import torch
import torchaudio
import numpy as np
import os

from os import listdir
from os.path import join

def minmax_scale(tensor, range_min=0, range_max=1):
    """
    Min-max scaling to [0, 1].
    """
    min_val = torch.amin(tensor, dim=(1, 2), keepdim=True)
    max_val = torch.amax(tensor, dim=(1, 2), keepdim=True)
    return range_min + (range_max - range_min) * (tensor - min_val) / (max_val - min_val + 1e-6)

def quantize(samples, bits=8, epsilon=0.01):
    """
    Linearly quantize a signal in [0, 1] to a signal in [0, q_levels - 1].
    """
    q_levels = 1 << bits
    samples *= q_levels - epsilon
    samples += epsilon / 2
    return samples.long() 

def dequantize(samples, bits=8):
    """
    Dequantize a signal in [0, q_levels - 1].
    """
    q_levels = 1 << bits
    return samples.float() / (q_levels / 2) - 1

def mu_law_encode(audio, bits=8):
    """
    Perform mu-law companding transformation.
    """
    mu = torch.tensor((1 << bits) - 1)

    # Audio must be min-max scaled between -1 and 1
    audio = minmax_scale(audio, range_min=-1, range_max=1)

    # Perform mu-law companding transformation.
    numerator = torch.log1p(mu * torch.abs(audio + 1e-8))
    denominator = torch.log1p(mu)
    encoded = torch.sign(audio) * (numerator / denominator)

    # Shift signal to [0, 1]
    encoded = (encoded + 1) / 2

    # Quantize signal to the specified number of levels.
    return quantize(encoded, bits=bits)

def mu_law_decode(encoded, bits=8):
    """
    Perform inverse mu-law transformation.
    """
    mu = (1 << bits) - 1
    # Invert the quantization
    x = dequantize(encoded, bits=bits)

    # Invert the mu-law transformation
    x = torch.sign(x) * ((1 + mu)**(torch.abs(x)) - 1) / mu

    # Returned values in range [-1, 1]
    return x

def linear_encode(samples, bits=8):
    """
    Perform scaling and linear quantization.
    """
    samples = samples.clone()
    samples = minmax_scale(samples)
    return quantize(samples, bits=bits)

def linear_decode(samples, bits=8):
    """
    Invert the linear quantization.
    """
    return dequantize(samples, bits=bits)

def q_zero(bits=8):
    """
    The quantized level of the 0.0 value.
    """
    return 1 << (bits - 1)


class AbstractAudioDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        bits=8,
        sample_len=None, 
        quantization='linear',
        return_type='autoregressive',
        drop_last=True,
        target_sr=None,
        context_len=None,
        pad_len=None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.bits = bits
        self.sample_len = sample_len
        self.quantization = quantization
        self.return_type = return_type
        self.drop_last = drop_last
        self.target_sr = target_sr
        self.zero = q_zero(bits)
        self.context_len = context_len
        self.pad_len = pad_len

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.file_names = NotImplementedError("Must be assigned in setup().")
        self.transforms = {}

        self.setup()
        self.create_quantizer(self.quantization)
        self.create_examples(self.sample_len)


    def setup(self):
        return NotImplementedError("Must assign a list of filepaths to self.file_names.")
    
    def __getitem__(self, index):
        # Load signal
        if self.sample_len is not None:
            file_name, start_frame, num_frames = self.examples[index]
            seq, sr = torchaudio.load(file_name, frame_offset=start_frame, num_frames=num_frames)
        else:
            seq, sr = torchaudio.load(self.examples[index])

        # Average non-mono signals across channels
        if seq.shape[0] > 1:
            seq = seq.mean(dim=0, keepdim=True)

        # Resample signal if required
        if self.target_sr is not None and sr != self.target_sr:
            if sr not in self.transforms:
                self.transforms[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            seq = self.transforms[sr](seq)

        # Transpose the signal to get (L, 1)
        seq = seq.transpose(0, 1)
        
        # Unsqueeze to (1, L, 1)
        seq = seq.unsqueeze(0)

        # Quantized signal
        qseq = self.quantizer(seq, self.bits)

        # Squeeze back to (L, 1)
        qseq = qseq.squeeze(0)

        # Return the signal
        if self.return_type == 'autoregressive':
            # Autoregressive training
            # x is [0,  qseq[0], qseq[1], ..., qseq[-2]]
            # y is [qseq[0], qseq[1], ..., qseq[-1]]
            y = qseq
            x = torch.roll(qseq, 1, 0) # Roll the signal 1 step
            x[0] = self.zero # Fill the first element with q_0
            x = x.squeeze(1) # Squeeze to (L, )
            if self.context_len is not None:
                y = y[self.context_len:] # Trim the signal
            if self.pad_len is not None:
                x = torch.cat((torch.zeros(self.pad_len, dtype=torch.long) + self.zero, x)) # Pad the signal
            return x, y
        elif self.return_type is None:
            return qseq
        else:
            raise NotImplementedError(f'Invalid return type {self.return_type}')

    def __len__(self):
        return len(self.examples)

    def create_examples(self, sample_len: int):
        # Get metadata for all files
        self.metadata = [
            torchaudio.info(file_name) for file_name in self.file_names
        ]

        if sample_len is not None:
            # Reorganize files into a flat list of (file_name, start_frame) pairs 
            # so that consecutive items are separated by sample_len
            self.examples = []
            for file_name, metadata in zip(self.file_names, self.metadata):
                # Update the sample_len if resampling to target_sr is required
                # This is because the resampling will change the length of the signal 
                # so we need to adjust the sample_len accordingly (e.g. if downsampling
                # the sample_len will need to be increased)
                sample_len_i = sample_len
                if self.target_sr is not None and metadata.sample_rate != self.target_sr:
                    sample_len_i = int(sample_len * metadata.sample_rate / self.target_sr)

                margin = metadata.num_frames % sample_len_i
                for start_frame in range(0, metadata.num_frames - margin, sample_len_i):
                    self.examples.append((file_name, start_frame, sample_len_i))
                
                if margin > 0 and not self.drop_last:
                    # Last (leftover) example is shorter than sample_len, and equal to the margin 
                    # (must be padded in collate_fn)
                    self.examples.append((file_name, metadata.num_frames - margin, margin))
        else:
            self.examples = self.file_names

    def create_quantizer(self, quantization: str):
        if quantization == 'linear':
            self.quantizer = linear_encode
            self.dequantizer = linear_decode
        elif quantization == 'mu-law':
            self.quantizer = mu_law_encode
            self.dequantizer = mu_law_decode
        else:
            raise ValueError('Invalid quantization type')


class QuantizedAudioDataset(AbstractAudioDataset):
    """
    Adapted from https://github.com/deepsound-project/samplernn-pytorch/blob/master/dataset.py
    """

    def __init__(
        self, 
        path, 
        bits=8, 
        ratio_min=0,
        ratio_max=1,
        sample_len=None,
        quantization='linear', # [linear, mu-law]
        return_type='autoregressive', # [autoregressive, None]
        drop_last=False,
        target_sr=None,
        context_len=None,
        pad_len=None,
        **kwargs,
    ):
        super().__init__(
            bits=bits,
            sample_len=sample_len,
            quantization=quantization,
            return_type=return_type,
            drop_last=drop_last,
            target_sr=target_sr,
            path=path,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            context_len=context_len,
            pad_len=pad_len,
            **kwargs,
        )

    def setup(self):
        from natsort import natsorted
        file_names = natsorted(
            [join(self.path, file_name) for file_name in listdir(self.path)]
        )
        self.file_names = file_names[
            int(self.ratio_min * len(file_names)) : int(self.ratio_max * len(file_names))
        ]

class SpeechCommands09(AbstractAudioDataset):

    CLASSES = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    ]

    CLASS_TO_IDX = dict(zip(CLASSES, range(len(CLASSES))))

    def __init__(
        self, 
        path, 
        bits=8, 
        split='train',
        sample_len=16000,
        quantization='linear', # [linear, mu-law]
        return_type='autoregressive', # [autoregressive, None]
        drop_last=False,
        target_sr=None,
        dequantize=False,
        pad_len=None,
        **kwargs,
    ):
        super().__init__(
            bits=bits,
            sample_len=sample_len,
            quantization=quantization,
            return_type=return_type,
            split=split,
            drop_last=drop_last,
            target_sr=target_sr,
            path=path,
            dequantize=dequantize,
            pad_len=pad_len,
            **kwargs,
        )

    def setup(self):
        with open(join(self.path, 'validation_list.txt')) as f:
            validation_files = set([line.rstrip() for line in f.readlines()])
        
        with open(join(self.path, 'testing_list.txt')) as f:
            test_files = set([line.rstrip() for line in f.readlines()])

        # Get all files in the paths named after CLASSES
        self.file_names = []
        for class_name in self.CLASSES:
            self.file_names += [
                (class_name, file_name)
                for file_name in listdir(join(self.path, class_name))
                if file_name.endswith('.wav')
            ]
        
        # Keep files based on the split
        if self.split == 'train':
            self.file_names = [
                join(self.path, class_name, file_name) 
                for class_name, file_name in self.file_names 
                if join(class_name, file_name) not in validation_files
                and join(class_name, file_name) not in test_files
            ]
        elif self.split == 'validation':
            self.file_names = [
                join(self.path, class_name, file_name)
                for class_name, file_name in self.file_names
                if join(class_name, file_name) in validation_files
            ]
        elif self.split == 'test':
            self.file_names = [
                join(self.path, class_name, file_name)
                for class_name, file_name in self.file_names
                if join(class_name, file_name) in test_files
            ]

    def __getitem__(self, index):
        item = super().__getitem__(index)
        x, y, *z = item
        if self.dequantize:
            x = self.dequantizer(x).unsqueeze(1)
        return x, y, *z
