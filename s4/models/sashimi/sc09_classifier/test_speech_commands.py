"""
Taken from https://github.com/tugstugi/pytorch-speech-commands and modified
by Karan Goel.
"""

import argparse
import os

import torch
import numpy as np

from functools import reduce
from natsort import natsorted
from scipy import linalg
from scipy.stats import norm, entropy
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from tqdm.auto import tqdm

from transforms import FixAudioLength, LoadAudio, ToMelSpectrogram, ToTensor


def fid(feat_data, feat_gen):
    """
    Calculate Frechet Inception Distance
    """
    # Means
    mu_data = np.mean(feat_data, axis=0)
    mu_gen = np.mean(feat_gen, axis=0)

    # Covariances
    try:
        sigma_data = np.cov(feat_data, rowvar=False)
        sigma_gen = np.cov(feat_gen, rowvar=False)

        covmean, _ = linalg.sqrtm(sigma_data.dot(sigma_gen), disp=False)
        if not np.isfinite(covmean).all():
            print("fid calculation produces singular product; adding perturbation to diagonal of cov estimates")
            offset = np.eye(sigma_data.shape[0]) * 1e-4
            covmean, _ = linalg.sqrtm((sigma_data + offset).dot(sigma_gen + offset))

        # Now calculate the FID
        fid_value = np.sum(np.square(mu_gen - mu_data)) + np.trace(sigma_gen + sigma_data - 2*covmean)

        return fid_value
    except ValueError:
        return np.inf

def inception_score(probs_gen):
    """
    Calculate Inception Score
    """
    # Set seed
    np.random.seed(0)

    # Shuffle probs_gen
    probs_gen = probs_gen[np.random.permutation(len(probs_gen))]

    # Split probs_gen into two halves
    probs_gen_1 = probs_gen[:len(probs_gen)//2]
    probs_gen_2 = probs_gen[len(probs_gen)//2:]

    # Calculate average label distribution for split 2
    mean_2 = np.mean(probs_gen_2, axis=0)

    # Compute the mean kl-divergence between the probability distributions
    # of the generated and average label distributions
    kl = entropy(probs_gen_1, np.repeat(mean_2[None, :], len(probs_gen_1), axis=0)).mean()

    # Compute the expected score
    is_score = np.exp(kl)

    return is_score

def modified_inception_score(probs_gen, n=10000):
    """
    Calculate Modified Inception Score
    """
    # Set seed
    np.random.seed(0)

    n_samples = len(probs_gen)

    all_kls = []
    for i in range(n):
        # Sample two prob vectors
        indices = np.random.choice(np.arange(n_samples), size=2, replace=True)
        probs_gen_1 = probs_gen[indices[0]]
        probs_gen_2 = probs_gen[indices[1]]

        # Calculate their KL
        kl = entropy(probs_gen_1, probs_gen_2)

        all_kls.append(kl)

    # Compute the score
    mis_score = np.exp(np.mean(all_kls))

    return mis_score

def am_score(probs_data, probs_gen):
    """
    Calculate AM Score
    """
    mean_data = np.mean(probs_data, axis=0)
    mean_gen = np.mean(probs_gen, axis=0)
    entropy_gen = np.mean(entropy(probs_gen, axis=1))
    am_score = entropy(mean_data, mean_gen) + entropy_gen

    return am_score


def two_proportions_z_test(p1, n1, p2, n2, significance_level, z_threshold=None):
    # Taken from https://github.com/eitanrich/gans-n-gmms/blob/master/utils/ndb.py
    # Per http://stattrek.com/hypothesis-test/difference-in-proportions.aspx
    # See also http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/binotest.htm
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1/n1 + 1/n2))
    z = (p1 - p2) / se
    # Allow defining a threshold in terms as Z (difference relative to the SE) rather than in p-values.
    if z_threshold is not None:
        return abs(z) > z_threshold
    p_values = 2.0 * norm.cdf(-1.0 * np.abs(z))    # Two-tailed test
    return p_values < significance_level


def ndb_score(feat_data, feat_gen):
    # Run K-Means cluster on feat_data with K=50
    kmeans = KMeans(n_clusters=50, random_state=0).fit(feat_data)

    # Get cluster labels for feat_data and feat_gen
    labels_data = kmeans.predict(feat_data)
    labels_gen = kmeans.predict(feat_gen)

    # Calculate number of data points in each cluster using np.unique
    counts_data = np.unique(labels_data, return_counts=True)[1]
    counts_gen = np.zeros_like(counts_data)
    values, counts = np.unique(labels_gen, return_counts=True)
    counts_gen[values] = counts

    # Calculate proportion of data points in each cluster
    prop_data = counts_data / len(labels_data)
    prop_gen = counts_gen / len(labels_gen)

    # Calculate number of bins with statistically different proportions
    different_bins = two_proportions_z_test(prop_data, len(labels_data), prop_gen, len(labels_gen), 0.05)
    ndb = np.count_nonzero(different_bins)

    return ndb/50.



def _nested_getattr(obj, attr, *args):
    """Get a nested property from an object.
    Example:
    ```
        model = ...
        weights = _nested_getattr(model, "layer4.weights")
    ```
    """
    return reduce(lambda o, a: getattr(o, a, *args), [obj] + attr.split("."))


class ActivationExtractor:
    """Class for extracting activations of a targeted intermediate layer."""

    def __init__(self):
        self.input = None
        self.output = None

    def add_hook(self, module, input, output):
        self.input = input
        self.output = output

class ActivationOp:
    def __init__(
        self,
        model: torch.nn.Module,
        target_module: str,
    ):
        self.model = model
        self.target_module = target_module

        try:
            target_module = _nested_getattr(model, target_module)
        except torch.nn.modules.module.ModuleAttributeError:
            raise ValueError(f"`model` does not have a submodule {target_module}")

        self.extractor = ActivationExtractor()
        target_module.register_forward_hook(self.extractor.add_hook)



CLASSES = 'zero, one, two, three, four, five, six, seven, eight, nine'.split(', ')

class SpeechCommandsDataset(Dataset):

    def __init__(self, folder, transform=None, classes=CLASSES, samples=False):

        self.classes = classes
        self.transform = transform

        if not samples:
            class_to_idx = {classes[i]: i for i in range(len(classes))}

            data = []
            for c in classes:
                d = os.path.join(folder, c)
                target = class_to_idx[c]
                for f in natsorted(os.listdir(d)):
                    if f.endswith(".wav"):
                        path = os.path.join(d, f)
                        data.append((path, target))

        else:
            data = []
            for f in natsorted(os.listdir(folder)):
                if f.endswith(".wav"):
                    path = os.path.join(folder, f)
                    data.append((path, -1))

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        data = {'path': path, 'target': target}

        if self.transform is not None:
            data = self.transform(data)

        return data


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train-dataset-dir", type=str, default='datasets/speech_commands/train', help='path of test dataset')
parser.add_argument("--test-dataset-dir", type=str, default='datasets/speech_commands/test', help='path of test dataset')
parser.add_argument("--sample-dir", type=str, default='datasets/speech_commands/test', help='path of test dataset')
parser.add_argument("--batch-size", type=int, default=128, help='batch size')
parser.add_argument("--dataload-workers-nums", type=int, default=4, help='number of workers for dataloader')
parser.add_argument("--input", choices=['mel32'], default='mel32', help='input of NN')
parser.add_argument("--threshold", action='store_true', help='tune thresholds to reject samples')
parser.add_argument("--save-probs", action='store_true', help='save classifier probs on samples')
parser.add_argument("model", help='a pretrained neural network model')
args = parser.parse_args()

model = torch.load(args.model)
model.float()

use_gpu = torch.cuda.is_available()
if use_gpu:
    torch.backends.cudnn.benchmark = True
    model.cuda()

n_mels = 32
if args.input == 'mel40':
    n_mels = 40

feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
transform = Compose([LoadAudio(), FixAudioLength(), feature_transform])

train_dataset = SpeechCommandsDataset(args.train_dataset_dir, transform)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=None,
    pin_memory=use_gpu,
    num_workers=args.dataload_workers_nums,
    drop_last=False,
    shuffle=False,
)

test_dataset = SpeechCommandsDataset(args.test_dataset_dir, transform)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    sampler=None,
    pin_memory=use_gpu,
    num_workers=args.dataload_workers_nums,
    drop_last=False,
    shuffle=False,
)

samples_dataset = SpeechCommandsDataset(
    args.sample_dir,
    transform,
    samples=False if args.sample_dir.rstrip("/").endswith('test') or args.sample_dir.rstrip("/").endswith('train') else True,
)
samples_dataloader = DataLoader(
    samples_dataset,
    batch_size=args.batch_size,
    sampler=None,
    pin_memory=use_gpu,
    num_workers=args.dataload_workers_nums,
    drop_last=False,
    shuffle=False,
)

@torch.no_grad()
def test(dataloader):
    model.eval()  # Set model to evaluate mode

    extractor = ActivationExtractor()
    module = model.module.classifier
    module.register_forward_hook(extractor.add_hook)

    correct = 0
    total = 0

    probs = []
    activations = []

    pbar = tqdm(dataloader, unit="audios", unit_scale=dataloader.batch_size)
    for batch in pbar:
        inputs = batch['input']
        inputs = inputs.unsqueeze(1)
        targets = batch['target']

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # forward
        outputs = model(inputs)
        outputs = torch.nn.functional.softmax(outputs, dim=1)

        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)

        probs.append(outputs.cpu().numpy())
        activations.append(extractor.input[0].cpu().numpy())

    probs = np.concatenate(probs)
    activations = np.concatenate(activations)
    accuracy = correct/total
    print("accuracy: %f%%" % (100*accuracy))
    return probs, activations

# Run test if train_probs and train_activations are not on disk
if not os.path.exists('cache/train_probs.npy') or not os.path.exists('cache/train_activations.npy'):
    train_probs, train_activations = test(train_dataloader)
    np.save('cache/train_probs.npy', train_probs)
    np.save('cache/train_activations.npy', train_activations)
else:
    train_probs = np.load('cache/train_probs.npy')
    train_activations = np.load('cache/train_activations.npy')

# Same for test
if not os.path.exists('cache/test_probs.npy') or not os.path.exists('cache/test_activations.npy'):
    test_probs, test_activations = test(test_dataloader)
    os.makedirs('cache', exist_ok=True)
    np.save('cache/test_probs.npy', test_probs)
    np.save('cache/test_activations.npy', test_activations)
else:
    test_probs = np.load('cache/test_probs.npy')
    test_activations = np.load('cache/test_activations.npy')

###############################################################################
# Calculate all scores
###############################################################################

print("------------------")
print("Train Set Scores")
print("------------------")
print('\tFID:', fid(train_activations, train_activations))
print('\tInception:', inception_score(train_probs))
print('\tM Inception:', modified_inception_score(train_probs))
print('\tAM:', am_score(train_probs, train_probs))
# print('\tNDB:', 0.)


print("------------------")
print("Test Set Scores")
print("------------------")
print('\tFID:', fid(train_activations, test_activations))
print('\tInception:', inception_score(test_probs))
print('\tM Inception:', modified_inception_score(test_probs))
print('\tAM:', am_score(train_probs, test_probs))
# print('\tNDB:', ndb_score(train_activations, test_activations))

# Train -> Samples
samples_probs, samples_activations = test(samples_dataloader)


if args.threshold:

    n_val = len(samples_probs) // 2
    n_test = len(samples_probs) // 2
    print("Tuning thresholds using IS: using %d samples for tuning and %d for calculating metrics" % (n_val, n_test))

    # Split into two parts, one for tuning thresholds and one for calculating metrics
    val_indices = sorted(np.random.choice(len(samples_probs), size=n_val, replace=False))
    test_indices = sorted(np.array(list(set(range(len(samples_probs))) - set(val_indices))))

    samples_probs_val = samples_probs[val_indices]
    samples_probs_test = samples_probs[test_indices]

    samples_activations_val = samples_activations[val_indices]
    samples_activations_test = samples_activations[test_indices]

    # Iterate over all thresholds
    all_scores = {'fid': {}, 'is': {}}
    for lower_threshold in tqdm(np.arange(0., 0.5, 0.1)):
        for upper_threshold in tqdm(np.arange(0.6, 1.0, 0.05)):
            all_scores['is'][(lower_threshold, upper_threshold)] = inception_score(samples_probs_val[int(lower_threshold * n_val):int(upper_threshold * n_val)])

    # Find the best score and calculate all metrics on the test set
    best_value = 0.
    best_thresholds_is = None
    for threshold, value in all_scores['is'].items():
        if value > best_value:
            best_thresholds_is = threshold

    print("------------------")
    print("Tuned Thresholds")
    print("------------------")
    print("\tBest thresholds (by IS tuning):", best_thresholds_is)
    print("\tBest IS score (on dev set):", all_scores['is'][best_thresholds_is])

    sample_activations_test_inception = samples_activations_test[int(best_thresholds_is[0] * n_test):int(best_thresholds_is[1] * n_test)]
    sample_probs_test_inception = samples_probs_test[int(best_thresholds_is[0] * n_test):int(best_thresholds_is[1] * n_test)]

    print("------------------")
    print("Sample Scores (with Tuned Thresholds)")
    print("------------------")
    print('\tFID:', fid(train_activations, sample_activations_test_inception))
    print('\tInception:', inception_score(sample_probs_test_inception))
    print('\tM Inception:', modified_inception_score(sample_probs_test_inception))
    print('\tAM:', am_score(train_probs, sample_probs_test_inception))
    # print('\tNDB:', ndb_score(train_activations, sample_activations_test_inception))

else:
    print("------------------")
    print("Sample Scores (no Threshold Tuning)")
    print("------------------")
    print('\tFID:', fid(train_activations, samples_activations))
    print('\tInception:', inception_score(samples_probs))
    print('\tM Inception:', modified_inception_score(samples_probs))
    print('\tAM:', am_score(train_probs, samples_probs))
    # print('\tNDB:', ndb_score(train_activations, samples_activations))

if args.save_probs:
    filename = args.sample_dir.rstrip("/").split("/")[-1]
    np.save(f'cache/{filename}-resnext-probs.npy', samples_probs)

# Info about probs
# print(np.unique(np.argmax(samples_probs, axis=1), return_counts=True))
# print(samples_probs[np.arange(samples_probs.shape[0]), np.argmax(samples_probs, axis=1)])
# print(samples_probs)
