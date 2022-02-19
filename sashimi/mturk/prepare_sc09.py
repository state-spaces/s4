import argparse
import numpy as np
import os
import shutil

from natsort import natsorted

digits = 'zero one two three four five six seven eight nine'.split()

def move_files(src_dir, src_files, target_dir, indices, discard=False):
    os.makedirs(target_dir, exist_ok=True)
    for i, digit in enumerate(digits):
        try:
            os.mkdir(target_dir + f'{digit}')
        except FileExistsError:
            raise FileExistsError(f'{target_dir}/{digit} already exists, please delete it before running this script.')
        for index in indices[i]:
            if not discard:
                shutil.copy(f'{src_dir}/{src_files[index]}', f'{target_dir}/{digit}/{src_files[index]}')
            else:
                shutil.copy(f'{src_dir}/{src_files[index]}', f'{target_dir}/{digit}/{src_files[index].split("/")[-1]}')

def standardize_filenames(target_dir):
    i = 0
    for digit in digits:
        if not os.path.exists(f'{target_dir}/{digit}/'): continue
        for f in natsorted(os.listdir(f'{target_dir}/{digit}/')):
            if f.endswith('.wav'):
                shutil.move(f'{target_dir}/{digit}/{f}', f'{target_dir}/{i}.wav')
                i += 1
        shutil.rmtree(f'{target_dir}/{digit}/')

def grab_indices(probs, samples_per_class=50):
    confident_indices = {}
    random_indices = {}
    for digit in range(10):
        # Rows with prediction = digit
        rows = np.zeros_like(probs)
        rows[probs.argmax(1) == digit] = probs[probs.argmax(1) == digit]
        # Sort rows by confidence and take the last 50 indices
        confident_indices[digit] = np.argsort(rows[:, digit])[-samples_per_class:]
        # Take a random sample of 50 digits
        random_indices[digit] = np.random.choice(np.where(probs.argmax(1) == digit)[0], samples_per_class, replace=False)
    return confident_indices, random_indices

def prepare(
    method, 
    cache_dir = '../sc09_classifier/cache/',
    sample_dir='../samples/sc09/',
    target_dir='sc09/sc09-unconditional-exp-confident/',
    n_samples=2048,
    samples_per_class=50,
):
    
    # Load outputs of SC09 classifier (ResNeXt)
    # Example: `2048-sashimi-diffwave-small-500k-resnext-probs.npy` (where method is `sashimi-diffwave-small-500k`)
    probs = np.load(f'{cache_dir}/{n_samples}-{method}-resnext-probs.npy')
    
    # List all .wav sample files in the method directory
    # Example: all .wav files in `../samples/sc09/sashimi-diffwave-small-500k/`
    files = list(natsorted([e for e in os.listdir(f'{sample_dir}/{n_samples}-{method}') if e.endswith('.wav')]))

    # Grab indices of the top 50 most-confident samples for each digit
    indices, _ = grab_indices(probs, samples_per_class)

    # Move the top 50 confident samples for each digit to the method directory
    move_files(
        f'{sample_dir}/{n_samples}-{method}',
        files, 
        f'{target_dir}/{method}/',
        indices,
    )
    
    # Rename the files to `0.wav, 1.wav, ...` and flatten the target directory structure
    standardize_filenames(f'{target_dir}/{method}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, nargs='+', help='methods to prepare', required=True)
    parser.add_argument('--cache_dir', type=str, default='../sc09_classifier/cache/')
    parser.add_argument('--sample_dir', type=str, default='../samples/sc09/')
    parser.add_argument('--target_dir', type=str, default='sc09/sc09-unconditional-exp-confident-repro/')
    parser.add_argument('--n_samples', type=int, default=2048)
    parser.add_argument('--samples_per_class', type=int, default=50)
    args = parser.parse_args()

    for method in args.methods:
        print(f'Preparing {method}...')
        prepare(
            method,
            args.cache_dir,
            args.sample_dir,
            args.target_dir,
            args.n_samples,
            args.samples_per_class,
        )
    print('Done!')