import os
import numpy as np
import pandas as pd
from sktime.utils.data_io import load_from_arff_to_dataframe
from sklearn.model_selection import train_test_split

DATA_PATH = 'data/'

def split_data(X_train_orig, y_train_orig, X_test_orig, y_test_orig, shuffle=True, seed=0):
    if shuffle:
        X_all = pd.concat((X_train_orig, X_test_orig))
        y_all = np.concatenate((y_train_orig, y_test_orig))
        X_train, X_eval, y_train, y_eval = train_test_split(X_all, y_all, test_size=0.3, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(X_eval, y_eval, test_size=0.5, random_state=seed+1)
    else:
        X_test, y_test = X_test_orig, y_test_orig
        X_train, X_val, y_train, y_val = train_test_split(X_train_orig, y_train_orig, test_size=0.20, random_state=seed)
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train_orig, y_train_orig = load_from_arff_to_dataframe(
    os.path.join(DATA_PATH, "EigenWorms_TRAIN.arff")
)
X_test_orig, y_test_orig = load_from_arff_to_dataframe(
    os.path.join(DATA_PATH, "EigenWorms_TEST.arff")
)

X_train, y_train, X_val, y_val, X_test, y_test = split_data(X_train_orig, y_train_orig, X_test_orig, y_test_orig, shuffle=True, seed=0)

def _to_numpy(X):
    return np.stack([np.stack(x) for x in X.to_numpy()]).swapaxes(-1, -2)

X_train = _to_numpy(X_train)
X_val = _to_numpy(X_val)
X_test = _to_numpy(X_test)

mean = np.mean(X_train.reshape((-1,6)), axis=0)
std = np.std(X_train.reshape((-1,6)), axis=0)
print(mean.shape, std.shape)
print(mean)
print(std)
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

np.save(os.path.join(DATA_PATH, "trainx.npy"), X_train)
np.save(os.path.join(DATA_PATH, "trainy.npy"), y_train.astype(int)-1)
np.save(os.path.join(DATA_PATH, "validx.npy"), X_val)
np.save(os.path.join(DATA_PATH, "validy.npy"), y_val.astype(int)-1)
np.save(os.path.join(DATA_PATH, "testx.npy"), X_test)
np.save(os.path.join(DATA_PATH, "testy.npy"), y_test.astype(int)-1)


for f in ['trainx', 'trainy', 'validx', 'validy', 'testx', 'testy']:
    df = np.load(f"data/{f}.npy")
    print(f, df.shape, df.dtype)
