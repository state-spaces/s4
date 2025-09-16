import numpy as np

from scipy import linalg
from scipy.stats import norm, entropy
from sklearn.cluster import KMeans

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
