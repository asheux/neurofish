import numpy as np
import pandas as pd

from scipy.stats import pearsonr

all_data = np.load('ZFish_dataNgenes.npz', allow_pickle=True) # Load data

def pre_process_data():
    def read_file(filename: str):
        if not filename.endswith('.csv'):
            raise Exception('Wrong file format.')
        return pd.read_csv(filename)

    file_names = all_data.files
    data_map = {}

    for file_name in file_names:
        df = pd.DataFrame(all_data[file_name])
        data_map[file_name] = df
    return data_map


def pca(data):
    mu = np.mean(data, axis=0) # compute the mean
    N, p = data.shape
    data_centered = data - mu
    cov = 1/N * data_centered.T @ data_centered # Compuete covariance
    # Perform the decomposition of the data covariance
    D, U = np.linalg.eig(cov) # D: vector of eigenvalues, U: eigenvectors matrix
    # make sure the eigenvalues are sorted (in descending order)
    index = np.argsort(D)[::-1]
    D = D[index]
    # arrange the eigenvectors according to the magnitude of the eigenvalues
    U = U[:, index]
    return U, D

def correlation(first, second):
    # The Pearson correlation coefficient measures the linear relationship between two datasets.
    # The p-value roughly indicates the probability of an uncorrelated system
    # statistic:  correlation coefficient.
    statistic, p_value = pearsonr(first, second)
    return statistic, p_value

processed_data_map = pre_process_data()
