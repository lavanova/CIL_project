from sklearn.manifold import LocallyLinearEmbedding, SpectralEmbedding
from sklearn.decomposition import FactorAnalysis, NMF
import numpy as np
import os
import csv
data_path = '../data'

def embedding(n_components=64):
    row_col = []
    label = []
    max_row = 1
    max_col = 1
    with open(os.path.join(data_path, "data_train.csv")) as f:
        reader = csv.reader(f, delimiter=',')
        for i, sample in enumerate(reader):
            if i == 0:
                continue
            if sample == None or sample == "":
                continue
            row = int(sample[0].split('_')[0][1:])
            max_row = max(max_row, row)
            col = int(sample[0].split('_')[1][1:])
            max_col = max(max_col, col)
            row_col.append([row, col])
            rating = int(sample[1])
            label.append(rating)

    row_col = np.asarray(row_col)
    label = np.asarray(label, dtype=np.float32).reshape((-1, 1)) #reshape??
    assert row_col.shape[0] == label.shape[0], "error sample number doesn't match with label number"

    matrix = np.zeros((max_row, max_col), dtype=np.float32)
    for i in range(row_col.shape[0]):
        matrix[ row_col[i, 0] - 1, row_col[i, 1] - 1 ] = label[i, 0]
    
    zero_emb = np.zeros((1, n_components), dtype=np.float32)

    row_spectral = SpectralEmbedding(n_components=n_components, random_state=0, n_jobs=8).fit_transform(matrix)
    col_spectral = SpectralEmbedding(n_components=n_components, random_state=0, n_jobs=8).fit_transform(matrix.T)

    row_spectral_embedding = np.concatenate((zero_emb, row_spectral), axis=0)
    col_spectral_embedding = np.concatenate((zero_emb, col_spectral), axis=0)
    np.save('./data/row_spectral_embedding.npy', row_spectral_embedding)
    np.save('./data/col_spectral_embedding.npy', col_spectral_embedding)

    row_lle = LocallyLinearEmbedding(n_components=n_components, random_state=0, n_jobs=8).fit_transform(matrix)
    col_lle = LocallyLinearEmbedding(n_components=n_components, random_state=0, n_jobs=8).fit_transform(matrix.T)

    row_lle_embedding = np.concatenate((zero_emb, row_lle), axis=0)
    col_lle_embedding = np.concatenate((zero_emb, col_lle), axis=0)
    np.save('./data/row_lle_embedding.npy', row_lle_embedding)
    np.save('./data/col_lle_embedding.npy', col_lle_embedding)

    row_factor = FactorAnalysis(n_components=n_components, random_state=0).fit_transform(matrix)
    col_factor = FactorAnalysis(n_components=n_components, random_state=0).fit_transform(matrix.T)

    row_factor_embedding = np.concatenate((zero_emb, row_factor), axis=0)
    col_factor_embedding = np.concatenate((zero_emb, col_factor), axis=0)
    np.save('./data/row_factor_embedding.npy', row_factor_embedding)
    np.save('./data/col_factor_embedding.npy', col_factor_embedding)

    row_nmf = NMF(n_components=n_components, random_state=0).fit_transform(matrix)
    col_nmf = NMF(n_components=n_components, random_state=0).fit_transform(matrix.T)

    row_nmf_embedding = np.concatenate((zero_emb, row_nmf), axis=0)
    col_nmf_embedding = np.concatenate((zero_emb, col_nmf), axis=0)
    np.save('./data/row_nmf_embedding.npy', row_nmf_embedding)
    np.save('./data/col_nmf_embedding.npy', col_nmf_embedding)

if __name__ == "__main__":
    embedding()
