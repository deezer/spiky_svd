''' General script for SVD computation. '''
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sp

K = 128

name = "cornell"
path = "pmi_{}.parquet".format(name)

M = pd.read_parquet(path)
pti1 = M["t1"].to_list()
pti2 = M["t2"].to_list()
pmi = M["pmi"].to_list()
del M
gc.collect()

# reindexing
all_pti1 = list(set(pti1))
all_pti2 = list(set(pti2))
n1 = len(all_pti1)
n2 = len(all_pti2)
map2pti1 = {}
for k, pti in enumerate(all_pti1):
   map2pti1[pti] = k
map2pti2 = {}
for k, pti in enumerate(all_pti2):
   map2pti2[pti] = k
del all_pti1
del all_pti2
pti1 = [map2pti1[el] for el in pti1]
pti2 = [map2pti2[el] for el in pti2]
del map2pti1
del map2pti2

# randomised svd computation
matrix_MMstar = sp.csr_matrix( (pmi, (pti1, pti2)), shape=(n1, n2) )
svd = randomized_svd(matrix_MMstar, K, random_state=None)
np.save("svd_{}".format(name), svd)