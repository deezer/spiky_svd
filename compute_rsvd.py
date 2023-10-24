''' General script for SVD computation. '''
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sp
import gc
from time import time

K = 128

name = "cornell"
path = "pmi_{}.parquet".format(name)
save_preprocess = False

##

print("preprocessing...")
M = pd.read_parquet(path)
pti1 = M["t1"].to_list()
pti2 = M["t2"].to_list()
pmi = M["pmi"].to_list()
del M
gc.collect()

all_pti = list(set(pti1) | set(pti2))
n = len(all_pti)

map2pti = {}
for k, pti in enumerate(all_pti):
   map2pti[pti] = k
if save_preprocess:
   np.save("idx_{}".format(name), all_pti)
del all_pti

pti1 = [map2pti[el] for el in pti1]
pti2 = [map2pti[el] for el in pti2]
del map2pti

print("instantiating sparse matrix ({},{})".format(n,n))
matrix_MMstar = sp.csr_matrix( (pmi, (pti1, pti2)), shape=(n, n) )
if save_preprocess:
   np.save("matrix_{}".format(name), matrix_MMstar)

rows, cols = matrix_MMstar.nonzero()
matrix_MMstar[cols, rows] = matrix_MMstar[rows, cols]

##

print("svd...")
ts = int(time())
svd = randomized_svd(matrix_MMstar, K, random_state=ts)
np.save("{}_{}_U".format(name, ts), svd[0])
np.save("{}_{}_S".format(name, ts), svd[1])
np.save("{}_{}_V".format(name, ts), svd[2])