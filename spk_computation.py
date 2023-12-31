'''
Compute spikiness metric Spk@f

Greedy heuristic to a partial set cover.
'''

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # optional

DATASET = "AoTM"
SVD_DIR = "svds"
SUBSAMPLE_POINTS = 1_000_000
CUTOFF_F = 64
COS_THETA_THRESHOLD = 0.9
RHO_THRESHOLD = 0.5

try:
    U = np.load("{}/{}/U.npy".format(SVD_DIR, DATASET.lower()))
    s = np.load("{}/{}/S.npy".format(SVD_DIR, DATASET.lower()))
    V = np.load("{}/{}/V.npy".format(SVD_DIR, DATASET.lower()))

except:
    raise ValueError("Not uploaded or unavailable data: `{}`".format(DATASET))

E = U * s
del U, s, V

print("Opened embeddings:", E.shape)


## prep data

subsample_idx = np.random.choice(np.arange(0, E.shape[0]), min(SUBSAMPLE_POINTS, E.shape[0]), replace=False)
F = E[subsample_idx, :CUTOFF_F]
F_norm = np.sqrt(np.sum(np.square(F), 1))
F = F / F_norm[:,None]
idx_top = np.argsort(F_norm)[::-1]
F = F[idx_top]
del F_norm


## efficient finding algorithm

max_iter = 50000  # fail-safe
managed = np.zeros(F.shape[0], dtype=bool)
all_idx = np.arange(0, F.shape[0])
scores = []
used_k = []
for k in tqdm(range(max_iter)):
    if managed[k]:
        continue
    managed[k] = True
    used_k.append(k)
    peak = F[k:k+1]
    belong_to_spike = np.sum(F[~managed] * peak, -1) > COS_THETA_THRESHOLD
    managed[~managed] = belong_to_spike
    scores.append(np.sum(managed))
    if scores[-1] > (F.shape[0] * RHO_THRESHOLD):
        break

print("Used peaks: abs val={}, ratio={:.2f}%".format(len(used_k), len(used_k) / F.shape[0] * 100))


## plot clustering profile

plt.plot(used_k, np.array(scores) / F.shape[0])
plt.title('({} peaks) {}/{} (dim {})'.format(len(used_k), k, F.shape[0], F.shape[1]))
plt.ylabel('ratio of clustered points')
plt.xlabel('added point with k-th higher norm')
plt.show()
