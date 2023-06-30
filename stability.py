'''
Code to compute stability curves of embeddings.
'''

import gc
from collections import defaultdict
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

paths_svd = [
 'svd/anonymized/date/20220521/E.parquet',
 'svd/anonymized/date/20220528/E.parquet',
 'svd/anonymized/date/20220625/E.parquet',
 'svd/anonymized/date/20220730/E.parquet',
 'svd/anonymized/date/20220903/E.parquet',
 'svd/anonymized/date/20221008/E.parquet',
 'svd/anonymized/date/20221112/E.parquet',
 'svd/anonymized/date/20221217/E.parquet',
 'svd/anonymized/date/20230114/E.parquet',
 'svd/anonymized/date/20230218/E.parquet',
 'svd/anonymized/date/20230325/E.parquet',
 'svd/anonymized/date/20230429/E.parquet',
]

path_ref_svd = paths_svd[0]
ref_date = path_ref_svd.split('/')[-2]
paths_date = [a.split('/')[-2] for a in paths_svd]

tids = np.load('svd/anonymized/tids.npy')  # track ids

## partition tracks

R = 5  # nb partition
track_partition = [ [] for _ in range(R) ]

svd_db = pd.read_parquet(path_ref_svd)
svd_db = svd_db.set_index('track_id')
norm = np.sqrt(np.sum(np.square( np.stack(svd_db.loc[tids]["vector"]) ), 1))
del svd_db
norm_idx = np.argsort(norm)

l = len(norm_idx) // R
for k in range(R):
    track_partition[k] = norm_idx[k*l:(k+1)*l]
    print(norm[norm_idx[k*l]], ':', norm[norm_idx[(k+1)*l]])


## sample some points to compute topK on

L = 1000  # sampled points per partitions
sample_idx = np.array([], dtype=int)

for considered in track_partition:
    rand_idx = np.random.choice( considered, L, replace=False )
    sample_idx = np.concatenate([sample_idx, rand_idx])


## compute topk for all possible targets

K = 500 # cut-off of the top-K
cosine = False  # otherwise, dot product
output_folder = 'top_dot'

map_ids = []
for r in range(R):
    map_ids.append({})
    for k, el in enumerate(track_partition[r]):
        map_ids[-1][el] = k
for i in range(len(sample_idx)):
    sample_idx[i] = map_ids[i // L][sample_idx[i]]


def get_topks(idxs, F, K, J = 10):
    ''' get top K from similarity '''
    topks = []
    val_top = []
    for j in range(0, len(idxs), J):
        sims = F[idxs[j:j+J]] @ F.T
        topks_loc = np.argpartition(sims, -K, axis=1)[:,-K:]
        vals_loc = np.stack([
                sims[k, topk] for k, topk in enumerate(topks_loc)
            ])
        topks.append(topks_loc)
        val_top.append(vals_loc)

    topks = np.concatenate(topks, 0)
    val_top = np.concatenate(val_top, 0)
    return topks, val_top


for k, (path, date) in enumerate(zip(paths_svd, paths_date)):
    print(k, date)
    svd_db = pd.read_parquet(path)

    svd_db = svd_db.set_index('track_id')
    svd_db = svd.loc[tids]
    F = np.stack(svd_db["vector"])
    if cosine:
        F = F / np.sqrt(np.sum(np.square( F ), 1, keepdims=True) )
    print('opened svd...')
    del svd_db

    # get top_k
    for i in range(R):
        top, vals = get_topks(sample_idx[i*L:(i+1)*L], F[track_partition[i]], K)
        print('saving top', i)
        np.save('{}/tops_{}_{}'.format(output_folder, date, i), (top, vals))
        gc.collect()


##
## plot results

target_folder = 'top_dot'
folder_recs = glob('{}/*.npy'.format(target_folder))
R = 5
L = 1000
K = 500

# recos[<date>][<norm range>]
recos = defaultdict(dict)
for path in folder_recs:
    split = path.split('/')[-1].split('.')[0].split('_')
    top_k =  np.load(path)
    r = int(split[2])
    recos[split[1]][r] = {
                    'idx': top_k[0].astype(int),
                    'val': top_k[1],
                }


def iou(s1, s2):
    return len(s1 & s2) / len(s1)

cutoffs = [10, 100, 500]  # other cutoff of K

# Creates a dict such that
# ref_reco[<class norm>][<cutoffs>] contains a list of sets of top-ks
ref_recos = defaultdict(dict)
for r in range(R):
    idx = recos[ref_date][r]['idx']
    vals = recos[ref_date][r]['val']
    sorted_idxes = np.argsort(-vals, -1)
    for k in cutoffs:
        ref_recos[r][k] = []
        for tracks, sorted_idx in zip(idx, sorted_idxes):
            ref_recos[r][k].append(set(tracks[sorted_idx[:k]]))

# Creates a dict of IoUs
# IoUs[idx_date][<class norm>][<cutoffs>]
IoUs = defaultdict(lambda: defaultdict(dict))
entropy_items = [[defaultdict(float) for _ in range(len(cutoffs))] for _ in range(R)]
for d, date in enumerate(paths_date[1:]):
    for r in range(R):
        idx = recos[date][r]['idx']
        vals = recos[date][r]['val']
        sorted_idxes = np.argsort(-vals, -1)

        for ki, k in enumerate(cutoffs):
            IoUs[d][r][k] = []
        for l, (tracks, sorted_idx) in enumerate(zip(idx, sorted_idxes)):
            for ki, k in enumerate(cutoffs):
                item_recos = tracks[sorted_idx[:k]]
                for el in item_recos.flatten():
                    entropy_items[r][ki][el] += 1.
                IoUs[d][r][k].append(
                        iou(set(item_recos), ref_recos[r][k][l] )
                    )


## create plot

markers = ['o', 'v', '^', 's', 'D']
cmap = plt.get_cmap("tab10")

K_ = cutoffs[2]
fig, ax = plt.subplots(1, 1, figsize=(8,5))
fact_conf = 1.96 / np.sqrt(L)

scores = defaultdict(list)
scores_std = defaultdict(list)

for d, date in enumerate(paths_date[1:]):
    for r in range(R):
        scores[r].append( np.mean( IoUs[d][r][K_] ) )
        scores_std[r].append( np.std( IoUs[d][r][K_] ) )

for r in range(R):
    ax.plot(scores[r], color=cmap(r), marker=markers[r])
    ax.fill_between(np.arange(0, len(scores[r])),
            np.array(scores[r]) + fact_conf * np.array(scores_std[r]),
            np.array(scores[r]) - fact_conf * np.array(scores_std[r]),
            alpha=0.2, color=cmap(r)
            )

ax.set_xticks(np.arange(0, len(scores[0]), 2))
ax.set_xticklabels(paths_date[1::2])
ax.set_title("IoU of top-{} from ref {} ({})".format(K_, paths_date[0], target_folder))
plt.xlabel('Date of SVD to compare')
plt.ylabel('Average IoU')
plt.show()
