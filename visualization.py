'''
Visualize the distributions in 3D
'''

import numpy as np
import matplotlib.pyplot as plt

DATASET = "Cornell"
SUBSAMPLE_POINTS = 5_000

if DATASET == "AOTM":
    U = np.load("svd/aotm/U.npy")
    s = np.load("svd/aotm/S.npy")
    V = np.load("svd/aotm/V.npy")
    color = 'r'

elif DATASET == "Cornell":
    U = np.load("svd/cornell/U.npy")
    s = np.load("svd/cornell/S.npy")
    V = np.load("svd/cornell/V.npy")
    color = 'darkorange'

elif DATASET == "LastFM":
    U = np.load("svd/lastfm/U.npy")
    s = np.load("svd/lastfm/S.npy")
    V = np.load("svd/lastfm/V.npy")
    color = 'm'

elif DATASET == "Movielens":
    U, s, V = np.load("svd/movielens/UsV.npy", allow_pickle=True)
    color = 'c'

else:
    raise ValueError("Not uploaded or unavailable:", DATASET)

E = U * s
del U, s, V
norm = np.sqrt(np.sum(np.square(E), 1))

print("Opened embeddings:", E.shape)

## visualization

PROJECTION_DIM = int(input('Enter projection first dimension: '))
name = str.lower(DATASET)
random_idx = np.random.choice(np.arange(0, E.shape[0]), SUBSAMPLE_POINTS, replace=False)  # sub-sample


def plot_dim(dir1, E, random_idx):
    ax.plot([min(E[random_idx, dir1]) - 1,max(E[random_idx, dir1]) + 1], [0, 0], [0, 0], 'k:', alpha=0.8, linewidth=1)
    ax.plot([0, 0], [min(E[random_idx, dir1+1]) - 1,max(E[random_idx, dir1+1]) + 1], [0, 0], 'k:', alpha=0.8, linewidth=1)
    ax.plot([0, 0], [0, 0], [min(E[random_idx, dir1+2]) - 1,max(E[random_idx, dir1+2]) + 1], 'k:', alpha=0.8, linewidth=1)

    ax.scatter(E[random_idx, dir1], E[random_idx, dir1+1], E[random_idx, dir1+2], c=color, alpha=0.1)
    ax.set_xlabel("dim {}".format(dir1))
    ax.set_ylabel("dim {}".format(dir1+1))
    ax.set_zlabel("dim {}".format(dir1+2))
    plt.title('({},{},{})'.format(dir1, dir1+1, dir1+2))

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
plot_dim(PROJECTION_DIM, E, random_idx)
plt.title("Dataset {} proj<{},{},{}>".format(name, PROJECTION_DIM, PROJECTION_DIM+1, PROJECTION_DIM+2))
plt.show()