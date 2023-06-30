# RecSys 2023 -- Spiky SVD

Code for our paper "Of Spiky SVDs and Music Recommendation" by D. Afchar, R. Hennequin and V. Guigue, that will be presented at RecSys 2023 (September 18-22nd, 2023).

## Dependencies

Our experiments run in Python and rely on a relatively light and standard toolkit in machine learning.  Specifically, the pip package versions we have used to run our experiments are as follows:

```
numpy==1.21.5
scikit-learn==1.0.2
scipy==1.7.3
pandas==1.3.5
```


## Experiments

1. Data is prepared using `ppmi.sql`;
2. SVDs are computed using Scikit-learn's [`randomized_svd()`](https://scikit-learn.org/1.0/modules/generated/sklearn.utils.extmath.randomized_svd.html), our precise script is provided in `compute_rsvd.py`;
3. **Figure 1**'s embedding scatterplots are obtained using `visualization.py` and may be used to have a fine-grained view on the spiking effect;
4. **Table 1** may be reproduced using `spk_computation.py`;
5. Stability plots are obtained using `stability.py`.


## Data

We report to each link provided in the paper to get the raw datasets.

To ease reproducibility, we added the precomputed SVDs of the `AoTM-2011`, `Cornell-Yes`, `LastFM-360K`, `Movielens-25M` datasets.
We cannot upload the precomputed SVDs of the two remaining datasets on GitHub due to the upload size limit (each `.npy` file is above 1GB). We will push said computed decompositions on Zenodo soon.

If not straightforward, scripts for selecting data from each dataset and converting it to a binary matrix are provided in the folder `data_extraction`.

We report some stats on each dataset:


| Dataset | #items | type |
| :----- | :----- | :----- |
| AoTM-2011 | 27.241 | track-playlist |
| Cornell | 44.442 | track-playlist |
| LastFM | 94.272 | artist-user |
| Movielens | 42.490 | movie-user |
| Spotify | 1.261.730 | track-playlist |
| Deezer | 1.438.603 | track-playlist |

The number of considered items may be lower than found in the original datasets due to the filtering of items with too few interactions (as described in `ppmi.sql`).

## Contact

[research@deezer.com](mailto:research@deezer.com)