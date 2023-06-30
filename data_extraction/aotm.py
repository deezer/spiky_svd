import numpy as np
import json

path = "AoTM/aotm2011_playlists.json"

with open(path) as f:
    aotm_db = json.load(f)

playlist_db = []
for k, el in enumerate(aotm_db):
    playlist_db.append([k, sum(el['filtered_lists'], [])])

with open('aotm_playlists.csv','w') as f:
    f.write('pid,tid\n')
    for pid, tids in playlist_db:
        for tid in tids:
            f.write('{},{}\n'.format(pid,tid))