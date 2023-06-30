import numpy as np
from tqdm import tqdm

db_path = "cornell_yes_playlist/train.txt"
db_path_2 = "cornell_yes_playlist/test.txt"

db_playlist = [('pid','tid')]

print("file 1...")
f = open(db_path)
f.readline()
f.readline()
pid = 0
line = f.readline().strip('\n')
lens = []
while line:
    tracks = line.split(' ')
    if len(tracks) >= 5:
        for t in tracks:
            if t != '':
                db_playlist.append((pid, t))
        pid += 1
    line = f.readline().strip('\n')
f.close()

print("file 2...")
f = open(db_path_2)
f.readline()
f.readline()
line = f.readline().strip('\n')
while line:
    tracks = line.split(' ')
    if len(tracks) >= 5:
        for t in tracks:
            if t != '':
                db_playlist.append((pid, t))
        pid += 1
    line = f.readline().strip('\n')
f.close()

with open('cornell_playlists.csv','w') as f:
    for pid, tid in db_playlist:
        f.write('{},{}\n'.format(pid, tid))