import numpy as np
from pbctools import pbc_dist, next_neighbor, molecule_recognition
from ase.io import read, write

data1 = read('100_steps_4koh.xyz', index=":")
pbc = np.loadtxt('pbc')
coords1 = np.array([frame.get_positions() for frame in data1])
atoms1 = np.array(data1[0].get_chemical_symbols())

molecules = molecule_recognition(coords1[0], atoms1, pbc)
print(molecules)
#{'HO': 4, 'K': 4, 'H2O': 92}

distances = pbc_dist(coords1, coords1, pbc)
distances.shape
#(100, 288, 288, 3)

coordsO = coords1[0][atoms1=="O"]
coordsO.shape
#(96, 3)
coordsH = coords1[0][atoms1=="H"]
coordsH.shape
#(188, 3)

indices, distances = next_neighbor(np.array([coordsO]), np.array([coordsH]), pbc)
print(indices)
# [[  0   1   2   3   ... 187]]
print(distances)
# [[0.9858405  0.9920111  0.9314136  0.9863206  ...  0.9906035 ]]
