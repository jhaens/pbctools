import numpy as np
from pbctools import pbc_dist, next_neighbor, molecule_recognition
from ase.io import read, write

coords1 = read('100_steps_4koh.xyz', index=":")
pbc = np.loadtxt('pbc')
for frame in coords1:
	frame.set_cell(np.array(pbc))

molecules = molecule_recognition(coords1[0])
print(molecules)
#{'HO': 4, 'K': 4, 'H2O': 92}

distances = pbc_dist(coords1)
print(distances.shape)
#(100, 288, 288, 3)

symbols = np.array(coords1[0].get_chemical_symbols())
coordsO = coords1[0].get_positions()[symbols == "O"]
print(coordsO.shape)
#(96, 3)
coordsH = coords1[0].get_positions()[np.array(coords1[0].get_chemical_symbols()) == "H"]
print(coordsH.shape)
#(188, 3)

indices, distances = next_neighbor(coordsO, coordsH, pbc)
print(indices)
# [[  0   1   2   3   ... 187]]
print(distances)
# [[0.9858405  0.9920111  0.9314136  0.9863206  ...  0.9906035 ]]
