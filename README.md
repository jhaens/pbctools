# pbctools

A lightweight Python package for periodic boundary condition calculations, neighbor analysis, and molecular recognition in trajectory data.

## Features

- **PBC Distance Calculation**: Compute distance vectors between atom sets across multiple frames
- **Nearest Neighbor Detection**: Find nearest neighbors with PBC support for trajectory data  
- **Molecule Recognition**: Identify molecular species using bond detection algorithms
- **High Performance**: C++ backend with OpenMP acceleration
- **Easy Integration**: Simple NumPy-based Python API

## Installation

Install from source:
```bash
git clone https://github.com/jhaens/pbctools.git
cd pbctools
pip install .
```

## Quick Start

```python
import numpy as np
from pbctools import pbc_dist, next_neighbor, molecule_recognition
from ase.io import read, write

# OPTION 1
# Load single trajectory frames data
coords1 = read('coord1.xyz')
coords1.set_cell(np.loadtxt('pbc1.txt'))
coords2 = read('coord2.xyz')
coords2.set_cell(np.loadtxt('pbc2.txt'))

# Calculate distance vectors between all atom pairs (ASE objects auto-extract PBC)
distances = pbc_dist(coords1, coords2)
print(f"Distance shape: {distances.shape}")  # (1, n_atoms1, n_atoms2, 3) for single-frame inputs

# Find nearest neighbors
indices, nn_dists = next_neighbor(coords1, coords2)
print(indices.shape, nn_dists.shape)  # (1, n_atoms1), (1, n_atoms1) for single-frame inputs

# Analyze molecular composition (single frame)
molecules = molecule_recognition(coords1)
print(f"Found molecules: {molecules}")  # e.g., {'H2O': 100}

# OPTION 2
# Load trajectory data
coords1 = read('traj1.xyz', index=':')
coords2 = read('traj2.xyz', index=':')
pbc = np.loadtxt('pbc.txt')
for frame in coords1:
	frame.set_cell(np.array(pbc))

# ...

```

## API Reference

### pbc_dist(coord1, coord2=None, pbc=None)
Calculate periodic boundary condition distance vectors.

Supports flexible inputs:
- ASE Atoms (single frame) or list of Atoms (trajectory) — PBC auto-extracted
- NumPy arrays

Parameters:
- coord1: ASE Atoms | list[Atoms] | np.ndarray
	- If ndarray: shape (n_frames, n_atoms1, 3) or (n_atoms1, 3) for single-frame
- coord2: ASE Atoms | list[Atoms] | np.ndarray | None (default: coord1)
	- If ndarray: shape (n_frames, n_atoms2, 3) or (n_atoms2, 3)
- pbc: np.ndarray | None
	- Required for ndarray inputs; shape (3, 3). Ignored when using ASE objects (read from Atoms).

Returns:
- np.ndarray with shape (n_frames, n_atoms1, n_atoms2, 3)
	- For single-frame inputs, n_frames = 1

Notes:
- To compute within-set distances, pass the same object/array for coord1 and coord2.

### next_neighbor(coord1, coord2=None, pbc=None)
Find nearest neighbors between two atom sets (PBC-aware).

Parameters:
- coord1: ASE Atoms | list[Atoms] | np.ndarray (…, 3)
- coord2: ASE Atoms | list[Atoms] | np.ndarray (…, 3) | None (default: coord1)
- pbc: np.ndarray (3, 3) | None

Returns:
- indices: np.ndarray of int with shape (n_frames, n_atoms1)
- distances: np.ndarray of float32 with shape (n_frames, n_atoms1)

### molecule_recognition(coords, atoms=None, pbc=None)
Identify molecular species in a single frame via bond detection.

Parameters:
- coords: ASE Atoms | np.ndarray, shape (n_atoms, 3)
- atoms: list[str] | np.ndarray[str] | None — required for ndarray input; ignored for ASE Atoms
- pbc: np.ndarray (3, 3) | None — required for ndarray input; ignored for ASE Atoms

Returns:
- dict[str, int] — molecular formulas and counts

## Performance

pbctools is optimized for large trajectory analysis:
- Multi-threaded C++ backend 
- Support for both orthogonal and triclinic unit cells
- Optimized distance calculations with PBC

## License

MIT License - see LICENSE file for details.