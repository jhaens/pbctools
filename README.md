# pbctools

A lightweight Python package for periodic boundary condition calculations, neighbor analysis, and molecular recognition in trajectory data.

## Features

- **PBC Distance Calculation**: Compute distance vectors between atom sets across multiple frames
- **Nearest Neighbor Detection**: Find nearest neighbors with PBC support for trajectory data  
- **Molecule Recognition**: Identify molecular species using bond detection algorithms
- **High Performance**: C++ backend with OpenMP acceleration
- **Easy Integration**: Simple NumPy-based Python API

## Installation

```bash
pip install pbctools
```

Or install from source:
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
data1 = read('coord1.xyz')
data1.set_cell(np.loadtxt('pbc1.txt'))
data2 = read('coord2.xyz')
data2.set_cell(np.loadtxt('pbc2.txt'))

# Convert ASE atoms to NumPy arrays
coords1 = data1.get_positions()
atoms1 = np.array(data1.get_chemical_symbols())
coords2 = data2.get_positions()
atoms2 = np.array(data2.get_chemical_symbols())
pbc = data1.get_cell()


# OPTION 2
# Load trajectory data
data1 = read('traj1.xyz', index=':')
data2 = read('traj2.xyz', index=':')
pbc = np.loadtxt('pbc.txt')

# Convert ASE trajectory to NumPy arrays
coords1 = np.array([frame.get_positions() for frame in data1])
atoms1 = np.array(data1[0].get_chemical_symbols())
coords2 = np.array([frame.get_positions() for frame in data2])
atoms2 = np.array(data2[0].get_chemical_symbols())

# Calculate distance vectors between all atom pairs
distances = pbc_dist(coords1, coords2, pbc)
print(f"Distance shape: {distances.shape}")  # (1000, 50, 30, 3)

# Find nearest neighbors
indices, distances = next_neighbor(coords1, coords2, pbc)
print(f"Nearest indices: {indices.shape}")   # (1000, 50)
print(f"Minimum distances: {distances.shape}")  # (1000, 50)

# Analyze molecular composition (single frame)
atoms = ['O', 'H', 'H'] * 100  # 100 water molecules
molecules = molecule_recognition(coords1[0], atoms1, pbc)
print(f"Found molecules: {molecules}")  # {'H2O': 100}
```

## API Reference

### pbc_dist(coord1, coord2, pbc)
Calculate periodic boundary condition distance vectors.

**Parameters:**
- `coord1`: np.ndarray, shape (n_frames, n_atoms1, 3)
- `coord2`: np.ndarray, shape (n_frames, n_atoms2, 3) 
- `pbc`: np.ndarray, shape (3, 3)

**Returns:**
- `distance vector`: np.ndarray, shape (n_frames, n_atoms1, n_atoms2, 3)

### next_neighbor(coord1, coord2, pbc)
Find nearest neighbors between two atom sets. If you are interested in the nearest neighbor within the same set, pass the same coordinates for both parameters `coord1` and `coord2`.

**Parameters:**
- `coord1`: np.ndarray, shape (n_frames, n_atoms1, 3)
- `coord2`: np.ndarray, shape (n_frames, n_atoms2, 3)
- `pbc`: np.ndarray, shape (3, 3) - PBC matrix

**Returns:**
- `indices`: np.ndarray, shape (n_frames, n_atoms1) - Nearest atom indices
- `distances`: np.ndarray, shape (n_frames, n_atoms1) - Minimum distances

### molecule_recognition(coords, atoms, pbc)
Identify molecular species in a single frame.

**Parameters:**
- `coords`: np.ndarray, shape (n_atoms, 3) - Single frame coordinates
- `atoms`: list[str], length n_atoms - Atomic symbols
- `pbc`: np.ndarray, shape (3, 3) - PBC matrix

**Returns:**
- `molecule_dict`: dict[str, int] - Molecular species and their counts 

## Performance

pbctools is optimized for large trajectory analysis:
- Multi-threaded C++ backend 
- Support for both orthogonal and triclinic unit cells
- Optimized distance calculations with PBC

## License

MIT License - see LICENSE file for details.