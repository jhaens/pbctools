"""
pbctools.core - Python API for PBC calculations, neighbor detection, and molecule recognition
"""

import numpy as np
from typing import Tuple, Dict, List
from ._cpp import pbctools_cpp


def pbc_dist(coord1: np.ndarray, coord2: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    """
    Calculate periodic boundary condition distance vectors between atom sets across multiple frames.
    
    Parameters
    ----------
    coord1 : np.ndarray
        Coordinates of first atom set, shape (n_frames, n_atoms1, 3)
    coord2 : np.ndarray  
        Coordinates of second atom set, shape (n_frames, n_atoms2, 3)
    pbc : np.ndarray
        Periodic boundary condition matrix, shape (3, 3)
        
    Returns
    -------
    np.ndarray
        Distance vectors, shape (n_frames, n_atoms1, n_atoms2, 3)
        
    Examples
    --------
    >>> import numpy as np
    >>> from pbctools import pbc_dist
    >>> 
    >>> # Create sample data
    >>> coord1 = np.random.rand(100, 50, 3)  # 100 frames, 50 atoms
    >>> coord2 = np.random.rand(100, 30, 3)  # 100 frames, 30 atoms  
    >>> pbc = np.eye(3) * 20.0  # 20 Å cubic box
    >>> 
    >>> # Calculate distance vectors
    >>> distances = pbc_dist(coord1, coord2, pbc)
    >>> print(distances.shape)  # (100, 50, 30, 3)
    """
    # Input validation
    coord1 = np.asarray(coord1, dtype=np.float32)
    coord2 = np.asarray(coord2, dtype=np.float32) 
    pbc = np.asarray(pbc, dtype=np.float32)
    
    if coord1.ndim != 3 or coord1.shape[-1] != 3:
        raise ValueError(f"coord1 must have shape (n_frames, n_atoms, 3), got {coord1.shape}")
    if coord2.ndim != 3 or coord2.shape[-1] != 3:
        raise ValueError(f"coord2 must have shape (n_frames, n_atoms, 3), got {coord2.shape}")
    if pbc.shape != (3, 3):
        raise ValueError(f"pbc must have shape (3, 3), got {pbc.shape}")
    if coord1.shape[0] != coord2.shape[0]:
        raise ValueError(f"coord1 and coord2 must have same number of frames: {coord1.shape[0]} vs {coord2.shape[0]}")
    
    # Call C++ backend
    result = pbctools_cpp.pbc_dist(coord1.tolist(), coord2.tolist(), pbc.tolist())
    return np.array(result, dtype=np.float32)


def next_neighbor(coord1: np.ndarray, coord2: np.ndarray, pbc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find nearest neighbors between two atom sets across multiple frames.
    
    Parameters
    ----------
    coord1 : np.ndarray
        Coordinates of first atom set, shape (n_frames, n_atoms1, 3)
    coord2 : np.ndarray
        Coordinates of second atom set, shape (n_frames, n_atoms2, 3)  
    pbc : np.ndarray
        Periodic boundary condition matrix, shape (3, 3)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - nearest_indices: indices of nearest atoms in coord2, shape (n_frames, n_atoms1)
        - min_distances: minimum distances, shape (n_frames, n_atoms1)
        
    Examples
    --------
    >>> import numpy as np
    >>> from pbctools import next_neighbor
    >>> 
    >>> # OH- ions and water molecules
    >>> oh_coords = np.random.rand(1000, 4, 3)    # 4 OH- ions
    >>> h2o_coords = np.random.rand(1000, 500, 3) # 500 water molecules  
    >>> pbc = np.eye(3) * 25.0
    >>> 
    >>> # Find nearest water to each OH-
    >>> indices, distances = next_neighbor(oh_coords, h2o_coords, pbc)
    >>> print(f"Nearest water indices: {indices.shape}")    # (1000, 4)
    >>> print(f"Minimum distances: {distances.shape}")      # (1000, 4)
    """
    # Input validation (similar to pbc_dist)
    coord1 = np.asarray(coord1, dtype=np.float32)
    coord2 = np.asarray(coord2, dtype=np.float32)
    pbc = np.asarray(pbc, dtype=np.float32)
    
    if coord1.ndim != 3 or coord1.shape[-1] != 3:
        raise ValueError(f"coord1 must have shape (n_frames, n_atoms, 3), got {coord1.shape}")
    if coord2.ndim != 3 or coord2.shape[-1] != 3:
        raise ValueError(f"coord2 must have shape (n_frames, n_atoms, 3), got {coord2.shape}")
    if pbc.shape != (3, 3):
        raise ValueError(f"pbc must have shape (3, 3), got {pbc.shape}")
    if coord1.shape[0] != coord2.shape[0]:
        raise ValueError(f"coord1 and coord2 must have same number of frames")
    
    # Call C++ backend  
    indices, distances = pbctools_cpp.next_neighbor(coord1.tolist(), coord2.tolist(), pbc.tolist())
    return np.array(indices, dtype=np.int32), np.array(distances, dtype=np.float32)


def molecule_recognition(coords: np.ndarray, atoms: np.ndarray, pbc: np.ndarray) -> Dict[str, int]:
    """
    Identify molecular species in a single frame using bond detection.
    
    Parameters
    ----------
    coords : np.ndarray
        Atomic coordinates for single frame, shape (n_atoms, 3)
    atoms : np.ndarray
        Atomic symbols, shape (n_atoms,)
    pbc : np.ndarray
        Periodic boundary condition matrix, shape (3, 3)
        
    Returns
    -------
    Dict[str, int]
        Dictionary with molecular formulas as keys and counts as values
        
    Examples
    --------
    >>> import numpy as np
    >>> from pbctools import molecule_recognition
    >>> 
    >>> # Single frame with water and hydroxide
    >>> coords = np.array([
    ...     [0.0, 0.0, 0.0],    # O (water)
    ...     [1.0, 0.0, 0.0],    # H (water)
    ...     [0.0, 1.0, 0.0],    # H (water)
    ...     [5.0, 5.0, 5.0],    # O (hydroxide)
    ...     [5.8, 5.0, 5.0],    # H (hydroxide)
    ... ])
    >>> atoms = np.array(['O', 'H', 'H', 'O', 'H'])
    >>> pbc = np.eye(3) * 10.0
    >>> 
    >>> molecules = molecule_recognition(coords, atoms, pbc)
    >>> print(molecules)  # {'H2O': 1, 'OH': 1}
    """
    # Input validation
    coords = np.asarray(coords, dtype=np.float32)
    pbc = np.asarray(pbc, dtype=np.float32)
    
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape (n_atoms, 3), got {coords.shape}")
    if atoms.shape[0] != coords.shape[0]:
        raise ValueError(f"Number of atoms ({len(atoms)}) must match coordinates ({coords.shape[0]})")
    if pbc.shape != (3, 3):
        raise ValueError(f"pbc must have shape (3, 3), got {pbc.shape}")
    
    # Call C++ backend
    result = pbctools_cpp.molecule_recognition(coords.tolist(), atoms.tolist(), pbc.tolist())
    return dict(result)