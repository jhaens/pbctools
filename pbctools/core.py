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
    # Call optimized C++ backend directly with numpy arrays
    return pbctools_cpp.pbc_dist(coord1, coord2, pbc)


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
    indices, distances = pbctools_cpp.next_neighbor(coord1, coord2, pbc)
    return np.asarray(indices, dtype=np.int32), np.asarray(distances, dtype=np.float32)


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
    result = pbctools_cpp.molecule_recognition(coords, list(atoms), pbc)
    return dict(result)