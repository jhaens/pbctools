import numpy as np
from typing import Tuple, Dict, List, Union, Optional
from ._cpp import pbctools_cpp

def _is_ase_atoms(obj):
    return hasattr(obj, 'get_positions') and hasattr(obj, 'get_chemical_symbols') and hasattr(obj, 'get_cell')
def _is_ase_trajectory(obj):
    return (isinstance(obj, (list, tuple)) and len(obj) > 0 and _is_ase_atoms(obj[0]))
def _extract_from_ase_atoms(atoms_obj):
    return atoms_obj.get_positions(), np.array(atoms_obj.get_chemical_symbols()), atoms_obj.get_cell()[:]
def _extract_from_ase_trajectory(traj):
    coords_list, atoms, pbc = [], None, None
    for frame in traj:
        frame_coords, frame_atoms, frame_pbc = _extract_from_ase_atoms(frame)
        coords_list.append(frame_coords)
        if atoms is None:   # Use atoms and pbc from first frame (assuming they're consistent)
            atoms, pbc = frame_atoms, frame_pbc
    return np.array(coords_list), atoms, pbc
def _prepare_coords_and_pbc(coord_input):
    """
    Convert various input formats to (coords, atoms, pbc) tuple.
    Parameters: coord_input: np.array, ASE Atoms object, List of ASE Atoms objects (trajectory)
    Returns: (coords, atoms, pbc) tuple where coords has shape (n_frames, n_atoms, 3)
    """
    if _is_ase_atoms(coord_input): # Single ASE Atoms object - convert to trajectory format
        coords, atoms, pbc = _extract_from_ase_atoms(coord_input)
        return coords, atoms, pbc
    elif _is_ase_trajectory(coord_input): # List/trajectory of ASE Atoms objects
        coords, atoms, pbc = _extract_from_ase_trajectory(coord_input)
        return coords, atoms, pbc
    else: # Assume numpy array input
        return coord_input, None

def pbc_dist(coord1, coord2=None, pbc=None):
    """
    Calculate periodic boundary condition distance vectors between atom sets across multiple frames.
    
    Parameters
    ----------
    coord1 : ASE Atoms, list of ASE Atoms, or np.ndarray
        Coordinates of first atom set, numpy array shape (n_frames, n_atoms1, 3) or ASE Atoms object or list of ASE Atoms objects.
    coord2 : ASE Atoms, list of ASE Atoms, or np.ndarray, optional
        Coordinates of second atom set, numpy array shape (n_frames, n_atoms2, 3) or ASE Atoms object or list of ASE Atoms objects. If None, coord1 is used for both sets.
    pbc : np.ndarray, optional
        Periodic boundary condition matrix, shape (3, 3). If None and using ASE objects, will be extracted from the first ASE object.

    Returns
    -------
    np.ndarray
        Distance vectors, shape (n_frames, n_atoms1, n_atoms2, 3)
        
    """
    if _is_ase_atoms(coord1) or _is_ase_trajectory(coord1):
        coord1, atoms1, pbc1 = _prepare_coords_and_pbc(coord1)
        coord2, _, _ = coord1, atoms1, pbc1 if coord2 is None else _prepare_coords_and_pbc(coord2)
        pbc = pbc if pbc is not None else pbc1 
    elif coord2 is None:
        coord2 = coord1

    coord1, coord2, pbc = np.asarray(coord1, dtype=np.float32), np.asarray(coord2, dtype=np.float32), np.asarray(pbc, dtype=np.float32)
    coord1 = coord1[np.newaxis, :, :] if coord1.ndim == 2 else coord1
    coord2 = coord2[np.newaxis, :, :] if coord2.ndim == 2 else coord2

    if coord1.ndim != 3 or coord1.shape[-1] != 3:
        raise ValueError(f"coord1 must have shape (n_frames, n_atoms, 3), got {coord1.shape}")
    if coord2.ndim != 3 or coord2.shape[-1] != 3:
        raise ValueError(f"coord2 must have shape (n_frames, n_atoms, 3), got {coord2.shape}")
    if pbc.shape != (3, 3):
        raise ValueError(f"pbc must have shape (3, 3), got {pbc.shape}")
    if coord1.shape[0] != coord2.shape[0]:
        raise ValueError(f"coord1 and coord2 must have same number of frames: {coord1.shape[0]} vs {coord2.shape[0]}")
    
    # Call optimized C++ backend 
    return pbctools_cpp.pbc_dist(coord1, coord2, pbc)


def next_neighbor(coord1, coord2=None, pbc=None):
    """
    Find nearest neighbors between two atom sets across multiple frames.
    
    Parameters
    ----------
    coord1 : ASE Atoms, list of ASE Atoms, or np.ndarray
        Coordinates of first atom set, numpy array shape (n_frames, n_atoms1, 3) or ASE Atoms object or list of ASE Atoms objects.
    coord2 : ASE Atoms, list of ASE Atoms, or np.ndarray
        Coordinates of second atom set, numpy array shape (n_frames, n_atoms2, 3) or ASE Atoms object or list of ASE Atoms objects. If None, coord1 is used for both sets.
    pbc : np.ndarray, optional
        Periodic boundary condition matrix, shape (3, 3). If None and using ASE objects, will be extracted from the first ASE object.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - nearest_indices: indices of nearest atoms in coord2, shape (n_frames, n_atoms1)
        - min_distances: minimum distances, shape (n_frames, n_atoms1)
        
    """
    if _is_ase_atoms(coord1) or _is_ase_trajectory(coord1):
        coord1, atoms1, pbc1 = _prepare_coords_and_pbc(coord1)
        coord2, atoms2, pbc2 = coord1, atoms1, pbc1 if coord2 is None else _prepare_coords_and_pbc(coord2)
        pbc = pbc if pbc is not None else pbc1 

    coord1, coord2, pbc = np.asarray(coord1, dtype=np.float32), np.asarray(coord2, dtype=np.float32), np.asarray(pbc, dtype=np.float32)
    coord1 = coord1[np.newaxis, :, :] if coord1.ndim == 2 else coord1
    coord2 = coord2[np.newaxis, :, :] if coord2.ndim == 2 else coord2
    
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


def molecule_recognition(coords, atoms = None, pbc = None):
    """
    Identify molecular species in a single frame using bond detection.
    
    Parameters
    ----------
    coords : ASE Atoms or np.ndarray
        Atomic coordinates for single frame, shape (n_atoms, 3) or ASE Atoms object (atoms and pbc extracted automatically)
    atoms : np.ndarray or list, optional
        Atomic symbols, shape (n_atoms,). Required if coords_input is numpy array.
    pbc : np.ndarray, optional
        Periodic boundary condition matrix, shape (3, 3). Required if coords_input is numpy array.

    Returns
    -------
    Dict[str, int]
        Dictionary with molecular formulas as keys and counts as values
        
    """
    if _is_ase_atoms(coords):
        coords, atoms, pbc = _prepare_coords_and_pbc(coords)
        pbc = pbc if pbc is not None else pbc 

    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape (n_atoms, 3), got {coords.shape}")
    if atoms.shape[0] != coords.shape[0]:
        raise ValueError(f"Number of atoms ({len(atoms)}) must match coordinates ({coords.shape[0]})")
    if pbc.shape != (3, 3):
        raise ValueError(f"pbc must have shape (3, 3), got {pbc.shape}")
    
    # Call C++ backend
    result = pbctools_cpp.molecule_recognition(coords, list(atoms), pbc)
    return dict(result)