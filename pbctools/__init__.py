"""
pbctools - Lightweight PBC analysis toolkit

A high-performance Python package for periodic boundary condition calculations,
neighbor analysis, and molecular recognition in trajectory data.
"""

from .core import pbc_dist, next_neighbor, molecule_recognition

__version__ = "0.1.0"
__author__ = "Jonas Hänseroth"
__email__ = "jonas.haenseroth@tu-ilmenau.de"

__all__ = [
    "pbc_dist",
    "next_neighbor", 
    "molecule_recognition",
]