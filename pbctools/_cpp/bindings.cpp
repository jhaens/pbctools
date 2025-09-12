// bindings.cpp - pybind11 bindings for pbctools

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "pbctools_cpp.h"

namespace py = pybind11;

PYBIND11_MODULE(pbctools_cpp, m) {
    m.doc() = "pbctools C++ extension - PBC calculations, neighbor detection, and molecule recognition";
    
    // Main functions
    m.def("pbc_dist", &pbctools::pbc_dist,
          py::arg("coord1"), py::arg("coord2"), py::arg("pbc"),
          "Calculate PBC distance vectors between atom sets across multiple frames.\n\n"
          "Parameters\n"
          "----------\n"
          "coord1 : list\n"
          "    Coordinates of first atom set, shape [n_frames][n_atoms1][3]\n"
          "coord2 : list\n"
          "    Coordinates of second atom set, shape [n_frames][n_atoms2][3]\n"
          "pbc : list\n"
          "    PBC matrix, shape [3][3]\n\n"
          "Returns\n"
          "-------\n"
          "list\n"
          "    Distance vectors, shape [n_frames][n_atoms1][n_atoms2][3]");
    
    m.def("next_neighbor", &pbctools::next_neighbor,
          py::arg("coord1"), py::arg("coord2"), py::arg("pbc"),
          "Find nearest neighbors between atom sets across multiple frames.\n\n"
          "Parameters\n"
          "----------\n"
          "coord1 : list\n"
          "    Coordinates of first atom set, shape [n_frames][n_atoms1][3]\n"
          "coord2 : list\n"
          "    Coordinates of second atom set, shape [n_frames][n_atoms2][3]\n"
          "pbc : list\n"
          "    PBC matrix, shape [3][3]\n\n"
          "Returns\n"
          "-------\n"
          "tuple\n"
          "    (indices, distances) with shapes [n_frames][n_atoms1]");
    
    m.def("molecule_recognition", &pbctools::molecule_recognition,
          py::arg("coords"), py::arg("atoms"), py::arg("pbc"),
          "Identify molecular species in a single frame.\n\n"
          "Parameters\n"
          "----------\n"
          "coords : list\n"
          "    Atomic coordinates, shape [n_atoms][3]\n"
          "atoms : list\n"
          "    Atomic symbols, length n_atoms\n"
          "pbc : list\n"
          "    PBC matrix, shape [3][3]\n\n"
          "Returns\n"
          "-------\n"
          "dict\n"
          "    Molecular formulas and their counts");
    
    // Utility functions
    m.def("matrix_determinant", &pbctools::matrix_determinant,
          py::arg("matrix"), "Calculate determinant of 3x3 matrix");
    
    m.def("matrix_inverse", &pbctools::matrix_inverse,
          py::arg("matrix"), "Calculate inverse of 3x3 matrix");
    
    m.def("is_orthogonal", &pbctools::is_orthogonal,
          py::arg("pbc"), "Check if PBC matrix is orthogonal");
}