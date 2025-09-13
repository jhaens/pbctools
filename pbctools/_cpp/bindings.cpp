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
              "Optimized PBC distance vectors (float32) using raw numpy buffers.\n"
              "coord1: (F,A1,3) float32, coord2: (F,A2,3) float32, pbc: (3,3) float32 -> (F,A1,A2,3)");

  m.def("next_neighbor", &pbctools::next_neighbor,
          py::arg("coord1"), py::arg("coord2"), py::arg("pbc"),
          "Nearest neighbors (indices int32, distances float32). coord1:(F,A1,3) coord2:(F,A2,3).");

  m.def("molecule_recognition", &pbctools::molecule_recognition,
          py::arg("coords"), py::arg("atoms"), py::arg("pbc"),
          "Identify molecular species in single frame (coords:(N,3) float32). Returns dict formula->count.");
    
      m.def("next_neighbor", &pbctools::next_neighbor,
              py::arg("coord1"), py::arg("coord2"), py::arg("pbc"),
              "Nearest neighbors (indices int32, distances float32). coord1:(F,A1,3) coord2:(F,A2,3).");
    
      m.def("molecule_recognition", &pbctools::molecule_recognition,
              py::arg("coords"), py::arg("atoms"), py::arg("pbc"),
              "Identify molecular species in single frame (coords:(N,3) float32). Returns dict formula->count.");
    
//     // Utility functions
//     m.def("matrix_determinant", &pbctools::matrix_determinant,
//           py::arg("matrix"), "Calculate determinant of 3x3 matrix");
    
//     m.def("matrix_inverse", &pbctools::matrix_inverse,
//           py::arg("matrix"), "Calculate inverse of 3x3 matrix");
    
//     m.def("is_orthogonal", &pbctools::is_orthogonal,
//           py::arg("pbc"), "Check if PBC matrix is orthogonal");
}