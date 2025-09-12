// pbctools_cpp.h - Header file for pbctools C++ implementation

#ifndef PBCTOOLS_CPP_H
#define PBCTOOLS_CPP_H

#include <vector>
#include <string>
#include <unordered_map>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace pbctools {

// Type aliases for clarity
using Coordinate = std::vector<float>;
using Frame = std::vector<Coordinate>;
using Trajectory = std::vector<Frame>;
using PBCMatrix = std::vector<std::vector<float>>;

// Optimized interfaces using pybind11 numpy arrays (C-contiguous float32 expected)
py::array_t<float> pbc_dist(
    py::array_t<float, py::array::c_style | py::array::forcecast> coord1,
    py::array_t<float, py::array::c_style | py::array::forcecast> coord2,
    py::array_t<float, py::array::c_style | py::array::forcecast> pbc);

std::pair<py::array_t<int>, py::array_t<float>> next_neighbor(
    py::array_t<float, py::array::c_style | py::array::forcecast> coord1,
    py::array_t<float, py::array::c_style | py::array::forcecast> coord2,
    py::array_t<float, py::array::c_style | py::array::forcecast> pbc);

py::dict molecule_recognition(
    py::array_t<float, py::array::c_style | py::array::forcecast> coords,
    py::list atoms,
    py::array_t<float, py::array::c_style | py::array::forcecast> pbc);

// Utility functions
bool is_orthogonal(const PBCMatrix& pbc);
PBCMatrix matrix_inverse(const PBCMatrix& matrix);
float matrix_determinant(const PBCMatrix& matrix);
std::vector<float> matrix_vector_multiply(const PBCMatrix& matrix, const Coordinate& vec);
std::vector<float> vector_matrix_multiply(const Coordinate& vec, const PBCMatrix& matrix);

// Bond detection for molecule recognition
bool is_bonded(const Coordinate& atom1, const Coordinate& atom2,
               const std::string& element1, const std::string& element2,
               const PBCMatrix& pbc);

// Van der Waals radius lookup
float get_vdw_radius(const std::string& element);

} // namespace pbctools

#endif // PBCTOOLS_CPP_H