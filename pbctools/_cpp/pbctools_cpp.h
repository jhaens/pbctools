// pbctools_cpp.h - Header file for pbctools C++ implementation

#ifndef PBCTOOLS_CPP_H
#define PBCTOOLS_CPP_H

#include <vector>
#include <string>
#include <unordered_map>

namespace pbctools {

// Type aliases for clarity
using Coordinate = std::vector<float>;
using Frame = std::vector<Coordinate>;
using Trajectory = std::vector<Frame>;
using PBCMatrix = std::vector<std::vector<float>>;

// Core PBC distance calculation for multiple frames
// Input: coord1 [n_frames][n_atoms1][3], coord2 [n_frames][n_atoms2][3], pbc [3][3]
// Output: distance_vectors [n_frames][n_atoms1][n_atoms2][3]
std::vector<std::vector<std::vector<Coordinate>>> pbc_dist(
    const Trajectory& coord1, 
    const Trajectory& coord2, 
    const PBCMatrix& pbc
);

// Single frame PBC distance calculation (helper function)
std::vector<std::vector<Coordinate>> pbc_dist_frame(
    const Frame& coord1, 
    const Frame& coord2, 
    const PBCMatrix& pbc
);

// Multi-frame nearest neighbor detection
// Input: coord1 [n_frames][n_atoms1][3], coord2 [n_frames][n_atoms2][3], pbc [3][3]
// Output: indices [n_frames][n_atoms1], distances [n_frames][n_atoms1]
std::pair<std::vector<std::vector<int>>, 
          std::vector<std::vector<float>>> next_neighbor(
    const Trajectory& coord1,
    const Trajectory& coord2,
    const PBCMatrix& pbc
);

// Molecule recognition for single frame
// Input: coords [n_atoms][3], atom_names [n_atoms], pbc [3][3]
// Output: molecular formula counts
std::unordered_map<std::string, int> molecule_recognition(
    const Frame& coords,
    const std::vector<std::string>& atoms,
    const PBCMatrix& pbc
);

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