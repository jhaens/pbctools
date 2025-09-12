// pbctools_cpp.cpp - Main C++ implementation for pbctools

#include "pbctools_cpp.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <algorithm>
#include <numeric>
#include <queue>
#include <unordered_set>
#include <optional>

#ifdef WITH_OPENMP
#include <omp.h>
omp_set_num_threads(4);
#endif

namespace pbctools {

//#######################
//## MATRIX OPERATIONS ##
//#######################

float matrix_determinant(const PBCMatrix& matrix) {
    return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
           matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
           matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
}

PBCMatrix matrix_inverse(const PBCMatrix& matrix) {
    float det = matrix_determinant(matrix);
    if (std::abs(det) < 1e-10f) {
        throw std::runtime_error("Matrix is singular and cannot be inverted.");
    }
    
    PBCMatrix inverse(3, std::vector<float>(3, 0.0f));
    inverse[0][0] = (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) / det;
    inverse[0][1] = (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) / det;
    inverse[0][2] = (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) / det;
    inverse[1][0] = (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) / det;
    inverse[1][1] = (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) / det;
    inverse[1][2] = (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) / det;
    inverse[2][0] = (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) / det;
    inverse[2][1] = (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) / det;
    inverse[2][2] = (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) / det;
    return inverse;
}

std::vector<float> matrix_vector_multiply(const PBCMatrix& matrix, const Coordinate& vec) {
    std::vector<float> result(3, 0.0f);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}

std::vector<float> vector_matrix_multiply(const Coordinate& vec, const PBCMatrix& matrix) {
    std::vector<float> result(3, 0.0f);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i] += vec[j] * matrix[j][i];
        }
    }
    return result;
}

bool is_orthogonal(const PBCMatrix& pbc) {
    const float tolerance = 1e-6f;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i != j && std::abs(pbc[i][j]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

//#######################
//## PBC DIST FUNCTION ##
//#######################

std::vector<std::vector<Coordinate>> pbc_dist_frame(
    const Frame& coord1, 
    const Frame& coord2, 
    const PBCMatrix& pbc) {
    
    bool is_ortho_pbc = is_orthogonal(pbc);
    std::vector<std::vector<Coordinate>> distance_vectors(
        coord1.size(), 
        std::vector<Coordinate>(coord2.size(), Coordinate(3, 0.0f))
    );
    
    if (is_ortho_pbc) {
        // Orthogonal PBC calculation
        for (size_t i = 0; i < coord1.size(); ++i) {
            for (size_t j = 0; j < coord2.size(); ++j) {
                Coordinate dist_vec(3);
                for (int dim = 0; dim < 3; ++dim) {
                    dist_vec[dim] = coord1[i][dim] - coord2[j][dim];
                    // Apply PBC
                    dist_vec[dim] = dist_vec[dim] - pbc[dim][dim] * 
                                   std::round(dist_vec[dim] / pbc[dim][dim]);
                }
                distance_vectors[i][j] = dist_vec;
            }
        }
    } else {
        // Non-orthogonal PBC calculation
        PBCMatrix inv_pbc = matrix_inverse(pbc);
        
        // Transform coordinates to fractional
        std::vector<Coordinate> frac_coord1(coord1.size());
        std::vector<Coordinate> frac_coord2(coord2.size());
        
        for (size_t i = 0; i < coord1.size(); ++i) {
            frac_coord1[i] = vector_matrix_multiply(coord1[i], inv_pbc);
        }
        for (size_t i = 0; i < coord2.size(); ++i) {
            frac_coord2[i] = vector_matrix_multiply(coord2[i], inv_pbc);
        }
        
        for (size_t i = 0; i < coord1.size(); ++i) {
            for (size_t j = 0; j < coord2.size(); ++j) {
                Coordinate frac_diff(3);
                for (int dim = 0; dim < 3; ++dim) {
                    frac_diff[dim] = frac_coord1[i][dim] - frac_coord2[j][dim];
                    frac_diff[dim] = frac_diff[dim] - std::round(frac_diff[dim]);
                }
                // Transform back to real coordinates
                distance_vectors[i][j] = matrix_vector_multiply(pbc, frac_diff);
            }
        }
    }
    
    return distance_vectors;
}

std::vector<std::vector<std::vector<Coordinate>>> pbc_dist(
    const Trajectory& coord1, 
    const Trajectory& coord2, 
    const PBCMatrix& pbc) {
    
    size_t n_frames = coord1.size();
    std::vector<std::vector<std::vector<Coordinate>>> result(n_frames);
    
#ifdef WITH_OPENMP
    #pragma omp parallel for
#endif
    for (size_t frame = 0; frame < n_frames; ++frame) {
        result[frame] = pbc_dist_frame(coord1[frame], coord2[frame], pbc);
    }
    
    return result;
}

//############################
//## NEXT NEIGHBOR FUNCTION ##
//############################

std::pair<std::vector<std::vector<int>>, 
          std::vector<std::vector<float>>> next_neighbor(
    const Trajectory& coord1,
    const Trajectory& coord2,
    const PBCMatrix& pbc) {
    
    size_t n_frames = coord1.size();
    std::vector<std::vector<int>> indices(n_frames);
    std::vector<std::vector<float>> distances(n_frames);
    
#ifdef WITH_OPENMP
    #pragma omp parallel for
#endif
    for (size_t frame = 0; frame < n_frames; ++frame) {
        auto dist_vectors = pbc_dist_frame(coord1[frame], coord2[frame], pbc);
        
        indices[frame].resize(coord1[frame].size());
        distances[frame].resize(coord1[frame].size());
        
        for (size_t i = 0; i < coord1[frame].size(); ++i) {
            float min_dist = std::numeric_limits<float>::max();
            int min_idx = -1;
            
            for (size_t j = 0; j < coord2[frame].size(); ++j) {
                // Calculate distance magnitude
                float dist = 0.0f;
                for (int k = 0; k < 3; ++k) {
                    dist += dist_vectors[i][j][k] * dist_vectors[i][j][k];
                }
                dist = std::sqrt(dist);
                
                if (dist < min_dist) {
                    min_dist = dist;
                    min_idx = static_cast<int>(j);
                }
            }
            
            indices[frame][i] = min_idx;
            distances[frame][i] = min_dist;
        }
    }
    
    return std::make_pair(indices, distances);
}

//###################################
//## MOLECULE RECOGNITION FUNCTION ##
//###################################

// Van der Waals radius lookup
float get_vdw_radius(const std::string& element) {
    static const std::unordered_map<std::string, float> vdw_radii = {
        {"H", 1.20f}, {"He", 1.40f}, {"Li", 1.82f}, {"Be", 1.53f}, {"B", 1.92f},
        {"C", 1.70f}, {"N", 1.55f}, {"O", 1.52f}, {"F", 1.47f}, {"Ne", 1.54f},
        {"Na", 2.27f}, {"Mg", 1.73f}, {"Al", 1.84f}, {"Si", 2.10f}, {"P", 1.80f},
        {"S", 1.80f}, {"Cl", 1.75f}, {"Ar", 1.88f}
    };
    
    std::string normalized_element = element;
    if (normalized_element.length() >= 1) {
        normalized_element[0] = std::toupper(normalized_element[0]);
    }
    if (normalized_element.length() >= 2) {
        normalized_element[1] = std::tolower(normalized_element[1]);
    }
    
    auto it = vdw_radii.find(normalized_element);
    if (it != vdw_radii.end()) {
        return it->second;
    }
    return 2.0f; // Default radius for unknown elements
}

std::unordered_map<std::string, int> molecule_recognition(
    const Frame& coords,
    const std::vector<std::string>& atoms,
    const PBCMatrix& pbc) {
    
    size_t num_atoms = atoms.size();
    std::vector<std::vector<size_t>> bond_graph(num_atoms);
    
    // Calculate distance matrix for bond detection
    auto dist_vectors = pbc_dist_frame(coords, coords, pbc);
    
    // Find maximum radius for cutoff
    float cutoff = 0.833f;
    for (size_t i = 0; i < num_atoms; ++i) {
        cutoff = std::max(cutoff, get_vdw_radius(atoms[i]));
    }
    cutoff *= 1.2f;
    
    // Detect bonds
    for (size_t i = 0; i < num_atoms; ++i) {
        float i_radius = get_vdw_radius(atoms[i]);
        
        for (size_t j = i + 1; j < num_atoms; ++j) {
            float j_radius = get_vdw_radius(atoms[j]);
            
            // Calculate distance magnitude
            float dist = 0.0f;
            for (int k = 0; k < 3; ++k) {
                dist += dist_vectors[i][j][k] * dist_vectors[i][j][k];
            }
            dist = std::sqrt(dist);
            
            float radii_sum = i_radius + j_radius;
            
            // VMD bond criteria
            if (0.03f < dist && dist < 0.6f * radii_sum && dist < cutoff) {
                bond_graph[i].push_back(j);
                bond_graph[j].push_back(i);
            }
        }
    }
    
    // Remove improper H-H bonds
    for (size_t i = 0; i < num_atoms; ++i) {
        if (atoms[i] != "H") continue;
        
        while (bond_graph[i].size() > 1) {
            // Find the longest bond to remove
            float max_distance = 0.0f;
            size_t max_idx = 0;
            
            for (size_t j : bond_graph[i]) {
                float dist = 0.0f;
                for (int k = 0; k < 3; ++k) {
                    dist += dist_vectors[i][j][k] * dist_vectors[i][j][k];
                }
                dist = std::sqrt(dist);
                
                if (dist > max_distance) {
                    max_distance = dist;
                    max_idx = j;
                }
            }
            
            // Remove bond
            bond_graph[i].erase(
                std::find(bond_graph[i].begin(), bond_graph[i].end(), max_idx)
            );
            bond_graph[max_idx].erase(
                std::find(bond_graph[max_idx].begin(), bond_graph[max_idx].end(), i)
            );
        }
    }
    
    // Find connected components (molecules) using BFS
    std::vector<bool> visited(num_atoms, false);
    std::vector<std::vector<size_t>> molecules;
    
    for (size_t i = 0; i < num_atoms; ++i) {
        if (!visited[i]) {
            std::vector<size_t> molecule;
            std::queue<size_t> queue;
            queue.push(i);
            visited[i] = true;
            
            while (!queue.empty()) {
                size_t current = queue.front();
                queue.pop();
                molecule.push_back(current);
                
                for (size_t neighbor : bond_graph[current]) {
                    if (!visited[neighbor]) {
                        queue.push(neighbor);
                        visited[neighbor] = true;
                    }
                }
            }
            molecules.push_back(molecule);
        }
    }
    
    // Create molecular formulas and count them
    std::unordered_map<std::string, int> molecular_formulas;
    
    for (const auto& molecule : molecules) {
        std::unordered_map<std::string, int> atom_counts;
        
        for (size_t atom_idx : molecule) {
            atom_counts[atoms[atom_idx]]++;
        }
        
        // Generate formula string (C first, H second, then alphabetically)
        std::string formula;
        std::vector<std::string> sorted_atoms;
        
        for (const auto& [atom_type, count] : atom_counts) {
            sorted_atoms.push_back(atom_type);
        }
        
        std::sort(sorted_atoms.begin(), sorted_atoms.end(), [](const std::string& a, const std::string& b) {
            if (a == "C") return true;
            if (b == "C") return false;
            if (a == "H") return true;
            if (b == "H") return false;
            return a < b;
        });
        
        for (const std::string& atom_type : sorted_atoms) {
            formula += atom_type;
            if (atom_counts[atom_type] > 1) {
                formula += std::to_string(atom_counts[atom_type]);
            }
        }
        
        molecular_formulas[formula]++;
    }
    
    return molecular_formulas;
}

} // namespace pbctools