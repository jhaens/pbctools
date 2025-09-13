// pbctools_cpp.cpp - Main C++ implementation for pbctools

#include <cmath>
#include <iostream>
#include <limits>
#include <algorithm>
#include <numeric>
#include <queue>
#include <unordered_set>
#include <optional>
#include <pybind11/pybind11.h>
#include "pbctools_cpp.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

namespace pbctools {

//##############
//## PBC_DIST ##
//##############


// Helper: compute inverse of a 3x3 (column-major via pointer)
static inline void invert_3x3(const float* m, float* inv) {
    float det = m[0]*(m[4]*m[8]-m[5]*m[7]) - m[1]*(m[3]*m[8]-m[5]*m[6]) + m[2]*(m[3]*m[7]-m[4]*m[6]);
    if (std::fabs(det) < 1e-12f) throw std::runtime_error("Singular PBC matrix");
    float id = 1.0f/det;
    inv[0] =  (m[4]*m[8]-m[5]*m[7]) * id;
    inv[1] = -(m[1]*m[8]-m[2]*m[7]) * id;
    inv[2] =  (m[1]*m[5]-m[2]*m[4]) * id;
    inv[3] = -(m[3]*m[8]-m[5]*m[6]) * id;
    inv[4] =  (m[0]*m[8]-m[2]*m[6]) * id;
    inv[5] = -(m[0]*m[5]-m[2]*m[3]) * id;
    inv[6] =  (m[3]*m[7]-m[4]*m[6]) * id;
    inv[7] = -(m[0]*m[7]-m[1]*m[6]) * id;
    inv[8] =  (m[0]*m[4]-m[1]*m[3]) * id;
}

pybind11::array_t<float> pbc_dist(
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> coord1,
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> coord2,
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> pbc) {

    auto b1 = coord1.request();
    auto b2 = coord2.request();
    auto bp = pbc.request();
    if (b1.ndim != 3 || b1.shape[2] != 3) throw std::runtime_error("coord1 must have shape (F,A1,3)");
    if (b2.ndim != 3 || b2.shape[2] != 3) throw std::runtime_error("coord2 must have shape (F,A2,3)");
    if (bp.ndim != 2 || bp.shape[0] != 3 || bp.shape[1] != 3) throw std::runtime_error("pbc must be (3,3)");
    if (b1.shape[0] != b2.shape[0]) throw std::runtime_error("coord1/coord2 frames mismatch");

    const float* pbc_ptr = static_cast<float*>(bp.ptr);
    bool ortho = std::fabs(pbc_ptr[1]) < 1e-7f && std::fabs(pbc_ptr[2]) < 1e-7f &&
                 std::fabs(pbc_ptr[3]) < 1e-7f && std::fabs(pbc_ptr[5]) < 1e-7f &&
                 std::fabs(pbc_ptr[6]) < 1e-7f && std::fabs(pbc_ptr[7]) < 1e-7f;

    ssize_t F = b1.shape[0];
    ssize_t A1 = b1.shape[1];
    ssize_t A2 = b2.shape[1];
    auto out = pybind11::array_t<float>({F, A1, A2, (ssize_t)3});
    auto bo = out.request();
    float* dst = static_cast<float*>(bo.ptr);
    const float* c1 = static_cast<float*>(b1.ptr);
    const float* c2 = static_cast<float*>(b2.ptr);

    float inv[9];
    if (!ortho) invert_3x3(pbc_ptr, inv);

    #ifdef WITH_OPENMP
    omp_set_num_threads(4);
    #pragma omp parallel for schedule(static)
    #endif
    for (ssize_t f=0; f<F; ++f) {
        const float* frame1 = c1 + f*A1*3;
        const float* frame2 = c2 + f*A2*3;
        float* frame_out = dst + f*A1*A2*3;
        for (ssize_t i=0;i<A1;++i) {
            const float* a1 = frame1 + i*3;
            for (ssize_t j=0;j<A2;++j) {
                const float* a2 = frame2 + j*3;
                float* v = frame_out + (i*A2 + j)*3;
                if (ortho) {
                    for (int d=0; d<3; ++d) {
                        float val = a1[d] - a2[d]; // orientation coord1 - coord2
                        float L = pbc_ptr[d*3 + d];
                        val -= L * std::round(val / L);
                        v[d] = val;
                    }
                } else {
                    float dx = a1[0]-a2[0];
                    float dy = a1[1]-a2[1];
                    float dz = a1[2]-a2[2];
                    float fx = dx*inv[0] + dy*inv[3] + dz*inv[6];
                    float fy = dx*inv[1] + dy*inv[4] + dz*inv[7];
                    float fz = dx*inv[2] + dy*inv[5] + dz*inv[8];
                    fx -= std::round(fx); fy -= std::round(fy); fz -= std::round(fz);
                    v[0] = pbc_ptr[0]*fx + pbc_ptr[1]*fy + pbc_ptr[2]*fz;
                    v[1] = pbc_ptr[3]*fx + pbc_ptr[4]*fy + pbc_ptr[5]*fz;
                    v[2] = pbc_ptr[6]*fx + pbc_ptr[7]*fy + pbc_ptr[8]*fz;
                }
            }
        }
    }
    return out;
}

//###################
//## NEXT_NEIGHBOR ##
//###################

std::pair<pybind11::array_t<int>, pybind11::array_t<float>> next_neighbor(
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> coord1,
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> coord2,
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> pbc) {
    auto dv = pbc_dist(coord1, coord2, pbc);
    auto bd = dv.request();
    ssize_t F = bd.shape[0];
    ssize_t A1 = bd.shape[1];
    ssize_t A2 = bd.shape[2];
    const float* vec = static_cast<float*>(bd.ptr);
    pybind11::array_t<int> indices({F, A1});
    pybind11::array_t<float> distances({F, A1});
    int* ip = static_cast<int*>(indices.request().ptr);
    float* dp = static_cast<float*>(distances.request().ptr);
    for (ssize_t f=0; f<F; ++f) {
        for (ssize_t i=0;i<A1;++i) {
            float best = std::numeric_limits<float>::max();
            int bestj = -1;
            for (ssize_t j=0;j<A2;++j) {
                const float* v = vec + (((f*A1)+i)*A2 + j)*3;
                float d = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
                if (d < best) { best = d; bestj = (int)j; }
            }
            ip[f*A1 + i] = bestj;
            dp[f*A1 + i] = std::sqrt(best);
        }
    }
    return {indices, distances};
}

//##########################
//## MOLECULE RECOGNITION ##
//##########################

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


pybind11::dict molecule_recognition(
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> coords,
    pybind11::list atoms,
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> pbc) {
    auto bc = coords.request();
    if (bc.ndim != 2 || bc.shape[1] != 3) throw std::runtime_error("coords must be (N,3)");
    size_t N = bc.shape[0];
    if ((size_t)atoms.size() != N) throw std::runtime_error("atoms length mismatch");
    // Expand to frames=1 for reuse
    auto coords_exp = pybind11::array_t<float>({(ssize_t)1, (ssize_t)N, (ssize_t)3});
    std::memcpy(coords_exp.mutable_data(), bc.ptr, N*3*sizeof(float));
    auto dv = pbc_dist(coords_exp, coords_exp, pbc); // shape (1,N,N,3)
    const float* vecs = static_cast<float*>(dv.request().ptr);

    // Build simple bond graph
    std::vector<std::string> atom_vec; atom_vec.reserve(N);
    for (auto a : atoms) atom_vec.push_back(a.cast<std::string>());
    std::vector<std::vector<size_t>> bonds(N);
    float cutoff = 0.833f;
    for (auto &s: atom_vec) cutoff = std::max(cutoff, get_vdw_radius(s));
    cutoff *= 1.2f;
    for (size_t i=0;i<N;++i) {
        float ri = get_vdw_radius(atom_vec[i]);
        for (size_t j=i+1;j<N;++j) {
            float rj = get_vdw_radius(atom_vec[j]);
            const float* v = vecs + ((i*N + j)*3);
            float d = std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
            float sumr = ri + rj;
            if (0.03f < d && d < 0.6f*sumr && d < cutoff) { bonds[i].push_back(j); bonds[j].push_back(i); }
        }
    }
    // Remove improper H-H bonds: if H has >1 bonds, iteratively remove the longest until one remains
    for (size_t i=0;i<N;++i) {
        if (atom_vec[i] != "H") continue;
        while (bonds[i].size() > 1) {
            // Find the longest bond from H i
            float max_d = -1.0f; size_t max_idx = 0; size_t max_pos = 0;
            for (size_t pos=0; pos<bonds[i].size(); ++pos) {
                size_t j = bonds[i][pos];
                // distance from dv at indices (i,j)
                const float* v = vecs + ((i*N + j)*3);
                float d = std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
                if (d > max_d) { max_d = d; max_idx = j; max_pos = pos; }
            }
            // Remove bond i - max_idx (both directions)
            bonds[i].erase(bonds[i].begin() + max_pos);
            auto &bj = bonds[max_idx];
            auto it = std::find(bj.begin(), bj.end(), i);
            if (it != bj.end()) bj.erase(it);
        }
    }

    // BFS components
    std::vector<int> vis(N,0); pybind11::dict out_dict;
    for (size_t i=0;i<N;++i) if(!vis[i]) {
        std::vector<size_t> comp; std::queue<size_t> q; q.push(i); vis[i]=1;
        while(!q.empty()) { auto v=q.front(); q.pop(); comp.push_back(v); for(auto nb: bonds[v]) if(!vis[nb]) {vis[nb]=1; q.push(nb);} }
        std::unordered_map<std::string,int> counts; for(auto idx: comp) counts[atom_vec[idx]]++;
        std::vector<std::string> keys; keys.reserve(counts.size()); for(auto &kv: counts) keys.push_back(kv.first);
        // Exact ordering: C first, H second, then alphabetical
        std::sort(keys.begin(), keys.end(), [](const std::string& a, const std::string& b){
            if (a == "C") return true;
            if (b == "C") return false;
            if (a == "H") return true;
            if (b == "H") return false;
            return a < b;
        });
        std::string formula; for(auto &k: keys){ formula += k; int cnt=counts[k]; if(cnt>1) formula += std::to_string(cnt);}        
        if (out_dict.contains(formula.c_str())) out_dict[formula.c_str()] = out_dict[formula.c_str()].cast<int>() + 1; else out_dict[formula.c_str()] = 1;
    }
    return out_dict;
}

} // namespace pbctools
