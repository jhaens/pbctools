import pytest
import numpy as np
from pbctools import pbc_dist, next_neighbor, molecule_recognition

class TestPbcDist:
    def test_basic_functionality(self):
        """Test basic pbc_dist functionality."""
        coord1 = np.random.rand(10, 5, 3).astype(np.float32)  
        coord2 = np.random.rand(10, 3, 3).astype(np.float32)
        pbc = np.eye(3, dtype=np.float32) * 20.0
        
        result = pbc_dist(coord1, coord2, pbc)
        
        assert result.shape == (10, 5, 3, 3)
        assert result.dtype == np.float32
    
    def test_orthogonal_pbc(self):
        """Test with orthogonal PBC (cubic box)."""
        coord1 = np.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]], dtype=np.float32)
        coord2 = np.array([[[19.0, 0.0, 0.0]]], dtype=np.float32)
        pbc = np.eye(3, dtype=np.float32) * 20.0
        
        result = pbc_dist(coord1, coord2, pbc)
        
        # Should give minimum image distance
        expected_dist_1 = 1.0  # 0 - 19 with PBC = 1
        expected_dist_2 = 9.0   # 10 - 19 with PBC = 9
        
        print(result)
        assert abs(result[0, 0, 0, 0]) - abs(expected_dist_1) < 1e-6
        assert abs(result[0, 1, 0, 0]) - abs(expected_dist_2) < 1e-6

    def test_input_validation(self):
        """Test input validation."""
        with pytest.raises(ValueError):
            pbc_dist(np.random.rand(2, 3), np.random.rand(5, 3, 3), np.eye(3))
        
        with pytest.raises(ValueError):
            pbc_dist(np.random.rand(5, 3, 3), np.random.rand(3, 3, 3), np.eye(2))


class TestNextNeighbor:
    def test_basic_functionality(self):
        """Test basic next_neighbor functionality."""
        coord1 = np.random.rand(10, 5, 3).astype(np.float32)
        coord2 = np.random.rand(10, 8, 3).astype(np.float32)
        pbc = np.eye(3, dtype=np.float32) * 20.0
        
        indices, distances = next_neighbor(coord1, coord2, pbc)
        
        assert indices.shape == (10, 5)
        assert distances.shape == (10, 5)
        assert indices.dtype == np.int32
        assert distances.dtype == np.float32
    
    def test_nearest_selection(self):
        """Test that nearest neighbor is correctly selected."""
        # Set up simple case where nearest neighbor is obvious
        coord1 = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
        coord2 = np.array([[[1.0, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 0.0, 0.0]]], dtype=np.float32)
        pbc = np.eye(3, dtype=np.float32) * 20.0
        
        indices, distances = next_neighbor(coord1, coord2, pbc)
        
        assert indices[0, 0] == 1  # Should select index 1 (closest at 0.1 distance)
        assert abs(distances[0, 0] - 0.1) < 1e-6


class TestMoleculeRecognition:
    def test_water_molecule(self):
        """Test recognition of water molecule."""
        # Simple water molecule geometry
        coords = np.array([
            [0.0, 0.0, 0.0],      # O
            [0.96, 0.0, 0.0],     # H1 
            [-0.24, 0.93, 0.0]    # H2
        ], dtype=np.float32)
        atoms = np.array(['O', 'H', 'H'])
        pbc = np.eye(3, dtype=np.float32) * 10.0
        
        result = molecule_recognition(coords, atoms, pbc)
        
        assert 'H2O' in result
        assert result['H2O'] == 1
    
    def test_multiple_molecules(self):
        """Test recognition of multiple molecules."""
        # Two separate water molecules
        coords = np.array([
            [0.0, 0.0, 0.0],      # O1
            [0.96, 0.0, 0.0],     # H1 
            [-0.24, 0.93, 0.0],   # H2
            [5.0, 0.0, 0.0],      # O2
            [5.96, 0.0, 0.0],     # H3
            [4.76, 0.93, 0.0]     # H4
        ], dtype=np.float32)
        atoms = np.array(['O', 'H', 'H', 'O', 'H', 'H'])
        pbc = np.eye(3, dtype=np.float32) * 10.0
        
        result = molecule_recognition(coords, atoms, pbc)
        
        assert 'H2O' in result
        assert result['H2O'] == 2
    
    def test_hydroxide_ion(self):
        """Test recognition of hydroxide ion."""
        coords = np.array([
            [0.0, 0.0, 0.0],      # O
            [0.96, 0.0, 0.0],     # H
        ], dtype=np.float32)
        atoms = np.array(['O', 'H'])
        pbc = np.eye(3, dtype=np.float32) * 10.0
        
        result = molecule_recognition(coords, atoms, pbc)
        
        assert 'HO' in result
        assert result['HO'] == 1


if __name__ == "__main__":
    pytest.main([__file__])