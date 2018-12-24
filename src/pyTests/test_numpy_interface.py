import unittest
from itertools import product
ranges = lambda *args: product(*[range(arg) for arg in args])
import xerus as xe
import numpy as np


class TestNumpyInterface(unittest.TestCase):
    def setUp(self):
        self.dimension = [10, 15, 20]
        self.ranks = [1, 8, 4, 1]

    def test_to_ndarray(self):
        ten = xe.TTTensor.random(self.dimension, self.ranks[1:-1])
        for lia in range(ten.degree()):
            comp_ten = ten.get_component(lia)
            comp_dim = comp_ten.dimensions
            comp_nd = comp_ten.to_ndarray()
            for (i, j, k) in ranges(*comp_dim):
                self.assertEqual(comp_ten[[i,j,k]], comp_nd[i,j,k])

    def test_from_ndarray(self):
        arr = np.random.randn(50,50)
        ten = xe.Tensor.from_ndarray(arr)
        for lia in range(arr.shape[0]):
            for lib in range(arr.shape[1]):
                self.assertEqual(ten[[lia, lib]], arr[lia,lib])


if __name__ == '__main__':
    unittest.main()
