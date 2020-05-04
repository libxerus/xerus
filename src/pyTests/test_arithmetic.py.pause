import unittest
from itertools import product
ranges = lambda *args: product(*[range(arg) for arg in args])
import xerus as xe
import numpy as np


class TestArithmetic(unittest.TestCase):
	def setUp(self):
		self.dimension = [10, 15, 20]
		self.ranks = [1, 8, 4, 1]

	def test_tensor_linear_combination(self):
		ten_a = xe.Tensor.ones(self.dimension)
		ten_b = xe.Tensor.ones(self.dimension) * 4
		ten_c = ten_a + ten_b
		for (i, j, k) in ranges(*self.dimension):
			self.assertAlmostEqual(ten_c[i, j, k], 5)
