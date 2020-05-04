import unittest
from itertools import product
ranges = lambda *args: product(*[range(arg) for arg in args])
import xerus as xe
import numpy as np


class TestTensorArithmetic(unittest.TestCase):
	def setUp(self):
		self.dimension = [10, 15, 20]
		self.ranks = [1, 8, 4, 1]

	def test_contractions(self):
		A = xe.Tensor.random([10,10])
		B = xe.TensorNetwork()
		i,j,k = xe.indices(3)
		C = xe.TensorNetwork(xe.Tensor.ones([10]))
		B(i,j,k) << A(i,j) * C(k)
		B(i,j,k) << A(i,j) * xe.TensorNetwork(xe.Tensor.ones([10]))(k)
		B(i,j,k) << A(i,j) * xe.Tensor.ones([10])(k)
		xe.Tensor(B)

	def test_tensor_linear_combination(self):
		ten_a = xe.Tensor.ones(self.dimension)
		ten_b = xe.Tensor.ones(self.dimension) * 4
		ten_c = ten_a + ten_b
		for (i, j, k) in ranges(*self.dimension):
			self.assertAlmostEqual(ten_c[i, j, k], 5)

	def test_simple_multiplication(self):
		ii,jj,kk = xe.indices(3)
		ten_a = xe.Tensor.ones([2,2])
		ten_b = xe.Tensor.ones([2,2]) * 2
		ten_c = xe.Tensor()

		ten_c(ii/2,jj/2) << ten_a(ii/2,kk/2) * ten_b(kk/2,jj/2) 
		self.assertAlmostEqual(ten_c.frob_norm(), 8)

		ten_c(ii&1,jj&1) << ten_a(ii&1,kk&1) * ten_b(kk&1,jj&1) 
		self.assertAlmostEqual(ten_c.frob_norm(), 8)

		ten_c(ii^1,jj^1) << ten_a(ii^1,kk^1) * ten_b(kk^1,jj^1) 
		self.assertAlmostEqual(ten_c.frob_norm(), 8)
