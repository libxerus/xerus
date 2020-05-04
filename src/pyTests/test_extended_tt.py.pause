from __future__ import division
import unittest
import itertools as _iter
import xerus as xe
import numpy as np
np.random.seed(1337)


ranges = lambda *args: _iter.product(*[range(arg) for arg in args])


class TestExtendedTT(unittest.TestCase):
	def setUp(self):
		self.dimension = [4, 6, 8]
		self.ranks = [1, 8, 5, 1]

	def test_from_function_set_component(self):
		tt = xe.TTTensor(self.dimension)
		arrs = []
		for i in range(len(self.dimension)):
			shape = [self.ranks[i], self.dimension[i], self.ranks[i+1]]
			arr = np.random.randn(*shape)
			x = xe.Tensor.from_function(shape, lambda x: arr[tuple(x)])
			tt.set_component(i, x)
			arrs.append(arr)
		for i in range(len(self.dimension)):
			self.assertTrue(np.all(arrs[i] == tt.get_component(i).to_ndarray()))

	def test_eval_hermite(self):
		from numpy.polynomial.hermite_e import hermeval
		basis = xe.PolynomBasis.Hermite

		# TODO: check with numpy hermite...

		tt = xe.TTTensor(self.dimension)
		arrs = []
		for i in range(len(self.dimension)):
			shape = [self.ranks[i], self.dimension[i], self.ranks[i+1]]
			arr = np.random.randn(*shape)
			x = xe.Tensor.from_function(shape, lambda x: arr[tuple(x)])
			tt.set_component(i, x)
			arrs.append(arr)

		for (i, j, k) in ranges(*self.dimension):
			x = xe.uq_tt_evaluate(tt, [j, k], basis)
			x = x[[i]]
			# loc_extt = extt([i, j, k])
			# self.assertLessEqual(np.abs((loc_extt - loc_xett)/loc_extt), 1e-10)
