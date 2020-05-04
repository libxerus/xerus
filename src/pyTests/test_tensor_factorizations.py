import unittest
import xerus as xe
import numpy as np


class TestTensorFactorizations(unittest.TestCase):
	def setUp(self):
		self.U = xe.Tensor()
		self.Vt = xe.Tensor()
		self.S = xe.Tensor()
		self.R = xe.Tensor()

	def test_qr(self):
		i,j,r1,r2 = xe.indices(4)
		T = xe.Tensor.random([6,6])

		(self.U(i,r1),self.R(r1,j)) << xe.QR(T(i,j))
		self.S(i,j) << self.U(i,r1) * self.R(r1,j)
		self.assertAlmostEqual((self.S-T).frob_norm(), 0)

	def test_svd(self):
		i,j,r1,r2 = xe.indices(4)
		T = xe.Tensor.random([6,6])

		(self.U(i,r1),self.S(r1,r2),self.Vt(r2,j)) << xe.SVD(T(i,j))
		self.R(i,j) << self.U(i,r1) * self.S(r1,r2) * self.Vt(r2,j)
		self.assertAlmostEqual((self.R-T).frob_norm(), 0)

