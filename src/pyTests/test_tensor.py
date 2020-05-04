import sys, unittest
from contextlib import contextmanager
from itertools import product
ranges = lambda *args: product(*[range(arg) for arg in args])
import xerus as xe
import numpy as np


# class assertNotRaises(object):
#     def __init__(self, exc_type=Exception, msg=None):
#         self.exc_type = exc_type
#         self.msg = " : " + msg if msg is not None else ""
#         self.exception = None
#     def __enter__(self): return self
#     def __exit__(self, exc_type, exc_value, traceback):
#         self.exception = exc_value
#         if isinstance(exc_value, self.exc_type):
#             text = "{} raised" + self.msg
#             raise AssertionError(text.format(exc_type.__name__)) from None


class TestTensor(unittest.TestCase):
	def setUp(self):
		self.dimension = [10, 15, 20]
		self.ranks = [1, 8, 4, 1]

	def test_existence(self):
		xe.Representation
		xe.Representation.Dense
		xe.Representation.Sparse

		xe.Initialisation
		xe.Initialisation.Zero
		# xe.Initialisation.None  #TODO: syntax error

		xe.Tensor


	@contextmanager
	def assertNotRaises(self, exc_type=Exception, msg=None):
		try:
			yield None
		except exc_type as e:
			text = "{} raised" + (" : "+msg if msg is not None else "")
			if sys.version_info >= (3, 0):
				raise self.failureException(text.format(e.__class__.__name__)) from None
			else:
				raise self.failureException(text.format(e.__class__.__name__))

	def test_construction(self):
		with self.assertNotRaises(msg="Order-0 TT"):
			A = xe.TTTensor([])
		with self.assertNotRaises(msg="Order-1 TT"):
			A = xe.TTTensor([6])
		with self.assertNotRaises(msg="Order-1 random TT"):
			A = xe.TTTensor.random([6], [])
		with self.assertNotRaises(msg="Order-0 HT"):
			A = xe.HTTensor([])
		with self.assertNotRaises(msg="Order-1 HT"):
			A = xe.HTTensor([6])
		with self.assertNotRaises(msg="Order-1 random HT"):
			A = xe.HTTensor.random([6], [])

	def test_core_move(self):
		with self.assertNotRaises():
			A = xe.HTTensor([6])
			A.move_core(0)


	def test_create_tensors_simple(self):
		xe.Tensor()
		xe.Tensor([3,3,3])
		xe.Tensor([3,3,3], repr=xe.Representation.Dense)
		xe.Tensor([3,3,3], repr=xe.Representation.Sparse)
		xe.Tensor([3,3,3], init=xe.Initialisation.Zero)
		# xe.Tensor([3,3,3], init=xe.Initialisation.None)
		xe.Tensor([3,3,3], init=xe.Initialisation.Zero, repr=xe.Representation.Dense)
		xe.Tensor([3,3,3], init=xe.Initialisation.Zero, repr=xe.Representation.Sparse)
		# xe.Tensor([3,3,3], init=xe.Initialisation.None, repr=xe.Representation.Dense)
		# xe.Tensor([3,3,3], init=xe.Initialisation.None, repr=xe.Representation.Sparse)

		xe.Tensor.from_function([3,3,3], lambda ijk: ijk[0]+ijk[1]+ijk[2])
		a = xe.Tensor.from_buffer(np.random.rand(3,3,3))
		xe.Tensor(a)

		a = xe.Tensor()
		xe.Tensor(a)

		xe.Tensor(xe.TTTensor([2]))

		# xe.Tensor(tensor_network)
		a = xe.Tensor.from_buffer(np.random.randn(6))

	def test_create_tensors(self):
		ten_init = xe.Tensor()

		ten_init_dim = xe.Tensor([3,4,5])
		self.assertAlmostEqual(ten_init_dim[0,0,0], 0)
		self.assertAlmostEqual(ten_init_dim.frob_norm(), 0)

		def fun(ind):
			return ind[0] + ind[1] + ind[2]
		ten_init_fun = xe.Tensor.from_function([7,7,7], fun)
		print(ten_init_fun.one_norm())
		self.assertAlmostEqual(ten_init_fun[3,4,5], 12)
		self.assertAlmostEqual(ten_init_fun.order(), 3)
		self.assertAlmostEqual(ten_init_fun.frob_norm(), np.sqrt(31899))
		self.assertAlmostEqual(ten_init_fun.one_norm(),3087)

		ten_init_ones = xe.Tensor.ones([2,3,2])
		self.assertAlmostEqual(ten_init_ones.frob_norm(), np.sqrt(12))

		ten_init_random = xe.Tensor.random([2,3,2])
		self.assertGreater(ten_init_random.frob_norm(), 0)

		ten_init_id = xe.Tensor.identity([2,3,2,3])
		self.assertAlmostEqual(ten_init_id.frob_norm(), np.sqrt(6))

		ten_init_kron = xe.Tensor.kronecker([2,3,2,3])
		self.assertAlmostEqual(ten_init_kron[1,1,1,1], 1)
		self.assertAlmostEqual(ten_init_kron[1,1,1,0], 0)

		ten_init_dirac = xe.Tensor.dirac([2,3,2,3],[1,1,1,1])
		self.assertAlmostEqual(ten_init_dirac[1,1,1,1], 1)
		self.assertAlmostEqual(ten_init_dirac[1,1,1,0], 0)

	def test_cast_tensor_0(self):
		a = xe.Tensor([])
		self.assertEqual(float(a), 0)

	def test_cast_tensor_1(self):
		a = xe.Tensor.ones([])
		self.assertEqual(float(a), 1)

	def test_reinterpret_dimensions(self):
		T = xe.Tensor.random([2,3,4,5])
		T_new = T.dense_copy()
		T.reinterpret_dimensions([6,20])
		self.assertAlmostEqual(T[4,6], T_new[1,1,1,1])
