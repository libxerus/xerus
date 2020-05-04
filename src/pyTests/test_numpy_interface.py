# import unittest
# from itertools import product
# ranges = lambda *args: product(*[range(arg) for arg in args])
# import xerus as xe
# import numpy as np


# class TestNumpyInterface(unittest.TestCase):
#     def setUp(self):
#         self.dimension = [10, 15, 20]
#         self.ranks = [1, 8, 4, 1]

#     def test_to_ndarray(self):
#         ten = xe.TTTensor.random(self.dimension, self.ranks[1:-1])
#         for lia in range(ten.degree()):
#             comp_ten = ten.get_component(lia)
#             comp_dim = comp_ten.dimensions
#             comp_nd = comp_ten.to_ndarray()
#             for (i, j, k) in ranges(*comp_dim):
#                 self.assertEqual(comp_ten[[i,j,k]], comp_nd[i,j,k])

#     def test_from_buffer(self):
#         arr = np.random.randn(50,50)
#         ten = xe.Tensor.from_buffer(arr)  #TO
#         for lia in range(arr.shape[0]):
#             for lib in range(arr.shape[1]):
#                 self.assertEqual(ten[[lia, lib]], arr[lia,lib])


# if __name__ == '__main__':
#     unittest.main()



import unittest
import numpy as np
import xerus as xe


def generate_random_tensors(num_tests, max_order=4, max_dimension=10, random=None):
	orders = random.randint(low=1, high=max_order+1, size=num_tests)
	dimensions = [random.randint(low=1, high=max_dimension, size=order) for order in orders]
	for dim in dimensions:
		yield xe.Tensor.random(dim.tolist())

def generate_random_ndarrays(num_tests, max_order=4, max_dimension=10, random=None):
	orders = random.randint(low=1, high=max_order+1, size=num_tests)
	dimensions = [random.randint(low=1, high=max_dimension, size=order) for order in orders]
	for dim in dimensions:
		yield random.randn(*dim)


list2str = lambda ls: "-".join(map(str, ls))


def __test_from_buffer_left(A):
	name = "test_frob_buffer_left_{}".format(list2str(A.shape))

	def test(self):
		B = np.array(xe.Tensor.from_buffer(A))
		self.assertTrue(np.all(A == B))

	return name, test


def __test_from_buffer_right(A):
	name = "test_frob_buffer_right_{}".format(list2str(A.dimensions))

	def test(self):
		B = xe.Tensor.from_buffer(np.array(A))
		self.assertLessEqual(xe.frob_norm(A - B), 1e-12)
		# self.assertEqual(A, B)

	return name, test


def build_TestNumpyInterface(seed, num_tests):
	random = np.random.RandomState(seed)
	odir = dict(__test_from_buffer_left(t) for t in generate_random_ndarrays(num_tests//2, random=random))
	odir.update(dict(__test_from_buffer_right(t) for t in generate_random_tensors(num_tests//2 + num_tests%2, random=random)))
	return type("TestNumpyInterface", (unittest.TestCase,), odir)


TestNumpyInterface = build_TestNumpyInterface(0, 20)
