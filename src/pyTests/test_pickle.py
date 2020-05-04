import unittest
import numpy as np
import xerus as xe
try:
	# python2
	# Here cPickle has to be used instead of pickle and a protocol version newer 2.0 has to be used.
	# From the documentation:
	#
	#    Note that only the cPickle module is supported on Python 2.7.
	#    The second argument to dumps is also crucial: it selects the pickle protocol version 2,
	#    since the older version 1 is not supported. Newer versions are also fine - for instance,
	#    specify -1 to always use the latest available version. 
	#
	#    Beware: failure to follow these instructions will cause important pybind11 memory 
	#    allocation routines to be skipped during unpickling, which will likely lead to memory 
	#    corruption and/or segmentation faults.
	import cPickle
	assert '2.0' in cPickle.compatible_formats
	dumps = lambda o: cPickle.dumps(o, protocol=-1)
	loads = cPickle.loads
except ImportError:
	# python3
	from pickle import dumps, loads


def generate_random_tensors(num_tests, max_order=4, max_dimension=10, random=None):
	orders = random.randint(low=1, high=max_order+1, size=num_tests)
	dimensions = [random.randint(low=1, high=max_dimension, size=order) for order in orders]
	for dim in dimensions:
		yield xe.Tensor.random(dim.tolist())


def generate_random_tttensors(num_tests, max_order=10, max_dimension=10, max_rank=4, random=None):
	orders = random.randint(low=1, high=max_order+1, size=num_tests)
	dimensions = [random.randint(low=1, high=max_dimension, size=order) for order in orders]
	ranks = [random.randint(low=1, high=max_rank, size=(order-1)) for order in orders]
	for dim, rk in zip(dimensions, ranks):
		yield xe.TTTensor.random(dim.tolist(), rk.tolist())


def generate_random_httensors(num_tests, max_order=10, max_dimension=10, max_rank=4, random=None):
	# orders = random.randint(low=1, high=max_order+1, size=num_tests)
	orders = random.randint(low=2, high=max_order+1, size=num_tests)  #TODO: change back to `low=1` when core_move-bug is fixed
	dimensions = [random.randint(low=1, high=max_dimension, size=order) for order in orders]
	# A perfect binary tree with `order` leaves has depth `log2(order)` and `sum(2**k for k in range(log2(order))) = 2*order-1` nodes.
	# With exception of the root every node has one link to its parent. This results in `2*order - 2` links.
	ranks = [random.randint(low=1, high=max_rank, size=2*(order-1)) for order in orders]
	for dim, rk in zip(dimensions, ranks):
		yield xe.HTTensor.random(dim.tolist(), rk.tolist())


list2str = lambda ls: "-".join(map(str, ls))


def __test_tensor(A):
	name = "test_pickle_{}".format(list2str(A.dimensions))

	def test_pickle(self):
		bytes = dumps(A)
		Au = loads(bytes)
		self.assertEqual(Au.dimensions, A.dimensions)
		error = xe.frob_norm(A-Au) / xe.frob_norm(A)
		self.assertLessEqual(error, 1e-16)

	return name, test_pickle


def __test_tttensor(A):
	name = "test_pickle_{}_{}".format(list2str(A.dimensions), list2str(A.ranks()))

	def test_pickle(self):
		bytes = dumps(A)
		Au = loads(bytes)
		self.assertEqual(Au.dimensions, A.dimensions)
		self.assertEqual(Au.ranks(), A.ranks())
		error = xe.frob_norm(A-Au) / xe.frob_norm(A)
		self.assertLessEqual(error, 1e-10)

	return name, test_pickle


def __test_httensor(A):
	name = "test_pickle_{}_{}".format(list2str(A.dimensions), list2str(A.ranks()))

	def test_pickle(self):
		bytes = dumps(A)
		Au = loads(bytes)
		self.assertEqual(Au.dimensions, A.dimensions)
		self.assertEqual(Au.ranks(), A.ranks())
		error = xe.frob_norm(A-Au) / xe.frob_norm(A)
		self.assertLessEqual(error, 1e-10)

	return name, test_pickle


def __test_tensornetwork(A):
	name = "test_pickle_{}_{}".format(list2str(A.dimensions), list2str(A.ranks()))

	A = xe.TensorNetwork(A)
	i, = xe.indices(1)
	def test_pickle(self):
		bytes = dumps(A)
		Au = loads(bytes)
		normA = xe.frob_norm(A)
		normAu = xe.frob_norm(Au)
		innerAuA = float(A(i&0) * Au(i&0))
		error = np.sqrt(max(normA**2 - 2*innerAuA + normAu**2, 0)) / normA
		self.assertLessEqual(error, 1e-7)

	return name, test_pickle


def build_TestPickleTensor(seed, num_tests):
	random = np.random.RandomState(seed)
	odir = dict(__test_tensor(t) for t in generate_random_tensors(num_tests, random=random))
	return type("TestPickleTensor", (unittest.TestCase,), odir)


def build_TestPickleTTTensor(seed, num_tests):
	random = np.random.RandomState(seed)
	odir = dict(__test_tttensor(t) for t in generate_random_tttensors(num_tests, random=random))
	return type("TestPickleTTTensor", (unittest.TestCase,), odir)


def build_TestPickleHTTensor(seed, num_tests):
	random = np.random.RandomState(seed)
	odir = dict(__test_httensor(t) for t in generate_random_httensors(num_tests, random=random))
	return type("TestPickleHTTensor", (unittest.TestCase,), odir)


def build_TestPickleTensorNetwork(seed, num_tests):
	random = np.random.RandomState(seed)
	odir = dict(__test_tensornetwork(t) for t in generate_random_tttensors(num_tests, random=random))
	return type("TestPickleTensorNetwork", (unittest.TestCase,), odir)


TestPickleTensor = build_TestPickleTensor(0, 20)
TestPickleTTTensor = build_TestPickleTTTensor(0, 20)
TestPickleHTTensor = build_TestPickleHTTensor(0, 20)
TestPickleTensor = build_TestPickleTensorNetwork(0, 20)
